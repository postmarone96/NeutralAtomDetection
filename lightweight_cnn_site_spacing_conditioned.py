from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cnn_site import (
    CNN_SITE_DECISION_THRESHOLD,
    compute_cnn_preprocessing,
    extract_site_crop,
    set_random_seed,
)
from mf_site import classification_fidelity
from utils import get_device


DEFAULT_DATA_DIR = Path("spacing_datasets_3x3")
DEFAULT_MODEL_DIR = Path("models_3x3")
DEFAULT_OUTPUT_DIR = Path("output_3x3")

SPACING_CONDITIONED_METHOD = "lightweight_cnn_site_spacing_conditioned"
POOLED_METHOD = "lightweight_cnn_site_pooled"
SPACING_CONDITIONED_ARCHITECTURE = "tiny_cnn_site_fc8_spacing_fc8_fc2"
POOLED_ARCHITECTURE = "tiny_cnn_site_fc8_fc8_fc2"

CROP_SIZE = 20
LEARNING_RATE = 1e-4
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-5
BATCH_SIZE = 4096
EVAL_BATCH_SIZE = 16384
NUM_WORKERS = 20
BASE_SEED = 42
DEVICE = "cuda:1"

SPACING_CENTER_PX = 16.0
SPACING_SCALE_PX = 12.0
SELECTED_SPACINGS_PX = None
CONDITION_ON_SPACING = True


def normalize_spacing_px(
    spacing_px,
    center_px=SPACING_CENTER_PX,
    scale_px=SPACING_SCALE_PX,
):
    return (np.asarray(spacing_px, dtype=np.float32) - center_px) / scale_px


def spacing_normalization_metadata(
    spacing_values_px=None,
    center_px=SPACING_CENTER_PX,
    scale_px=SPACING_SCALE_PX,
):
    metadata = {
        "formula": "(spacing_px - center_px) / scale_px",
        "center_px": float(center_px),
        "scale_px": float(scale_px),
        "normalized_min": float((4.0 - center_px) / scale_px),
        "normalized_max": float((28.0 - center_px) / scale_px),
    }
    if spacing_values_px is not None:
        spacing_values_px = np.asarray(spacing_values_px, dtype=np.float32)
        metadata["spacing_values_px"] = spacing_values_px.copy()
        metadata["source_min_px"] = float(np.min(spacing_values_px))
        metadata["source_max_px"] = float(np.max(spacing_values_px))
    return metadata


def make_spacing_balanced_split(
    spacing_px,
    seed,
    train_fraction=0.60,
    val_fraction=0.20,
    selected_spacings_px=None,
):
    spacing_px = np.asarray(spacing_px, dtype=np.float32)
    rng = np.random.default_rng(seed)

    if selected_spacings_px is None:
        spacing_values = np.sort(np.unique(spacing_px))
    else:
        spacing_values = np.asarray(selected_spacings_px, dtype=np.float32)

    train_parts = []
    val_parts = []
    test_parts = []

    for spacing_value in spacing_values:
        spacing_indices = np.flatnonzero(np.isclose(spacing_px, spacing_value))
        if spacing_indices.size < 5:
            raise ValueError(
                f"At least 5 samples are required for spacing {spacing_value:g} px, "
                f"got {spacing_indices.size}"
            )

        spacing_indices = rng.permutation(spacing_indices)
        n_train = int(train_fraction * spacing_indices.size)
        n_val = int(val_fraction * spacing_indices.size)

        train_parts.append(spacing_indices[:n_train])
        val_parts.append(spacing_indices[n_train:n_train + n_val])
        test_parts.append(spacing_indices[n_train + n_val:])

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))

    return train_idx, val_idx, test_idx


class SpacingCnnSiteDataset(Dataset):
    """
    Reads one full image and returns all per-site 20x20 crops.

    The HDF5 schema is:
      images:     (N, H, W)
      labels:     (N, 9)
      sites:      (N, 9, 2)
      spacing_px: (N,)
    """
    def __init__(
        self,
        h5_path,
        indices,
        crop_size=CROP_SIZE,
        image_key="images",
        label_key="labels",
        site_key="sites",
        spacing_key="spacing_px",
        mean_image=None,
        scale=None,
        spacing_center_px=SPACING_CENTER_PX,
        spacing_scale_px=SPACING_SCALE_PX,
    ):
        self.h5_path = h5_path
        self.indices = list(indices)
        self.crop_size = int(crop_size)
        self.image_key = image_key
        self.label_key = label_key
        self.site_key = site_key
        self.spacing_key = spacing_key
        self.mean_image = None if mean_image is None else np.asarray(mean_image, dtype=np.float32)
        self.scale = None if scale is None else float(scale)
        self.spacing_center_px = float(spacing_center_px)
        self.spacing_scale_px = float(spacing_scale_px)
        self._h5 = None

        if self.crop_size != 20:
            raise ValueError("spacing-conditioned lightweight CNN-site uses fixed 20x20 crops")

        with h5py.File(h5_path, "r") as f:
            image_shape = f[image_key].shape
            label_shape = f[label_key].shape
            site_shape = f[site_key].shape
            spacing_shape = f[spacing_key].shape

        if len(image_shape) != 3:
            raise ValueError(f"expected images shape (N,H,W), got {image_shape}")
        if len(label_shape) != 2:
            raise ValueError(f"expected labels shape (N,Q), got {label_shape}")
        if len(site_shape) != 3 or site_shape[2] != 2:
            raise ValueError(f"expected sites shape (N,Q,2), got {site_shape}")
        if len(spacing_shape) != 1:
            raise ValueError(f"expected spacing_px shape (N,), got {spacing_shape}")
        if image_shape[0] != label_shape[0] or image_shape[0] != site_shape[0]:
            raise ValueError("images, labels, and sites must have the same first dimension")
        if image_shape[0] != spacing_shape[0]:
            raise ValueError("images and spacing_px must have the same first dimension")
        if label_shape[1] != site_shape[1]:
            raise ValueError("labels and sites must have the same site count")

        self.n_images = int(image_shape[0])
        self.image_shape = tuple(image_shape[1:])
        self.n_sites = int(label_shape[1])

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, idx):
        file_index = int(self.indices[idx])

        f = self._file()
        image = f[self.image_key][file_index].astype(np.float32)
        if self.mean_image is not None:
            image = image - self.mean_image
        if self.scale is not None:
            image = image / self.scale

        sites = f[self.site_key][file_index].astype(np.float32)
        crops = np.stack(
            [
                extract_site_crop(
                    image=image,
                    site=site,
                    crop_size=self.crop_size,
                )
                for site in sites
            ],
            axis=0,
        )

        labels = (f[self.label_key][file_index] > 0.5).astype(np.int64)
        spacing_px = np.float32(f[self.spacing_key][file_index])
        spacing_norm = np.float32(
            (spacing_px - self.spacing_center_px) / self.spacing_scale_px
        )
        spacing_norm_sites = np.full(self.n_sites, spacing_norm, dtype=np.float32)
        spacing_px_sites = np.full(self.n_sites, spacing_px, dtype=np.float32)

        return (
            torch.from_numpy(crops[:, None, :, :]).float(),
            torch.from_numpy(labels).long(),
            torch.from_numpy(spacing_norm_sites).float(),
            torch.from_numpy(spacing_px_sites).float(),
        )


class LightweightCnnSiteEncoder(nn.Module):
    def __init__(self, crop_size=CROP_SIZE, embedding_dim=8):
        super().__init__()
        crop_size = int(crop_size)
        pooled_side = crop_size // 4
        if pooled_side <= 0:
            raise ValueError("crop_size must be at least 4")

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(32 * pooled_side * pooled_side, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.embedding(self.features(x))


class PooledLightweightCnnSiteNet(nn.Module):
    def __init__(self, crop_size=CROP_SIZE):
        super().__init__()
        self.encoder = LightweightCnnSiteEncoder(crop_size=crop_size, embedding_dim=8)
        self.classifier = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x, spacing_norm=None):
        embedding = self.encoder(x)
        return self.classifier(embedding)


class SpacingConditionedLightweightCnnSiteNet(nn.Module):
    def __init__(self, crop_size=CROP_SIZE):
        super().__init__()
        self.encoder = LightweightCnnSiteEncoder(crop_size=crop_size, embedding_dim=8)
        self.classifier = nn.Sequential(
            nn.Linear(9, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x, spacing_norm):
        if spacing_norm is None:
            raise ValueError("spacing-conditioned model requires spacing_norm")
        if spacing_norm.ndim == 1:
            spacing_norm = spacing_norm[:, None]
        embedding = self.encoder(x)
        return self.classifier(torch.cat([embedding, spacing_norm.float()], dim=1))


def model_method(condition_on_spacing):
    return SPACING_CONDITIONED_METHOD if condition_on_spacing else POOLED_METHOD


def model_architecture(condition_on_spacing):
    return SPACING_CONDITIONED_ARCHITECTURE if condition_on_spacing else POOLED_ARCHITECTURE


def build_model(condition_on_spacing, crop_size=CROP_SIZE, device="cpu"):
    if condition_on_spacing:
        model = SpacingConditionedLightweightCnnSiteNet(crop_size=crop_size)
    else:
        model = PooledLightweightCnnSiteNet(crop_size=crop_size)
    return model.to(device)


def load_lightweight_spacing_cnn_site_model(best, device="cpu"):
    condition_on_spacing = bool(best.get("condition_on_spacing", True))
    model = build_model(
        condition_on_spacing=condition_on_spacing,
        crop_size=int(best.get("crop_size", CROP_SIZE)),
        device=device,
    )
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


def make_spacing_site_loader(
    h5_path,
    indices,
    crop_size,
    mean_image,
    scale,
    batch_size,
    num_workers,
    shuffle,
    pin_memory=False,
    spacing_center_px=SPACING_CENTER_PX,
    spacing_scale_px=SPACING_SCALE_PX,
):
    dataset = SpacingCnnSiteDataset(
        h5_path=h5_path,
        indices=indices,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        spacing_center_px=spacing_center_px,
        spacing_scale_px=spacing_scale_px,
    )
    image_batch_size = max(1, batch_size // dataset.n_sites)

    return DataLoader(
        dataset,
        batch_size=image_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate_loss_accuracy(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0

    for crops, labels, spacing_norm, _spacing_px in dataloader:
        crops = crops.to(device)
        labels = labels.to(device)
        spacing_norm = spacing_norm.to(device)

        batch, n_sites = labels.shape
        crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
        labels = labels.reshape(batch * n_sites)
        spacing_norm = spacing_norm.reshape(batch * n_sites)

        logits = model(crops, spacing_norm)
        loss = loss_fn(logits, labels)

        batch_examples = int(labels.numel())
        total_loss += float(loss.item()) * batch_examples
        total_examples += batch_examples
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def calc_lightweight_spacing_cnn_site_scores_from_loader(
    dataloader,
    model,
    device="cpu",
):
    model.eval()
    all_scores = []
    all_labels = []
    all_spacing_px = []

    for crops, labels, spacing_norm, spacing_px in dataloader:
        batch, n_sites = labels.shape
        crops = crops.to(device)
        spacing_norm = spacing_norm.to(device)

        crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
        spacing_norm = spacing_norm.reshape(batch * n_sites)

        probabilities = torch.softmax(model(crops, spacing_norm), dim=1)[:, 1]
        scores = probabilities.reshape(batch, n_sites)

        all_scores.append(scores.cpu())
        all_labels.append(labels.float())
        all_spacing_px.append(spacing_px.float())

    return (
        torch.cat(all_scores).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_spacing_px).numpy(),
    )


@torch.no_grad()
def calc_lightweight_spacing_cnn_site_scores(dataloader, best, device="cpu"):
    model = load_lightweight_spacing_cnn_site_model(best, device=device)
    return calc_lightweight_spacing_cnn_site_scores_from_loader(
        dataloader=dataloader,
        model=model,
        device=device,
    )


def train_lightweight_spacing_cnn_site(
    h5_path,
    train_idx,
    val_idx,
    crop_size=CROP_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device="cpu",
    mean_image=None,
    scale=None,
    condition_on_spacing=CONDITION_ON_SPACING,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
):
    train_loader = make_spacing_site_loader(
        h5_path=h5_path,
        indices=train_idx,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=str(device).startswith("cuda"),
    )
    val_loader = make_spacing_site_loader(
        h5_path=h5_path,
        indices=val_idx,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
    )

    n_sites = train_loader.dataset.n_sites
    model = build_model(
        condition_on_spacing=condition_on_spacing,
        crop_size=crop_size,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_state = None
    best_val_loss = np.inf
    best_epoch = -1
    no_improvement_epochs = 0
    stopped_early = False

    progress = tqdm(
        range(epochs),
        desc=f"{Path(h5_path).stem} {model_method(condition_on_spacing)}",
        unit="epoch",
        leave=True,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        train_correct = 0

        for crops, labels, spacing_norm, _spacing_px in train_loader:
            crops = crops.to(device)
            labels = labels.to(device)
            spacing_norm = spacing_norm.to(device)

            batch, n_sites = labels.shape
            crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
            labels = labels.reshape(batch * n_sites)
            spacing_norm = spacing_norm.reshape(batch * n_sites)

            optimizer.zero_grad(set_to_none=True)
            logits = model(crops, spacing_norm)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_examples = int(labels.numel())
            train_loss_sum += float(loss.item()) * batch_examples
            train_examples += batch_examples
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = train_loss_sum / max(train_examples, 1)
        train_accuracy = train_correct / max(train_examples, 1)
        val_metrics = evaluate_loss_accuracy(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
        )

        if val_metrics["loss"] < best_val_loss - early_stopping_min_delta:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            no_improvement_epochs += 1

        progress.set_postfix(
            train_loss=f"{train_loss:.4g}",
            val_loss=f"{val_metrics['loss']:.4g}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best=best_epoch,
        )

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and no_improvement_epochs >= early_stopping_patience
        ):
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")

    stopped_epoch = int(history[-1]["epoch"])
    model.load_state_dict(best_state)

    val_scores, val_labels, _val_spacing_px = calc_lightweight_spacing_cnn_site_scores_from_loader(
        dataloader=val_loader,
        model=model,
        device=device,
    )

    thresholds = np.full(n_sites, CNN_SITE_DECISION_THRESHOLD, dtype=np.float32)
    fidelities = np.zeros(n_sites, dtype=np.float32)
    for q in range(n_sites):
        fidelities[q] = classification_fidelity(
            val_scores[:, q],
            val_labels[:, q],
            CNN_SITE_DECISION_THRESHOLD,
        )

    condition_on_spacing = bool(condition_on_spacing)
    return {
        "method": model_method(condition_on_spacing),
        "architecture": model_architecture(condition_on_spacing),
        "condition_on_spacing": condition_on_spacing,
        "state_dict": best_state,
        "crop_size": int(crop_size),
        "threshold": thresholds,
        "val_fidelity": fidelities,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "stopped_epoch": stopped_epoch,
        "stopped_early": bool(stopped_early),
        "epochs_trained": int(len(history)),
        "early_stopping_patience": (
            None
            if early_stopping_patience is None
            else int(early_stopping_patience)
        ),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "spacing_normalization": spacing_normalization_metadata(),
    }


def fidelity_by_spacing(scores, labels, spacing_px, thresholds):
    spacing_per_image = spacing_px[:, 0]
    result = {}
    for spacing_value in np.sort(np.unique(spacing_per_image)):
        image_mask = np.isclose(spacing_per_image, spacing_value)
        site_fidelities = np.zeros(labels.shape[1], dtype=np.float32)
        for q in range(labels.shape[1]):
            site_fidelities[q] = classification_fidelity(
                scores[image_mask, q],
                labels[image_mask, q],
                thresholds[q],
            )
        key = f"{float(spacing_value):g}"
        result[key] = {
            "site_fidelity": site_fidelities,
            "mean_fidelity": float(np.mean(site_fidelities)),
            "n_images": int(np.count_nonzero(image_mask)),
        }
    return result


def evaluate_lightweight_spacing_cnn_site_model(dataloader, best, device="cpu"):
    scores, labels, spacing_px = calc_lightweight_spacing_cnn_site_scores(
        dataloader,
        best,
        device=device,
    )
    thresholds = np.asarray(best["threshold"])

    fids = np.zeros(labels.shape[1], dtype=np.float32)
    for q in range(labels.shape[1]):
        fids[q] = classification_fidelity(
            scores[:, q],
            labels[:, q],
            thresholds[q],
        )

    preds = scores >= thresholds[None, :]
    errors = preds != (labels > 0.5)

    return {
        "scores": scores,
        "labels": labels,
        "spacing_px": spacing_px,
        "fidelity": fids,
        "mean_fidelity": float(np.mean(fids)),
        "fidelity_by_spacing": fidelity_by_spacing(
            scores=scores,
            labels=labels,
            spacing_px=spacing_px,
            thresholds=thresholds,
        ),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


def plot_learning_curves(history, output_path, title=None, best_epoch=None):
    if not history:
        raise ValueError("cannot plot learning curves from empty history")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_accuracy = [item["train_accuracy"] for item in history]
    val_accuracy = [item["val_accuracy"] for item in history]

    fig, loss_ax = plt.subplots(figsize=(7.5, 4.8))
    loss_ax.plot(epochs, train_loss, label="train loss", color="tab:blue")
    loss_ax.plot(epochs, val_loss, label="val loss", color="tab:orange")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Cross-entropy loss")
    loss_ax.grid(True, alpha=0.3)

    acc_ax = loss_ax.twinx()
    acc_ax.plot(
        epochs,
        train_accuracy,
        label="train accuracy",
        color="tab:green",
        linestyle="--",
    )
    acc_ax.plot(
        epochs,
        val_accuracy,
        label="val accuracy",
        color="tab:red",
        linestyle="--",
    )
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_ylim(0.0, 1.0)

    if best_epoch is not None and best_epoch > 0:
        loss_ax.axvline(
            best_epoch,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"best epoch {best_epoch}",
        )

    handles = loss_ax.get_legend_handles_labels()
    acc_handles = acc_ax.get_legend_handles_labels()
    loss_ax.legend(
        handles[0] + acc_handles[0],
        handles[1] + acc_handles[1],
        loc="best",
        fontsize=9,
    )

    if title is not None:
        loss_ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def read_spacing_values(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "spacing_values_px" in f:
            return f["spacing_values_px"][:].astype(np.float32)
        return np.sort(np.unique(f["spacing_px"][:].astype(np.float32)))


def run_one_split(
    h5_path,
    split_seed,
    crop_size,
    learning_rate,
    epochs,
    batch_size,
    eval_batch_size,
    num_workers,
    device,
    condition_on_spacing,
    selected_spacings_px,
    early_stopping_patience,
    early_stopping_min_delta,
):
    with h5py.File(h5_path, "r") as f:
        spacing_px = f["spacing_px"][:].astype(np.float32)
        spacing_values_px = (
            f["spacing_values_px"][:].astype(np.float32)
            if "spacing_values_px" in f
            else np.sort(np.unique(spacing_px))
        )

    train_idx, val_idx, test_idx = make_spacing_balanced_split(
        spacing_px=spacing_px,
        seed=split_seed,
        selected_spacings_px=selected_spacings_px,
    )
    set_random_seed(split_seed, device=device)

    mean_image, scale = compute_cnn_preprocessing(h5_path, train_idx)

    trained = train_lightweight_spacing_cnn_site(
        h5_path=h5_path,
        train_idx=train_idx,
        val_idx=val_idx,
        crop_size=crop_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        mean_image=mean_image,
        scale=scale,
        condition_on_spacing=condition_on_spacing,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )

    trained["spacing_normalization"] = spacing_normalization_metadata(
        spacing_values_px=spacing_values_px,
    )

    test_loader = make_spacing_site_loader(
        h5_path=h5_path,
        indices=test_idx,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=str(device).startswith("cuda"),
    )

    test_metrics = evaluate_lightweight_spacing_cnn_site_model(
        test_loader,
        trained,
        device=device,
    )

    return {
        "seed": split_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean_image": mean_image,
        "scale": scale,
        "spacing_values_px": spacing_values_px,
        "selected_spacings_px": (
            None
            if selected_spacings_px is None
            else np.asarray(selected_spacings_px, dtype=np.float32)
        ),
        "best_epoch": int(trained["best_epoch"]),
        "best_val_loss": float(trained["best_val_loss"]),
        "stopped_epoch": int(trained["stopped_epoch"]),
        "stopped_early": bool(trained["stopped_early"]),
        "epochs_trained": int(trained["epochs_trained"]),
        "thresholds": np.asarray(trained["threshold"]).copy(),
        "val_fidelity": np.asarray(trained["val_fidelity"]).copy(),
        "test_fidelity": np.asarray(test_metrics["fidelity"]).copy(),
        "test_mean_fidelity": float(test_metrics["mean_fidelity"]),
        "test_fidelity_by_spacing": test_metrics["fidelity_by_spacing"],
        "test_raw_accuracy": float(test_metrics["raw_accuracy"]),
        "test_errors_per_site": np.asarray(
            test_metrics["errors_per_site"]
        ).copy(),
        "test_total_errors": int(test_metrics["total_errors"]),
        "trained_model": trained,
    }


def main(
    data_dir=DEFAULT_DATA_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    crop_size=CROP_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    base_seed=BASE_SEED,
    device=DEVICE,
    condition_on_spacing=CONDITION_ON_SPACING,
    selected_spacings_px=SELECTED_SPACINGS_PX,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
    save_learning_curves=True,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    if save_learning_curves:
        learning_curve_dir = output_dir / "learning_curves"
        learning_curve_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no .h5 datasets found in {data_dir}")

    device = device or get_device()
    condition_on_spacing = bool(condition_on_spacing)
    method = model_method(condition_on_spacing)
    architecture = model_architecture(condition_on_spacing)

    for h5_path in h5_paths:
        output_path = model_dir / f"{h5_path.stem}_{method}.joblib"

        with h5py.File(h5_path, "r") as f:
            n_samples = f["images"].shape[0]

        if n_samples < 5:
            print(f"Skipping {h5_path}: dataset has only {n_samples} images")
            continue

        result = run_one_split(
            h5_path=h5_path,
            split_seed=base_seed,
            crop_size=crop_size,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            device=device,
            condition_on_spacing=condition_on_spacing,
            selected_spacings_px=selected_spacings_px,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        trained = result["trained_model"]

        save_obj = {
            **trained,
            "method": method,
            "architecture": architecture,
            "condition_on_spacing": condition_on_spacing,
            "h5_path": str(h5_path),
            "seed": base_seed,
            "train_idx": result["train_idx"],
            "val_idx": result["val_idx"],
            "test_idx": result["test_idx"],
            "mean_image": result["mean_image"],
            "scale": result["scale"],
            "crop_size": int(crop_size),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "epochs_trained": result["epochs_trained"],
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "early_stopping_patience": (
                None
                if early_stopping_patience is None
                else int(early_stopping_patience)
            ),
            "early_stopping_min_delta": float(early_stopping_min_delta),
            "stopped_epoch": result["stopped_epoch"],
            "stopped_early": result["stopped_early"],
            "thresholds": result["thresholds"],
            "test_fidelity": result["test_fidelity"],
            "test_mean_fidelity": result["test_mean_fidelity"],
            "test_fidelity_by_spacing": result["test_fidelity_by_spacing"],
            "test_raw_accuracy": result["test_raw_accuracy"],
            "test_errors_per_site": result["test_errors_per_site"],
            "test_total_errors": result["test_total_errors"],
            "spacing_values_px": result["spacing_values_px"],
            "selected_spacings_px": result["selected_spacings_px"],
            "spacing_normalization": spacing_normalization_metadata(
                spacing_values_px=result["spacing_values_px"],
            ),
            "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
        }

        joblib.dump(save_obj, output_path)

        learning_curve_path = None
        if save_learning_curves:
            learning_curve_path = learning_curve_dir / (
                f"{h5_path.stem}_{method}_learning_curve.png"
            )
            try:
                plot_learning_curves(
                    history=trained["history"],
                    output_path=learning_curve_path,
                    title=f"{method} learning curve: {h5_path.stem}",
                    best_epoch=result["best_epoch"],
                )
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                print(f"Skipping learning curve for {h5_path.name}: {exc}")
                learning_curve_path = None

        print(
            f"Saved {output_path} | "
            f"mean F={result['test_mean_fidelity']:.8f} | "
            f"best epoch={result['best_epoch']} | "
            f"trained epochs={result['epochs_trained']}"
        )
        if learning_curve_path is not None:
            print(f"Saved learning curve: {learning_curve_path}")


if __name__ == "__main__":
    main(
        data_dir=DEFAULT_DATA_DIR,
        model_dir=DEFAULT_MODEL_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    )
