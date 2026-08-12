from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from mf_site import classification_fidelity
from utils import H5Dataset, extract_patches, get_device


DEFAULT_DATA_DIR = Path(
    "dataset_3x3"
)
DEFAULT_MODEL_DIR = Path("models_3x3")
CNN_SITE_DECISION_THRESHOLD = 0.5
CROP_SIZE = 10
LEARNING_RATE = 1e-4
EPOCHS = 40
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 2048
NUM_WORKERS = 2
BASE_SEED = 42
DEVICE = "cuda"


def make_random_split(
    n_samples: int,
    seed: int,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
):
    if n_samples < 5:
        raise ValueError("At least 5 samples are required.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def set_random_seed(seed, device="cpu"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def compute_cnn_preprocessing(h5_path, train_idx, image_key="images", chunk_size=64):
    """
    Paper-style CNN preprocessing:
    I' = (I - mean training pixel intensity) / (train max - train min).
    """
    if len(train_idx) == 0:
        raise ValueError(f"Cannot compute preprocessing for {h5_path}: train_idx is empty")

    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))

    with h5py.File(h5_path, "r") as f:
        images = f[image_key]
        total = 0.0
        count = 0
        min_value = np.inf
        max_value = -np.inf

        for start in range(0, len(train_idx), chunk_size):
            chunk_idx = train_idx[start:start + chunk_size]
            image_chunk = images[chunk_idx].astype(np.float32)
            total += float(image_chunk.sum(dtype=np.float64))
            count += int(image_chunk.size)
            min_value = min(min_value, float(image_chunk.min()))
            max_value = max(max_value, float(image_chunk.max()))

    mean_pixel = np.float32(total / count)
    scale = float(max_value - min_value)
    if scale == 0.0:
        scale = 1.0

    return mean_pixel, scale


def extract_site_crop(image, site, crop_size):
    height, width = image.shape
    half = crop_size // 2
    pad = half + 1

    padded = np.pad(
        image,
        pad_width=((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    center_x = int(np.rint(site[0])) + pad
    center_y = int(np.rint(site[1])) + pad

    x0 = center_x - half
    y0 = center_y - half
    crop = padded[y0:y0 + crop_size, x0:x0 + crop_size]

    if crop.shape != (crop_size, crop_size):
        raise ValueError(
            f"expected crop shape {(crop_size, crop_size)}, got {crop.shape} "
            f"from image shape {(height, width)}"
        )

    return crop.astype(np.float32, copy=False)


class CnnSiteDataset(Dataset):
    """
    Reads one full-array image and returns all single-site crops.

    This avoids reading the same HDF5 image once per site.
    """
    def __init__(
        self,
        h5_path,
        indices,
        crop_size=10,
        image_key="images",
        label_key="labels",
        site_key="sites",
        mean_image=None,
        scale=None,
    ):
        self.h5_path = h5_path
        self.indices = list(indices)
        self.crop_size = int(crop_size)
        self.image_key = image_key
        self.label_key = label_key
        self.site_key = site_key
        self.mean_image = None if mean_image is None else np.asarray(mean_image, dtype=np.float32)
        self.scale = None if scale is None else float(scale)
        self._h5 = None

        if self.crop_size <= 6:
            raise ValueError("CNN-site crop_size must be greater than 6 for three valid 3x3 conv layers")

        with h5py.File(h5_path, "r") as f:
            self.n_images = f[image_key].shape[0]
            self.sites = f[site_key][:].astype(np.float32)

        self.n_sites = int(self.sites.shape[0])

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

        crops = np.stack(
            [
                extract_site_crop(
                    image=image,
                    site=site,
                    crop_size=self.crop_size,
                )
                for site in self.sites
            ],
            axis=0,
        )
        labels = (f[self.label_key][file_index] > 0.5).astype(np.int64)

        return (
            torch.from_numpy(crops[:, None, :, :]).float(),
            torch.from_numpy(labels).long(),
        )


class CnnSiteNet(nn.Module):
    def __init__(self, crop_size=10):
        super().__init__()
        crop_size = int(crop_size)
        conv_side = crop_size - 6
        if conv_side <= 0:
            raise ValueError("crop_size must be greater than 6")

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * conv_side * conv_side, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_site_loader(
    h5_path,
    indices,
    crop_size,
    mean_image,
    scale,
    batch_size,
    num_workers,
    shuffle,
    pin_memory=False,
):
    dataset = CnnSiteDataset(
        h5_path=h5_path,
        indices=indices,
        crop_size=crop_size,
        mean_image=mean_image,
        scale=scale,
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

    for crops, labels in dataloader:
        crops = crops.to(device)
        labels = labels.to(device)

        batch, n_sites = labels.shape
        crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
        labels = labels.reshape(batch * n_sites)

        logits = model(crops)
        loss = loss_fn(logits, labels)

        batch = int(labels.numel())
        total_loss += float(loss.item()) * batch
        total_examples += batch
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def calc_cnn_site_scores_from_site_loader(
    dataloader,
    model,
    device="cpu",
):
    model.eval()
    all_scores = []
    all_labels = []

    for crops, labels in dataloader:
        batch, n_sites = labels.shape
        crops = crops.to(device)
        crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])

        probabilities = torch.softmax(model(crops), dim=1)[:, 1]
        scores = probabilities.reshape(batch, n_sites)

        all_scores.append(scores.cpu())
        all_labels.append(labels.float())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def load_cnn_site_model(best, device="cpu"):
    model = CnnSiteNet(crop_size=int(best.get("crop_size", 10))).to(device)
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def calc_cnn_site_scores(dataloader, best, device="cpu"):
    model = load_cnn_site_model(best, device=device)
    crop_size = int(best.get("crop_size", 10))

    all_scores = []
    all_labels = []

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()
        sites = sites[0].to(device)

        batch, n_sites = labels.shape
        patches = extract_patches(images, sites, crop_size)
        patches = patches.reshape(batch, n_sites, crop_size, crop_size)
        crops = patches.reshape(batch * n_sites, 1, crop_size, crop_size)

        probabilities = torch.softmax(model(crops), dim=1)[:, 1]
        scores = probabilities.reshape(batch, n_sites)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def train_cnn_site(
    h5_path,
    train_idx,
    val_idx,
    crop_size=10,
    learning_rate=1e-4,
    epochs=40,
    batch_size=64,
    num_workers=2,
    device="cpu",
    mean_image=None,
    scale=None,
):
    train_loader = make_site_loader(
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
    val_loader = make_site_loader(
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
    model = CnnSiteNet(crop_size=crop_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    best_state = None
    best_val_loss = np.inf
    best_epoch = -1

    progress = tqdm(range(epochs), desc=f"{Path(h5_path).stem} CNN-site", unit="epoch", leave=False)
    for epoch in progress:
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        train_correct = 0

        for crops, labels in train_loader:
            crops = crops.to(device)
            labels = labels.to(device)

            batch, n_sites = labels.shape
            crops = crops.reshape(batch * n_sites, 1, crops.shape[-2], crops.shape[-1])
            labels = labels.reshape(batch * n_sites)

            optimizer.zero_grad(set_to_none=True)
            logits = model(crops)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch = int(labels.numel())
            train_loss_sum += float(loss.item()) * batch
            train_examples += batch
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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        progress.set_postfix(
            train_loss=f"{train_loss:.4g}",
            val_loss=f"{val_metrics['loss']:.4g}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
        )

    if best_state is None:
        raise RuntimeError("CNN-site training did not produce a best checkpoint")

    model.load_state_dict(best_state)
    val_scores, val_labels = calc_cnn_site_scores_from_site_loader(
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

    return {
        "method": "cnn_site",
        "architecture": "cnn_site_3conv_128fc",
        "state_dict": best_state,
        "crop_size": int(crop_size),
        "threshold": thresholds,
        "val_fidelity": fidelities,
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    }


def evaluate_cnn_site_model(dataloader, best, device="cpu"):
    scores, labels = calc_cnn_site_scores(dataloader, best, device=device)
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
        "fidelity": fids,
        "mean_fidelity": float(np.mean(fids)),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


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
):
    with h5py.File(h5_path, "r") as f:
        n_samples = f["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )
    set_random_seed(split_seed, device=device)

    mean_image, scale = compute_cnn_preprocessing(h5_path, train_idx)

    trained = train_cnn_site(
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
    )

    test_loader = DataLoader(
        H5Dataset(
            h5_path,
            indices=test_idx,
            mean_image=mean_image,
            scale=scale,
        ),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    test_metrics = evaluate_cnn_site_model(
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
        "best_epoch": int(trained["best_epoch"]),
        "best_val_loss": float(trained["best_val_loss"]),
        "thresholds": np.asarray(trained["threshold"]).copy(),
        "val_fidelity": np.asarray(trained["val_fidelity"]).copy(),
        "test_fidelity": np.asarray(test_metrics["fidelity"]).copy(),
        "test_mean_fidelity": float(test_metrics["mean_fidelity"]),
        "test_raw_accuracy": float(test_metrics["raw_accuracy"]),
        "test_errors_per_site": np.asarray(
            test_metrics["errors_per_site"]
        ).copy(),
        "test_total_errors": int(test_metrics["total_errors"]),
        "trained_model": trained,
    }


def main(
    data_dir,
    model_dir,
    crop_size=CROP_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    base_seed=BASE_SEED,
    device=DEVICE,
):
    model_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no .h5 datasets found in {data_dir}")

    device = device or get_device()

    for h5_path in h5_paths:
        output_path = model_dir / f"{h5_path.stem}_cnn_site.joblib"

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
        )
        trained = result["trained_model"]

        save_obj = {
            **trained,
            "method": "cnn_site",
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
            "batch_size": int(batch_size),
            "eval_batch_size": int(eval_batch_size),
            "thresholds": result["thresholds"],
            "test_fidelity": result["test_fidelity"],
            "test_mean_fidelity": result["test_mean_fidelity"],
            "test_raw_accuracy": result["test_raw_accuracy"],
            "test_errors_per_site": result["test_errors_per_site"],
            "test_total_errors": result["test_total_errors"],
            "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
            "architecture": "cnn_site_3conv_128fc",
        }

        joblib.dump(save_obj, output_path)

        print(
            f"Saved {output_path} | "
            f"mean F={result['test_mean_fidelity']:.8f} | "
            f"best epoch={result['best_epoch']}"
        )


if __name__ == "__main__":
    main(
        data_dir=DEFAULT_DATA_DIR,
        model_dir=DEFAULT_MODEL_DIR,
    )
