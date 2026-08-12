from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cnn_site import (
    CNN_SITE_DECISION_THRESHOLD,
    compute_cnn_preprocessing,
    evaluate_loss_accuracy,
    make_random_split,
    make_site_loader,
    set_random_seed,
)
from mf_site import classification_fidelity
from utils import H5Dataset, extract_patches, get_device


DEFAULT_DATA_DIR = Path("dataset_3x3")
DEFAULT_MODEL_DIR = Path("models_3x3")
DEFAULT_OUTPUT_DIR = Path("output_3x3")
LIGHTWEIGHT_CNN_SITE_METHOD = "lightweight_cnn_site"
LIGHTWEIGHT_CNN_SITE_ARCHITECTURE = "tiny_cnn_site_3conv_pool_8fc"
CROP_SIZE = 10
LEARNING_RATE = 1e-4
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 40
EARLY_STOPPING_MIN_DELTA = 1e-5
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 2048
NUM_WORKERS = 4
BASE_SEED = 42
DEVICE = "cuda"


class LightweightCnnSiteNet(nn.Module):
    """
    Hardware-friendly CNN-site model for one 20x20 site crop.

    20x20x1
      Conv3x3,8,pad1 -> ReLU -> MaxPool2x2
      Conv3x3,16,pad1 -> ReLU -> MaxPool2x2
      Conv3x3,32,pad1 -> ReLU
      Flatten(5*5*32) -> FC8 -> ReLU -> FC2
    """
    def __init__(self, crop_size=20):
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
        self.classifier = nn.Sequential(
            nn.Linear(32 * pooled_side * pooled_side, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_lightweight_cnn_site_model(best, device="cpu"):
    model = LightweightCnnSiteNet(
        crop_size=int(best.get("crop_size", CROP_SIZE))
    ).to(device)
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def calc_lightweight_cnn_site_scores_from_site_loader(
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


@torch.no_grad()
def calc_lightweight_cnn_site_scores(dataloader, best, device="cpu"):
    model = load_lightweight_cnn_site_model(best, device=device)
    crop_size = int(best.get("crop_size", CROP_SIZE))

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


def train_lightweight_cnn_site(
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
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
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
    model = LightweightCnnSiteNet(crop_size=crop_size).to(device)
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
        desc=f"{Path(h5_path).stem} tiny CNN-site",
        unit="epoch",
        leave=False,
    )
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
        raise RuntimeError("tiny CNN-site training did not produce a best checkpoint")

    stopped_epoch = int(history[-1]["epoch"])

    model.load_state_dict(best_state)
    val_scores, val_labels = calc_lightweight_cnn_site_scores_from_site_loader(
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
        "method": LIGHTWEIGHT_CNN_SITE_METHOD,
        "architecture": LIGHTWEIGHT_CNN_SITE_ARCHITECTURE,
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


def evaluate_lightweight_cnn_site_model(dataloader, best, device="cpu"):
    scores, labels = calc_lightweight_cnn_site_scores(dataloader, best, device=device)
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
    early_stopping_patience,
    early_stopping_min_delta,
):
    with h5py.File(h5_path, "r") as f:
        n_samples = f["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )
    set_random_seed(split_seed, device=device)

    mean_image, scale = compute_cnn_preprocessing(h5_path, train_idx)

    trained = train_lightweight_cnn_site(
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
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
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

    test_metrics = evaluate_lightweight_cnn_site_model(
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
        "stopped_epoch": int(trained["stopped_epoch"]),
        "stopped_early": bool(trained["stopped_early"]),
        "epochs_trained": int(trained["epochs_trained"]),
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
    output_dir=DEFAULT_OUTPUT_DIR,
    crop_size=CROP_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    base_seed=BASE_SEED,
    device=DEVICE,
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

    for h5_path in h5_paths:
        output_path = model_dir / f"{h5_path.stem}_{LIGHTWEIGHT_CNN_SITE_METHOD}.joblib"

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
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        trained = result["trained_model"]

        save_obj = {
            **trained,
            "method": LIGHTWEIGHT_CNN_SITE_METHOD,
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
            "test_raw_accuracy": result["test_raw_accuracy"],
            "test_errors_per_site": result["test_errors_per_site"],
            "test_total_errors": result["test_total_errors"],
            "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
            "architecture": LIGHTWEIGHT_CNN_SITE_ARCHITECTURE,
        }

        joblib.dump(save_obj, output_path)

        learning_curve_path = None
        if save_learning_curves:
            learning_curve_path = learning_curve_dir / (
                f"{h5_path.stem}_{LIGHTWEIGHT_CNN_SITE_METHOD}_learning_curve.png"
            )
            try:
                plot_learning_curves(
                    history=trained["history"],
                    output_path=learning_curve_path,
                    title=f"Tiny CNN-site learning curve: {h5_path.stem}",
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
