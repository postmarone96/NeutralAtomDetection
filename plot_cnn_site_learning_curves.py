import csv
import pickle
import re
from pathlib import Path

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


model_dir = Path("models_3x3")
output_dir = Path("output_3x3/cnn_site_learning_curves")
summary_csv_path = output_dir / "cnn_site_learning_curve_summary.csv"

model_re = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")
model_patterns = (
    "*cnn_site*.joblib",
)


def unique_model_paths(model_dir, patterns):
    paths = []
    seen = set()
    for pattern in patterns:
        for path in sorted(model_dir.glob(pattern)):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def parse_model_name(path):
    match = model_re.fullmatch(path.name)
    if match is None:
        return None
    exposure_s = float(match.group(1))
    return exposure_s, 1000.0 * exposure_s, match.group(2)


def get_history(model_object):
    if "history" in model_object:
        return model_object["history"]
    if "trained_model" in model_object and "history" in model_object["trained_model"]:
        return model_object["trained_model"]["history"]
    return None


def get_best_epoch(model_object, history):
    if "best_epoch" in model_object:
        return int(model_object["best_epoch"])
    if "trained_model" in model_object and "best_epoch" in model_object["trained_model"]:
        return int(model_object["trained_model"]["best_epoch"])
    if history:
        val_losses = np.asarray([entry["val_loss"] for entry in history], dtype=float)
        return int(np.argmin(val_losses) + 1)
    return None


def get_best_val_loss(model_object, history):
    if "best_val_loss" in model_object:
        return float(model_object["best_val_loss"])
    if "trained_model" in model_object and "best_val_loss" in model_object["trained_model"]:
        return float(model_object["trained_model"]["best_val_loss"])
    if history:
        return float(np.min([entry["val_loss"] for entry in history]))
    return None


def plot_learning_curve(
    history,
    output_path,
    title,
    best_epoch=None,
):
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    val_accuracy = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axes[0].plot(epochs, train_loss, label="train loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="val loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, label="train accuracy", linewidth=2)
    axes[1].plot(epochs, val_accuracy, label="val accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(
                best_epoch,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label=f"best epoch {best_epoch}",
            )

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_summary(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "method",
        "architecture",
        "exposure_s",
        "exposure_ms",
        "crop_size",
        "epochs_recorded",
        "best_epoch",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "final_train_accuracy",
        "final_val_accuracy",
        "plot_path",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(
    model_dir=model_dir,
    output_dir=output_dir,
    summary_csv_path=summary_csv_path,
    model_patterns=model_patterns,
):
    rows = []

    for model_path in unique_model_paths(model_dir, model_patterns):
        parsed = parse_model_name(model_path)
        if parsed is None:
            continue
        exposure_s, exposure_ms, method = parsed

        try:
            model_object = joblib.load(model_path)
        except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
            print(f"Skipping {model_path.name}: could not load ({exc})")
            continue

        history = get_history(model_object)
        if not history:
            print(f"Skipping {model_path.name}: no history")
            continue

        best_epoch = get_best_epoch(model_object, history)
        best_val_loss = get_best_val_loss(model_object, history)
        architecture = str(model_object.get("architecture", "unknown"))
        crop_size = model_object.get("crop_size", "")
        output_path = output_dir / f"{model_path.stem}_learning_curve.png"

        plot_learning_curve(
            history=history,
            output_path=output_path,
            title=(
                f"{method}, {exposure_ms:g} ms"
                f" | crop={crop_size}"
                f" | {architecture}"
            ),
            best_epoch=best_epoch,
        )

        final = history[-1]
        rows.append(
            {
                "model": model_path.name,
                "method": method,
                "architecture": architecture,
                "exposure_s": exposure_s,
                "exposure_ms": exposure_ms,
                "crop_size": crop_size,
                "epochs_recorded": len(history),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "final_train_loss": float(final["train_loss"]),
                "final_val_loss": float(final["val_loss"]),
                "final_train_accuracy": float(final["train_accuracy"]),
                "final_val_accuracy": float(final["val_accuracy"]),
                "plot_path": str(output_path),
            }
        )

        print(f"Saved learning curve: {output_path}")

    save_summary(rows, summary_csv_path)
    print(f"Saved summary: {summary_csv_path}")
    print(f"Plotted {len(rows)} learning curves")


if __name__ == "__main__":
    main(
        model_dir=model_dir,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
        model_patterns=model_patterns,
    )
