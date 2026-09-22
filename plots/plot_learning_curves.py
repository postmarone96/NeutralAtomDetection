"""Learning-curve plot for the site models."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(history, output_path, title=None, best_epoch=None):
    if not history:
        raise ValueError("cannot plot learning curves from empty history")

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

    accuracy_ax = loss_ax.twinx()
    accuracy_ax.plot(
        epochs,
        train_accuracy,
        label="train accuracy",
        color="tab:green",
        linestyle="--",
    )
    accuracy_ax.plot(
        epochs,
        val_accuracy,
        label="val accuracy",
        color="tab:red",
        linestyle="--",
    )
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_ylim(0.0, 1.0)

    if best_epoch is not None and best_epoch > 0:
        loss_ax.axvline(
            best_epoch,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"best epoch {best_epoch}",
        )

    loss_handles, loss_labels = loss_ax.get_legend_handles_labels()
    accuracy_handles, accuracy_labels = accuracy_ax.get_legend_handles_labels()
    loss_ax.legend(
        loss_handles + accuracy_handles,
        loss_labels + accuracy_labels,
        loc="best",
        fontsize=9,
    )

    if title is not None:
        loss_ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path
