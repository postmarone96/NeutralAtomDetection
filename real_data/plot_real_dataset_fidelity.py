import csv
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np

from plots.plot_fidelity_delta_from_cnn_site import (
    dpi,
    figsize,
    one_error_note,
    one_error_pp,
    y_axis_limits,
)
from plots.plot_fidelity_vs_exposure import display_method_label, only_4_8_16
from plots.plot_learning_curves import plot_learning_curves


real_data_dir = Path(__file__).resolve().parent
output_dir = real_data_dir / "output"
architecture_model_dir = output_dir / "models_4_8_16"
model_dirs = (output_dir / "models", architecture_model_dir)
plot_path = output_dir / "real_dataset_vs_cnn_site.png"
csv_path = output_dir / "real_dataset_fidelity.csv"
learning_curve_dir = output_dir / "learning_curves"
draw_learning_curves = False

methods = (
    "cnn_site",
    "lightweight_cnn_site",
    "lightweight_cnn_site_neighbor",
    "lightweight_cnn_site_neighbor_weighted",
)
baseline_method = "cnn_site"
marker_size = 90
show = False


def collect_records():
    records = []
    for model_dir in model_dirs:
        for path in sorted(model_dir.glob("*.joblib")):
            model = joblib.load(path)
            if model["method"] not in methods:
                continue

            if draw_learning_curves and "history" in model:
                curve_path = (
                    learning_curve_dir
                    / f"{model['dataset_key']}_{model['method']}_learning_curve.png"
                )
                plot_learning_curves(
                    history=model["history"],
                    output_path=curve_path,
                    title=f"{model['method']} learning curve: {model['dataset_key']}",
                    best_epoch=model["best_epoch"],
                )
                print(f"Saved learning curve: {curve_path}")

            fidelity = np.asarray(model["test_fidelity"], dtype=float)
            mean_fidelity = float(np.mean(fidelity))
            records.append(
                {
                    "dataset_key": model["dataset_key"],
                    "dataset_label": model["dataset_label"],
                    "method": model["method"],
                    "channels": tuple(model["channels"]) if "channels" in model else None,
                    "mean_fidelity": mean_fidelity,
                    "mean_infidelity": 1.0 - mean_fidelity,
                    "total_errors": int(model["test_total_errors"]),
                    "errors_per_site": np.asarray(model["test_errors_per_site"], dtype=float),
                    "worst_site_fidelity": float(np.min(fidelity)),
                    "std_fidelity": float(np.std(fidelity)),
                    "n_sites": int(fidelity.size),
                    "n_images": int(model["n_images"]),
                    "best_epoch": model["best_epoch"] if "best_epoch" in model else "",
                    "epochs_trained": model["epochs_trained"] if "epochs_trained" in model else "",
                    "stopped_early": model["stopped_early"] if "stopped_early" in model else "",
                    "source": str(model_dir),
                    "path": str(path),
                }
            )
    return records


def add_error_ratios(records):
    for dataset_key in {record["dataset_key"] for record in records}:
        group = [record for record in records if record["dataset_key"] == dataset_key]
        baseline = [record for record in group if record["method"] == baseline_method]
        for record in group:
            record["error_rate_vs_baseline"] = (
                "" if not baseline
                else record["mean_infidelity"] / baseline[0]["mean_infidelity"]
            )
            record["delta_fidelity_pp"] = (
                "" if not baseline
                else 100.0 * (record["mean_fidelity"] - baseline[0]["mean_fidelity"])
            )
        if not baseline:
            print(f"{dataset_key}: no {baseline_method} model, error ratios left empty")


def print_records(records):
    print(f"{'dataset':<14}  {'method':<42}  {'fidelity':>10}  {'errors':>8}  {'vs base':>8}  {'dF (p.p.)':>10}")
    for record in sorted(records, key=lambda row: (row["dataset_key"], methods.index(row["method"]), str(row["channels"]))):
        ratio = record["error_rate_vs_baseline"]
        delta = record["delta_fidelity_pp"]
        print(
            f"{record['dataset_key']:<14}  {record['method']:<42}  "
            f"{record['mean_fidelity']:>10.6f}  {record['total_errors']:>8d}  "
            f"{'' if ratio == '' else f'{ratio:>8.2f}'}  "
            f"{'' if delta == '' else f'{delta:>+10.4f}'}"
        )


def save_csv(records, path):
    columns = [key for key in records[0] if key != "errors_per_site"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in columns})
    print(f"Saved {path}")


def bar_label(record):
    label = display_method_label(record["method"])
    if record["channels"] and not only_4_8_16:
        label += " " + "/".join(str(value) for value in record["channels"])
    return label


def plot_records(records, path, legend_note):
    baseline = {
        record["dataset_key"]: record for record in records
        if record["method"] == baseline_method
    }
    if not baseline:
        raise ValueError(f"no {baseline_method} model to compare against")

    rows = sorted(
        [record for record in records if record["method"] != baseline_method],
        key=lambda record: (methods.index(record["method"]), str(record["channels"])),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color="black", linewidth=1)

    bars = []
    for position, record in enumerate(rows):
        reference = baseline[record["dataset_key"]]
        delta = 100.0 * (record["mean_fidelity"] - reference["mean_fidelity"])
        bars.append(
            ax.bar(position, delta, width=0.55, zorder=2, color=f"C{position}")
        )
        ax.annotate(
            f"{delta:+.2f}",
            (position, delta),
            textcoords="offset points",
            xytext=(0, 6 if delta > 0 else -14),
            ha="center", va="bottom" if delta > 0 else "top", fontsize=11,
        )

    ax.set_xticks([])
    ax.set_ylabel("Δ Fidelity (p.p.)")
    ax.set_title("Fidelity Difference from CNN Site")
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    (note,) = ax.plot([], [], " ")
    ax.legend(
        [*bars, note],
        [*(bar_label(record) for record in rows), legend_note],
    )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    print(f"Saved {path}")
    return fig


def main():
    records = collect_records()
    if not records:
        raise FileNotFoundError(f"no models for {methods} in {list(model_dirs)}")

    if only_4_8_16:
        records = [
            record for record in records
            if record["method"] == baseline_method
            or record["source"] == str(architecture_model_dir)
        ]

    add_error_ratios(records)
    print_records(records)

    error_pp, n_decisions = one_error_pp(records)
    for record in records:
        record["one_error_pp"] = error_pp

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(records, csv_path)
    fig = plot_records(records, plot_path, one_error_note(error_pp, n_decisions))

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
