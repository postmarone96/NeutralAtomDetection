from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
plots_dir = Path(__file__).resolve().parent
for path in (project_root, plots_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from lightweight_cnn_site import crop_size
from plot_fidelity_delta_from_cnn_site import baseline_method, collect_delta_rows, target_methods
from plot_fidelity_vs_exposure import (
    collect_records,
    default_data_dir,
    default_model_dir,
    default_output_dir,
    display_method_label,
)
from utils import find_h5_for_exposure


selected_exposures = None
psf_model_dirs = {
    1.0: default_model_dir,
    2.5: project_root / "models_psf_sweep" / "psf2.5",
    3.0: project_root / "models_psf_sweep" / "psf3.0",
    3.5: project_root / "models_psf_sweep" / "psf3.5",
}
psf_data_dirs = {
    1.0: default_data_dir,
    2.5: project_root / "psf_sweep" / "dataset_3x3_psf2.5",
    3.0: project_root / "psf_sweep" / "dataset_3x3_psf3.0",
    3.5: project_root / "psf_sweep" / "dataset_3x3_psf3.5",
}
methods = (
    "cnn_site",
    "lightweight_cnn_site",
    "lightweight_cnn_site_neighbor",
    "lightweight_cnn_site_neighbor_weighted",
    "lightweight_cnn_site_features",
)

example_exposure_s = 0.01
example_site_pattern = (1, 1, 1, 1, 0, 1, 1, 1, 1)
highlighted_site = 4

output_dir = default_output_dir
plot_path = output_dir / "infidelity_vs_exposure_per_psf.png"
figsize = (28, 15)
dpi = 200
y_axis_limits = None
y_axis_limits_ratio = [0, 2]
show = False


def collect_psf_records():
    records = {}
    for light_source_stdev, model_dir in psf_model_dirs.items():
        records[light_source_stdev] = [
            record
            for record in collect_records(model_dirs=(model_dir,), selected_exposures=selected_exposures)
            if record["method"] in methods
        ]
    return records


def load_example_image(data_dir):
    path = find_h5_for_exposure(data_dir, example_exposure_s)
    with h5py.File(path, "r") as f:
        labels = f["labels"][:] > 0.5
        matches = np.flatnonzero((labels == np.asarray(example_site_pattern, dtype=bool)).all(axis=1))
        if matches.size == 0:
            raise ValueError(f"no image in {path} has site pattern {example_site_pattern}")
        return {
            "image": f["images"][matches[0]].astype(np.float64),
            "sites": f["sites"][:],
            "labels": labels[matches[0]],
            "image_index": int(matches[0]),
        }


def print_counts(records):
    print(f"{'psf':>4}  {'method':<42}  {'exposures':>9}")
    for light_source_stdev, psf_records in records.items():
        for method in methods:
            count = sum(record["method"] == method for record in psf_records)
            print(f"{light_source_stdev:>4g}  {method:<42}  {count:>9}")


def plot_example(ax, example, light_source_stdev):
    image_artist = ax.imshow(example["image"], cmap="viridis")
    ax.figure.colorbar(image_artist, ax=ax, shrink=0.8, label="pixel value")
    half = crop_size // 2
    for site_index, (x, y) in enumerate(example["sites"]):
        highlighted = site_index == highlighted_site
        ax.add_patch(
            Rectangle(
                (np.rint(x) - half - 0.5, np.rint(y) - half - 0.5),
                crop_size,
                crop_size,
                fill=False,
                edgecolor="red" if highlighted else "white",
                linewidth=2.2 if highlighted else 0.6,
                linestyle="-" if highlighted else "--",
                alpha=1.0 if highlighted else 0.35,
                zorder=3 if highlighted else 2,
            )
        )
        ax.plot(
            x,
            y,
            marker="o" if example["labels"][site_index] else "x",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="red" if highlighted else "white",
            markeredgewidth=1.5,
        )
    ax.set_title(
        f"light_source_stdev {light_source_stdev:g} px, {example_exposure_s * 1000:g} ms"
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_infidelity(ax, psf_records, light_source_stdev):
    for index, method in enumerate(methods):
        group = sorted(
            [record for record in psf_records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )
        if not group:
            continue
        ax.plot(
            [record["exposure_ms"] for record in group],
            [record["plot_infidelity"] for record in group],
            marker="o",
            linewidth=2,
            color=f"C{index}",
            label=display_method_label(method),
        )

    name = "normal dataset" if psf_model_dirs[light_source_stdev] == default_model_dir else "PSF sweep"
    ax.set_title(f"light_source_stdev {light_source_stdev:g} px ({name})")
    ax.set_yscale("log")
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)
    ax.grid(True, which="both", alpha=0.3)
    if ax.get_lines():
        ax.legend(fontsize=8)


def plot_error_ratio(ax, psf_records):
    ax.axhline(1.0, color="black", linewidth=1)
    delta_rows = collect_delta_rows(psf_records)
    for method in target_methods:
        rows = sorted(
            [row for row in delta_rows if row["method"] == method],
            key=lambda row: row["exposure_ms"],
        )
        if not rows:
            continue
        ax.plot(
            [row["exposure_ms"] for row in rows],
            [
                row["baseline_mean_infidelity"] / max(row["mean_infidelity"], 1e-12)
                for row in rows
            ],
            marker="o",
            linewidth=2,
            color=f"C{methods.index(method)}",
            label=display_method_label(method),
        )

    ax.set_xlabel("Exposure time (ms)")
    ax.set_yscale("linear")
    if y_axis_limits_ratio is not None:
        ax.set_ylim(y_axis_limits_ratio)
    ax.grid(True, which="both", alpha=0.3)
    if ax.get_lines()[1:]:
        ax.legend(fontsize=8)


def plot_psf_records(records, examples, path):
    fig, axes = plt.subplots(3, len(psf_model_dirs), figsize=figsize, constrained_layout=True)

    for column, light_source_stdev in enumerate(psf_model_dirs):
        plot_example(axes[0, column], examples[light_source_stdev], light_source_stdev)
        plot_infidelity(axes[1, column], records[light_source_stdev], light_source_stdev)
        plot_error_ratio(axes[2, column], records[light_source_stdev])
        if column > 0:
            axes[1, column].sharey(axes[1, 0])
            axes[2, column].sharey(axes[2, 0])

    axes[1, 0].set_ylabel("Infidelity, 1 - F")
    axes[2, 0].set_ylabel(f"errors of {display_method_label(baseline_method)} per error of this model")
    fig.suptitle(
        f"Top: example atom pattern per PSF width, {crop_size}x{crop_size} crops "
        f"(red = site {highlighted_site}; o = atom, x = empty). "
        "Middle: infidelity vs exposure time. "
        f"Bottom: error ratio against {display_method_label(baseline_method)}, above one means fewer errors"
    )
    fig.savefig(path, dpi=dpi)

    return fig, axes


def main():
    if set(psf_model_dirs) != set(psf_data_dirs):
        raise ValueError("psf_model_dirs and psf_data_dirs must list the same PSF widths")

    records = collect_psf_records()
    if not any(records.values()):
        raise ValueError(f"no records for {methods} in {list(psf_model_dirs.values())}")
    examples = {
        light_source_stdev: load_example_image(data_dir)
        for light_source_stdev, data_dir in psf_data_dirs.items()
    }

    print_counts(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, _axes = plot_psf_records(records, examples, plot_path)
    print(f"Saved {plot_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
