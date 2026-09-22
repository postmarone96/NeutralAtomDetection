import csv
import pickle
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader

from baselines import calc_gaussian_scores_best, calc_square_scores_best
from cnn_array import calc_cnn_array_scores
from cnn_site import calc_cnn_site_scores, make_site_loader
from lightweight_cnn_site import (
    calc_lightweight_cnn_site_logits_from_site_loader,
    lightweight_cnn_site_method,
    load_lightweight_cnn_site_model,
)
from mf_array import calc_best_scores_array
from mf_site import calc_best_scores
from utils import (
    H5Dataset,
    exposure_matches,
    fold_splits_method,
    method_matches,
    parse_model_name,
    resolve_h5_path,
)


default_data_dir = project_root / "dataset_3x3"
default_spacing_data_dir = project_root / "spacing_datasets_3x3"
default_model_dir = project_root / "models_3x3"
default_spacing_model_dir = project_root / "spacing_models_3x3"
architecture_model_dirs = {
    project_root / "models_3x3_4_8_16": "_4_8_16",
}
default_output_dir = project_root / "output_3x3"
batch_size = 500
num_workers = 0
device = "cpu"
selected_exposures = None
y_axis_limits = None
# y_axis_limits = (5e-2, 9e-1)
make_score_histogram_plots = False
hist_methods = None
max_hist_repeats = None
show = True
print_available_methods = False
print_record_table = False
save_selected_records_csv = True
only_4_8_16 = True

selected_plot_groups = (
    "cnn",
    "classical",
    "matched_filter",
    # "array",
    # "all",
)
extra_plot_methods = ()

method_groups = {
    "cnn": (
        "cnn_site",
    #     "lightweight_cnn_site",
    #     "lightweight_cnn_site_neighbor",
    #     "lightweight_cnn_site_neighbor_weighted",
    ),
    "classical": (
        "square",
        "gaussian",
        "wei",
    ),
    "matched_filter": (
        # "mf_site",
        "mf_site_10_splits",
        # "mf_array",
        "mf_array_10_splits",
        # "mf_array_gaussian",
        # "mf_array_gaussian_10_splits",
    ),
    # "array": (
    #     "cnn_array",
    # ),

}

method_labels = {
    "square": "Square",
    "gaussian": "Gaussian",
    "cnn_site": "CNN Site",
    "lightweight_cnn_site": "TinyCNN Site",
    "lightweight_cnn_site_neighbor": "TinyCNN Site + 8 Neighbors",
    "lightweight_cnn_site_neighbor_weighted": "TinyCNN Site + 8 weighted Neighbors",
    "lightweight_cnn_site_features": "TinyCNN Site + learned neighbors",
    "cnn_array": "CNN array",
    "mf_site": "MF site",
    "mf_array": "MF array",
    "mf_array_gaussian": "MF array Gaussian",
    "wei": "Projection-based state reconstruction",
    "image_analysis_projection": "Projection-based",
}


def method_family(method):
    if method in {
        "lightweight_cnn_site_neighbor",
        "lightweight_cnn_site_neighbor_weighted",
        "lightweight_cnn_site_features",
        "lightweight_cnn_site_spacing_conditioned",
    }:
        return "lightweight_cnn_site"
    return fold_splits_method(method)


def base_method_label(method):
    return method_labels.get(method, display_method_label(method))


def display_method_label(method):
    for suffix in architecture_model_dirs.values():
        if method.endswith(suffix):
            base_label = display_method_label(method[: -len(suffix)])
            if only_4_8_16:
                return base_label
            variant = suffix.strip("_").replace("_", "/")
            return f"{base_label} {variant}"

    if method in method_labels:
        return method_labels[method]

    normalized = method_family(method)
    label = method_labels.get(normalized, method)

    return label


def resolve_selected_methods(records):
    if not selected_plot_groups and not extra_plot_methods:
        return None

    if "all" in selected_plot_groups:
        methods = {record["method"] for record in records}
    else:
        methods = set(extra_plot_methods)
        for group_name in selected_plot_groups:
            if group_name not in method_groups:
                raise KeyError(f"unknown plot method group: {group_name}")
            methods.update(method_groups[group_name])

        for suffix in architecture_model_dirs.values():
            methods.update(f"{method}{suffix}" for method in set(methods))

    return methods


def get_test_fidelity(model):
    if "test_fidelity" in model:
        return np.asarray(model["test_fidelity"], dtype=float)

    if "mean_fidelity_per_site" in model:
        return np.asarray(model["mean_fidelity_per_site"], dtype=float)

    raise KeyError("model has no test_fidelity or mean_fidelity_per_site")


def get_theoretical_infidelity(model):
    if "calibration_mean_error_rate_estimate" in model:
        return float(model["calibration_mean_error_rate_estimate"])

    if "calibration_error_rate_estimate" in model:
        return float(np.mean(model["calibration_error_rate_estimate"]))

    if "calibration_fidelity_estimate" in model:
        return float(1.0 - np.mean(model["calibration_fidelity_estimate"]))

    return None


def collect_joblib_records(
    model_dir,
    selected_exposures=None,
    method_suffix="",
    data_dir=None,
):
    records = []
    if not model_dir.exists():
        return records

    for path in sorted(model_dir.glob("*.joblib")):
        parsed = parse_model_name(path)
        if parsed is None:
            continue

        exposure_s, exposure_ms, path_method = parsed
        if not exposure_matches(exposure_s, selected_exposures):
            continue

        try:
            model = joblib.load(path)
        except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
            continue

        try:
            fidelity = get_test_fidelity(model)
        except KeyError as exc:
            continue

        method = path_method
        if method_suffix:
            method = f"{method}{method_suffix}"

        mean_fidelity = float(np.mean(fidelity))
        mean_infidelity = float(1.0 - mean_fidelity)
        theoretical_infidelity = get_theoretical_infidelity(model)

        records.append(
            {
                "exposure_s": exposure_s,
                "exposure_ms": exposure_ms,
                "method": method,
                "mean_fidelity": mean_fidelity,
                "mean_infidelity": mean_infidelity,
                "plot_infidelity": max(mean_infidelity, 1e-12),
                "theoretical_infidelity": theoretical_infidelity,
                "plot_theoretical_infidelity": (
                    None
                    if theoretical_infidelity is None
                    else max(theoretical_infidelity, 1e-12)
                ),
                "std_fidelity": float(np.std(fidelity)),
                "n_sites": int(fidelity.size),
                "path": path,
                "source": model_dir,
                "data_dir": data_dir,
            }
        )

    return records


def collect_records(
    model_dirs,
    spacing_model_dirs=(),
    selected_exposures=None,
    data_dir=None,
    spacing_data_dir=None,
    architecture_model_dirs=architecture_model_dirs,
):
    records = []

    for model_dir in model_dirs:
        records.extend(
            collect_joblib_records(
                Path(model_dir),
                selected_exposures=selected_exposures,
                data_dir=data_dir,
            )
        )

    for model_dir in spacing_model_dirs:
        records.extend(
            collect_joblib_records(
                Path(model_dir),
                selected_exposures=selected_exposures,
                method_suffix="_spacing",
                data_dir=spacing_data_dir,
            )
        )

    for model_dir, suffix in architecture_model_dirs.items():
        records.extend(
            collect_joblib_records(
                Path(model_dir),
                selected_exposures=selected_exposures,
                method_suffix=suffix,
                data_dir=data_dir,
            )
        )

    return records


def plot_records(records, output_path):
    if not records:
        raise ValueError("no model records found")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    methods = sorted({record["method"] for record in records})

    for method in methods:
        group = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )

        x = [record["exposure_ms"] for record in group]
        y = [record["plot_infidelity"] for record in group]
        label = display_method_label(method)

        ax.plot(x, y, marker="o", linewidth=2, label=label)

        theoretical_group = [
            record for record in group
            if record.get("plot_theoretical_infidelity") is not None
            and method != "wei"
        ]
        if theoretical_group:
            tx = [record["exposure_ms"] for record in theoretical_group]
            ty = [
                record["plot_theoretical_infidelity"]
                for record in theoretical_group
            ]
            ax.plot(
                tx,
                ty,
                marker="x",
                linestyle="--",
                linewidth=1.5,
                label=f"{label} theoretical",
            )

    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Infidelity, 1 - F")
    ax.set_title("Infidelity vs Exposure Time")
    ax.set_yscale("log")
    log_formatter = ax.yaxis.get_major_formatter()
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position: (
                "0"
                if np.isclose(value, 1e-12, rtol=1e-6, atol=0)
                else log_formatter(value, position)
            )
        )
    )
    if y_axis_limits is None and any(
        record["plot_infidelity"] <= 1e-12 for record in records
    ):
        top = int(np.ceil(np.log10(max(record["plot_infidelity"] for record in records))))
        ax.set_yticks(10.0 ** np.arange(-12, top + 1, 2))
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)
    ax.grid(True, which="both", alpha=0.3)
    if len(methods) > 10:
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    return fig, ax

def make_test_loader(
    h5_path,
    test_idx,
    method,
    model,
    mean_image,
    scale,
    batch_size,
    num_workers,
):
    normalized = method_family(method)
    if normalized == "lightweight_cnn_site":
        return make_site_loader(
            h5_path=h5_path,
            indices=test_idx,
            crop_size=model["crop_size"],
            mean_image=mean_image,
            scale=scale,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            return_sites=model["method"] != lightweight_cnn_site_method,
        )

    use_preprocessing = normalized in {
        "cnn_array",
        "cnn_site",
        "lightweight_cnn_site",
        "mf_site",
        "mf_array",
        "mf_array_gaussian",
    }

    return DataLoader(
        H5Dataset(
            h5_path,
            indices=test_idx,
            mean_image=mean_image if use_preprocessing else None,
            scale=scale if use_preprocessing else None,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def calc_test_scores(dataloader, trained_model, method, device):
    normalized = method_family(method)

    if normalized == "square":
        scores, _, labels = calc_square_scores_best(
            dataloader,
            trained_model,
            device=device,
        )
    elif normalized == "gaussian":
        scores, _, labels = calc_gaussian_scores_best(
            dataloader,
            trained_model,
            device=device,
        )
    elif normalized == "cnn_site":
        scores, labels = calc_cnn_site_scores(
            dataloader,
            trained_model,
            device=device,
        )
    elif normalized == "lightweight_cnn_site":
        logits, labels = calc_lightweight_cnn_site_logits_from_site_loader(
            dataloader,
            load_lightweight_cnn_site_model(trained_model, device=device),
            device=device,
        )
        scores = logits[..., 1] - logits[..., 0]
    elif normalized == "cnn_array":
        scores, labels = calc_cnn_array_scores(
            dataloader,
            trained_model,
            device=device,
        )
    elif normalized == "mf_site":
        scores, labels = calc_best_scores(
            dataloader,
            trained_model,
            device=device,
        )
    elif normalized in {"mf_array", "mf_array_gaussian"}:
        scores, labels = calc_best_scores_array(
            dataloader,
            trained_model,
            device=device,
        )
    else:
        raise ValueError(f"unsupported method for score histogram: {method}")

    return scores, labels


def model_thresholds(trained_model, method, n_sites, repeat=None):
    if method_family(method) == "lightweight_cnn_site":
        return np.zeros(n_sites, dtype=np.float32)
    if "threshold" in trained_model:
        return np.asarray(trained_model["threshold"], dtype=np.float32)
    if "thresholds" in trained_model:
        return np.asarray(trained_model["thresholds"], dtype=np.float32)
    if repeat is not None and "thresholds" in repeat:
        return np.asarray(repeat["thresholds"], dtype=np.float32)
    raise KeyError("model has no threshold or thresholds")


def score_margin_histogram_data(
    record,
    model,
    data_dir,
    batch_size,
    num_workers,
    device,
    max_repeats=None,
):
    h5_path = resolve_h5_path(model, data_dir, record["exposure_s"])
    method = record["method"]

    dark_margins = []
    bright_margins = []

    if "repeat_results" in model:
        repeats = model["repeat_results"]
        if max_repeats is not None:
            repeats = repeats[:max_repeats]

        for repeat in repeats:
            trained_model = repeat["trained_model"]
            dataloader = make_test_loader(
                h5_path=h5_path,
                test_idx=repeat["test_idx"],
                method=method,
                model=trained_model,
                mean_image=repeat.get("mean_image"),
                scale=repeat.get("scale"),
                batch_size=batch_size,
                num_workers=num_workers,
            )
            scores, labels = calc_test_scores(
                dataloader,
                trained_model,
                method=method,
                device=device,
            )
            thresholds = model_thresholds(
                trained_model, method, scores.shape[1], repeat=repeat
            )
            margins = scores - thresholds[None, :]
            dark_margins.append(margins[labels <= 0.5])
            bright_margins.append(margins[labels > 0.5])
    else:
        dataloader = make_test_loader(
            h5_path=h5_path,
            test_idx=model["test_idx"],
            method=method,
            model=model,
            mean_image=model.get("mean_image"),
            scale=model.get("scale"),
            batch_size=batch_size,
            num_workers=num_workers,
        )
        scores, labels = calc_test_scores(
            dataloader,
            model,
            method=method,
            device=device,
        )
        thresholds = model_thresholds(model, method, scores.shape[1])
        margins = scores - thresholds[None, :]
        dark_margins.append(margins[labels <= 0.5])
        bright_margins.append(margins[labels > 0.5])

    dark = np.concatenate(dark_margins).astype(np.float64)
    bright = np.concatenate(bright_margins).astype(np.float64)
    dark = dark[np.isfinite(dark)]
    bright = bright[np.isfinite(bright)]

    false_bright = int(np.count_nonzero(dark >= 0.0))
    false_dark = int(np.count_nonzero(bright < 0.0))
    total = int(dark.size + bright.size)

    return {
        "dark": dark,
        "bright": bright,
        "false_bright": false_bright,
        "false_dark": false_dark,
        "total": total,
        "accuracy": 1.0 - ((false_bright + false_dark) / max(total, 1)),
    }


def plot_score_margin_histogram(record, hist_data, output_path, bins=80):
    dark = hist_data["dark"]
    bright = hist_data["bright"]
    if dark.size == 0 and bright.size == 0:
        raise ValueError(f"no score margins to plot for {record['path'].name}")

    combined = np.concatenate([dark, bright])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if np.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.hist(
        [dark, bright],
        bins=np.linspace(lo, hi, bins + 1),
        label=[
            f"dark label 0 ({dark.size:,})",
            f"bright label 1 ({bright.size:,})",
        ],
        color=["tab:blue", "tab:orange"],
        alpha=0.68,
    )
    ax.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="threshold",
    )

    ax.set_xlabel("score - site threshold")
    ax.set_ylabel("test count")
    ax.set_title(
        f"{display_method_label(record['method'])}, "
        f"{record['exposure_ms']:g} ms test score distribution"
    )
    ax.text(
        0.98,
        0.95,
        (
            f"false bright: {hist_data['false_bright']:,}\n"
            f"false dark: {hist_data['false_dark']:,}\n"
            f"raw accuracy: {hist_data['accuracy']:.6f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return fig, ax


def plot_score_histograms(
    records,
    data_dir,
    output_dir,
    batch_size,
    num_workers,
    device,
    selected_methods=None,
    max_repeats=None,
):
    histogram_dir = output_dir / "score_histograms"
    histogram_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for record in records:
        if not method_matches(record["method"], selected_methods):
            continue

        model = joblib.load(record["path"])
        try:
            hist_data = score_margin_histogram_data(
                record=record,
                model=model,
                data_dir=record.get("data_dir") or data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                max_repeats=max_repeats,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"Skipping histogram for {record['path'].name}: {exc}")
            continue

        output_path = histogram_dir / f"{record['path'].stem}_score_margin_histogram.png"
        plot_score_margin_histogram(record, hist_data, output_path)
        saved_paths.append(output_path)
        print(f"Saved histogram: {output_path}")

    return saved_paths


def print_table(records):
    records = sorted(records, key=lambda record: (record["method"], record["exposure_ms"]))

    print(
        f"{'method':<36} "
        f"{'exposure ms':>12} "
        f"{'mean F':>12} "
        f"{'1-F':>12} "
        f"{'theory 1-F':>12} "
        f"{'sites':>8}  "
        f"model"
    )
    print("-" * 110)

    for record in records:
        theoretical_infidelity = record.get("theoretical_infidelity")
        theoretical_text = (
            "—"
            if theoretical_infidelity is None
            else f"{theoretical_infidelity:.8f}"
        )
        sites_text = "—" if record["n_sites"] <= 0 else f"{record['n_sites']:d}"
        print(
            f"{record['method']:<36} "
            f"{record['exposure_ms']:>12.4g} "
            f"{record['mean_fidelity']:>12.8f} "
            f"{record['mean_infidelity']:>12.8f} "
            f"{theoretical_text:>12} "
            f"{sites_text:>8}  "
            f"{record['path'].name}"
        )


def print_available_method_summary(records):
    print("Available methods:")
    methods = sorted({record["method"] for record in records})
    for method in methods:
        group = [
            record for record in records
            if record["method"] == method
        ]
        exposures = sorted({record["exposure_ms"] for record in group})
        exposure_text = (
            "—"
            if not exposures
            else f"{exposures[0]:g}-{exposures[-1]:g} ms"
        )
        print(
            f"  {method:<36} "
            f"{len(exposures):>3} exposure points  "
            f"{exposure_text}"
        )


def write_records_csv(records, output_path):
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "exposure_s",
                "exposure_ms",
                "mean_fidelity",
                "mean_infidelity",
                "std_fidelity",
                "n_sites",
                "source",
                "path",
            ],
        )
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (item["method"], item["exposure_ms"]),
        ):
            writer.writerow(
                {
                    "method": record["method"],
                    "exposure_s": record["exposure_s"],
                    "exposure_ms": record["exposure_ms"],
                    "mean_fidelity": record["mean_fidelity"],
                    "mean_infidelity": record["mean_infidelity"],
                    "std_fidelity": record["std_fidelity"],
                    "n_sites": record["n_sites"],
                    "source": record["source"],
                    "path": record["path"],
                }
            )


def print_empty_records_diagnostics(
    model_dirs,
    spacing_model_dirs,
    all_records,
):
    print("No selected model records found.")
    print(f"Project root: {project_root}")
    print("Checked model directories:")
    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        n_joblibs = len(list(model_dir.glob("*.joblib"))) if model_dir.exists() else 0
        print(f"  {model_dir}  exists={model_dir.exists()}  joblibs={n_joblibs}")
    print("Checked spacing model directories:")
    for model_dir in spacing_model_dirs:
        model_dir = Path(model_dir)
        n_joblibs = len(list(model_dir.glob("*.joblib"))) if model_dir.exists() else 0
        print(f"  {model_dir}  exists={model_dir.exists()}  joblibs={n_joblibs}")

    if all_records:
        print("Available methods from found records:")
        print_available_method_summary(all_records)
    else:
        print("No readable records were found in those paths.")

    print("Selected plot groups:")
    for group_name in selected_plot_groups:
        print(f"  {group_name}")


def main(
    model_dirs=(
        default_model_dir,
    ),
    spacing_model_dirs=(
        default_spacing_model_dir,
    ),
    data_dir=default_data_dir,
    spacing_data_dir=default_spacing_data_dir,
    output_dir=default_output_dir,
    batch_size=batch_size,
    num_workers=num_workers,
    device=device,
    selected_exposures=selected_exposures,
    make_score_histograms=make_score_histogram_plots,
    hist_methods=hist_methods,
    max_hist_repeats=max_hist_repeats,
    show=show,
):
    output_path = output_dir / "fidelity_vs_exposure.png"
    csv_output_path = output_dir / "fidelity_vs_exposure_records.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = collect_records(
        model_dirs=model_dirs,
        spacing_model_dirs=spacing_model_dirs,
        selected_exposures=selected_exposures,
        data_dir=data_dir,
        spacing_data_dir=spacing_data_dir,
    )
    if print_available_methods:
        print_available_method_summary(all_records)

    selected_methods = resolve_selected_methods(all_records)
    records = [
        record for record in all_records
        if method_matches(record["method"], selected_methods)
    ]
    if only_4_8_16:
        architecture_suffixes = tuple(architecture_model_dirs.values())
        records = [
            record for record in records
            if not record["method"].startswith("lightweight_cnn_site")
            or record["method"].endswith(architecture_suffixes)
        ]

    if not records:
        print_empty_records_diagnostics(
            model_dirs=model_dirs,
            spacing_model_dirs=spacing_model_dirs,
            all_records=all_records,
        )
        return

    if print_record_table:
        print()
        print_table(records)

    plot_records(records, output_path)

    print(f"\nSaved plot: {output_path}")
    if save_selected_records_csv:
        write_records_csv(records, csv_output_path)
        print(f"Saved selected records: {csv_output_path}")

    if make_score_histograms:
        plot_score_histograms(
            records=records,
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            selected_methods=hist_methods,
            max_repeats=max_hist_repeats,
        )

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main(
        data_dir=default_data_dir,
        spacing_data_dir=default_spacing_data_dir,
        output_dir=default_output_dir,
    )
