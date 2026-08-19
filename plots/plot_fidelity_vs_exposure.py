import csv
import pickle
import re
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from baselines import calc_gaussian_scores_best, calc_square_scores_best
from cnn_array import calc_cnn_array_scores
from cnn_site import calc_cnn_site_scores
from lightweight_cnn_site import calc_lightweight_cnn_site_scores
from mf_array import calc_best_scores_array
from mf_site import calc_best_scores
from utils import H5Dataset


model_re = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")
default_data_dir = project_root / "dataset_3x3"
default_spacing_data_dir = project_root / "spacing_datasets_3x3"
default_model_dir = project_root / "models_3x3"
default_spacing_model_dir = project_root / "spacing_models_3x3"
default_pruned_model_dirs = (
    project_root / "pruned_output/models/cnn_site",
    project_root / "pruned_output/models/lightweight_cnn_site",
)
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
show = False
exclude_method_keywords = (
    "_real",
)
print_available_methods = False
print_selected_methods = False
print_record_table = False
save_selected_records_csv = False
print_skipped_models = False

selected_plot_groups = (
    "core_cnn",
    "lightweight_cnn",
    # "spacing_cnn",
    # "pruned",
    # "classical",
    # "matched_filter",
    # "array",
    # real datasets are plotted by plots/plot_real_dataset_fidelity.py
    # "all",
)
extra_plot_methods = ()

method_groups = {
    "core_cnn": (
        "cnn_site",
    ),
    "lightweight_cnn": (
        "lightweight_cnn_site",
        "lightweight_cnn_site_fc2",
        "lightweight_cnn_site_neighbor",
        "lightweight_cnn_site_neighbor_weighted",
    ),
    "spacing_cnn": (
        "cnn_site_spacing",
        "lightweight_cnn_site_spacing",
        "lightweight_cnn_site_fc2_spacing",
        "lightweight_cnn_site_neighbor_spacing",
        "lightweight_cnn_site_neighbor_weighted_spacing",
        # "lightweight_cnn_site_spacing_conditioned",
    ),
    "pruned": (
        "cnn_site_pruned_p30",
        "cnn_site_pruned_p50",
        "cnn_site_pruned_p70",
        "lightweight_cnn_site_pruned_p30",
        "lightweight_cnn_site_pruned_p50",
        "lightweight_cnn_site_pruned_p70",
    ),
    "classical": (
        "square",
        "gaussian",
        "wei",
        "image_analysis_projection",
    ),
    "matched_filter": (
        "mf_site",
        "mf_site_10_splits",
        "mf_array",
        "mf_array_10_splits",
        "mf_array_gaussian",
        "mf_array_gaussian_10_splits",
    ),
    "array": (
        "cnn_array",
    ),
    "real": (
        "square_real",
        "gaussian_real",
        "cnn_site_real",
        "lightweight_cnn_site_real",
        "mf_site_real",
        "mf_array_real",
        "mf_array_gaussian_real",
        "wei_real",
    ),
}

method_labels = {
    "square": "Square",
    "square_real": "Square real",
    "gaussian": "Gaussian",
    "gaussian_real": "Gaussian real",
    "cnn_site": "CNN site",
    "cnn_site_spacing": "CNN site spacing data",
    "cnn_site_real": "CNN site real",
    "lightweight_cnn_site": "Tiny CNN site",
    "lightweight_cnn_site_fc2": "Tiny CNN site FC2",
    "lightweight_cnn_site_neighbor": "Tiny CNN site + neighbors",
    "lightweight_cnn_site_neighbor_weighted": "Tiny CNN site + weighted neighbors",
    "lightweight_cnn_site_spacing": "Tiny CNN site spacing data",
    "lightweight_cnn_site_fc2_spacing": "Tiny CNN site FC2 spacing data",
    "lightweight_cnn_site_neighbor_spacing": "Tiny CNN site + neighbors spacing data",
    "lightweight_cnn_site_neighbor_weighted_spacing": (
        "Tiny CNN site + weighted neighbors spacing data"
    ),
    "lightweight_cnn_site_spacing_conditioned": "Tiny CNN site spacing-conditioned",
    "lightweight_cnn_site_real": "Tiny CNN site real",
    "cnn_array": "CNN array",
    "mf_site": "MF site",
    "mf_site_real": "MF site real",
    "mf_array": "MF array",
    "mf_array_real": "MF array real",
    "mf_array_gaussian": "MF array Gaussian",
    "mf_array_gaussian_real": "MF array Gaussian real",
    "wei": "Wei Projection",
    "wei_real": "Wei Projection real",
    "image_analysis_projection": "Wei Projection",
}


def normalize_method(method):
    if method.endswith("_spacing"):
        method = method.removesuffix("_spacing")
    if method == "square_real":
        return "square"
    if method == "gaussian_real":
        return "gaussian"
    if method == "cnn_site_real":
        return "cnn_site"
    if method == "lightweight_cnn_site_real":
        return "lightweight_cnn_site"
    if method in {
        "lightweight_cnn_site_fc2",
        "lightweight_cnn_site_neighbor",
        "lightweight_cnn_site_neighbor_weighted",
        "lightweight_cnn_site_spacing_conditioned",
    }:
        return "lightweight_cnn_site"
    if method == "mf_site_real":
        return "mf_site"
    if method == "mf_array_real":
        return "mf_array"
    if method == "mf_array_gaussian_real":
        return "mf_array_gaussian"
    if method == "wei_real":
        return "wei"
    if method.startswith("cnn_array_") and method.endswith("_splits"):
        return "cnn_array"
    if method.startswith("cnn_site_") and method.endswith("_splits"):
        return "cnn_site"
    if method.startswith("lightweight_cnn_site_") and method.endswith("_splits"):
        return "lightweight_cnn_site"
    if method.startswith("mf_site_") and method.endswith("_splits"):
        return "mf_site"
    if method.startswith("mf_array_gaussian_") and method.endswith("_splits"):
        return "mf_array_gaussian"
    if method.startswith("mf_array_") and method.endswith("_splits"):
        return "mf_array"
    return method


def base_method_label(method):
    return method_labels.get(method, display_method_label(method))


def display_method_label(method):
    if method in method_labels:
        return method_labels[method]

    if method.endswith("_spacing"):
        base_method = method.removesuffix("_spacing")
        return f"{base_method_label(base_method)} spacing data"

    pruned_match = re.fullmatch(r"(.+)_pruned_p([0-9]+)", method)
    if pruned_match is not None:
        base_method = pruned_match.group(1)
        prune_percent = int(pruned_match.group(2))
        return f"{base_method_label(base_method)} pruned {prune_percent}%"

    normalized = normalize_method(method)
    label = method_labels.get(normalized, method)

    split_match = re.fullmatch(
        r"(cnn_array|cnn_site|lightweight_cnn_site|mf_site|mf_array|mf_array_gaussian)_([0-9]+)_splits",
        method,
    )
    if split_match is not None:
        n_splits = int(split_match.group(2))
        suffix = "split" if n_splits == 1 else "splits"
        return f"{label} ({n_splits} {suffix})"

    return label


def method_is_excluded(method):
    return any(keyword in method for keyword in exclude_method_keywords)


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

    return {
        method for method in methods
        if not method_is_excluded(method)
    }


def parse_model_name(path):
    match = model_re.fullmatch(path.name)
    if match is None:
        return None

    exposure_s = float(match.group(1))
    exposure_ms = exposure_s * 1000.0
    method = match.group(2)
    # if method.startswith("mf_site_") and method.endswith("_splits"):
    #     method = "mf_site"
    # if method.startswith("mf_array_") and method.endswith("_splits"):
    #     method = "mf_array"

    return exposure_s, exposure_ms, method


def exposure_matches(exposure_s, selected_exposures):
    if not selected_exposures:
        return True
    return any(np.isclose(exposure_s, selected) for selected in selected_exposures)


def method_matches(method, selected_methods):
    if not selected_methods:
        return True
    selected_methods = set(selected_methods)
    return method in selected_methods


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
            if print_skipped_models:
                print(f"Skipping {path.name}: could not load joblib file ({exc})")
            continue

        try:
            fidelity = get_test_fidelity(model)
        except KeyError as exc:
            if print_skipped_models:
                print(f"Skipping {path.name}: {exc}")
            continue

        method = path_method
        if method_suffix:
            method = f"{method}{method_suffix}"
        if method_is_excluded(method):
            continue

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

def resolve_h5_path(model, data_dir, exposure_s):
    h5_path = model.get("h5_path")
    if h5_path is not None:
        h5_path = Path(h5_path)
        if h5_path.exists():
            return h5_path

    for path in sorted(data_dir.glob("*.h5")):
        parsed = re.fullmatch(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5", path.name)
        if parsed is None:
            continue
        if np.isclose(float(parsed.group(1)), exposure_s):
            return path

    raise FileNotFoundError(
        f"could not find h5 file for exposure {exposure_s:g} in {data_dir}"
    )


def make_test_loader(
    h5_path,
    test_idx,
    method,
    mean_image,
    scale,
    batch_size,
    num_workers,
):
    normalized = normalize_method(method)
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
    normalized = normalize_method(method)

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
        scores, labels = calc_lightweight_cnn_site_scores(
            dataloader,
            trained_model,
            device=device,
        )
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


def model_thresholds(trained_model, repeat=None):
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
            thresholds = model_thresholds(trained_model, repeat=repeat)
            margins = scores - thresholds[None, :]
            dark_margins.append(margins[labels <= 0.5])
            bright_margins.append(margins[labels > 0.5])
    else:
        dataloader = make_test_loader(
            h5_path=h5_path,
            test_idx=model["test_idx"],
            method=method,
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
        thresholds = model_thresholds(model)
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
        *default_pruned_model_dirs,
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

    if not records:
        print_empty_records_diagnostics(
            model_dirs=model_dirs,
            spacing_model_dirs=spacing_model_dirs,
            all_records=all_records,
        )
        return

    if print_selected_methods:
        print("\nSelected methods:")
        for method in sorted({record["method"] for record in records}):
            print(f"  {method}")

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
