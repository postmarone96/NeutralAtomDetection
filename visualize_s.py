import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


MODEL_DIR = Path("models_3x3")
OUTPUT_DIR = Path("output_3x3")

# MODEL_DIR = Path("models_20x20")
# OUTPUT_DIR = Path("output_20x20")

METHODS = ["mf_site", "mf_array", "mf_array_gaussian", "square", "gaussian"]
THRESHOLD_METHODS = ["mf_site", "mf_array", "mf_array_gaussian"]

METHOD_LABELS = {
    "mf_site": "MF-site",
    "mf_array": "MF-array",
    "mf_array_gaussian": "MF-array Gaussian",
    "square": "Square",
    "gaussian": "Gaussian",
}

METHOD_COLORS = {
    "mf_site": "tab:blue",
    "mf_array": "tab:red",
    "mf_array_gaussian": "tab:purple",
    "square": "tab:orange",
    "gaussian": "tab:green",
}

MODEL_RE = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")
S_USAGE_VALUES = np.arange(2, 15)
EXPOSURE_XLIM = (0.010, 0.100)

def make_repeated_record(path, exposure, method, model):
    repeats = model["repeat_results"]

    # Shape: (n_repeats, n_sites)
    s = np.stack([np.asarray(r["best_s"], dtype=float) for r in repeats])
    threshold = np.stack([np.asarray(r["thresholds"], dtype=float) for r in repeats])
    val_fidelity = np.stack([np.asarray(r["val_fidelity"], dtype=float) for r in repeats])

    # Collapse repeats -> one value per site, like the old single-run files.
    mean_s_per_site = s.mean(axis=0)
    mean_threshold_per_site = threshold.mean(axis=0)
    mean_val_fidelity_per_site = val_fidelity.mean(axis=0)

    return {
        "path": path,
        "exposure": exposure,
        "method": method,
        "s": mean_s_per_site,
        "s_usage": mean_s_per_site,
        "threshold": mean_threshold_per_site,
        "mean_s": float(np.mean(mean_s_per_site)),
        "std_s": float(np.std(mean_s_per_site)),
        "mean_threshold": float(np.mean(mean_threshold_per_site)),
        "std_threshold": float(np.std(mean_threshold_per_site)),
        "mean_fidelity": float(np.mean(mean_val_fidelity_per_site)),
        "mean_infidelity": float(1.0 - np.mean(mean_val_fidelity_per_site)),
        "n_sites": int(mean_s_per_site.size),
        "n_repeats": int(s.shape[0]),
        "is_repeated": True,
    }

def parse_model_name(path):
    match = MODEL_RE.fullmatch(path.name)
    if match is None:
        return None

    exposure = float(match.group(1))
    method = match.group(2)

    return exposure, method


def normalize_method(method):
    if method.startswith("mf_site_") and method.endswith("_splits"):
        return "mf_site"
    if method.startswith("mf_array_gaussian_") and method.endswith("_splits"):
        return "mf_array_gaussian"
    if method.startswith("mf_array_") and method.endswith("_splits"):
        return "mf_array"
    return method


def get_val_fidelity(model):
    if "val_fidelity" in model:
        return np.asarray(model["val_fidelity"], dtype=float)
    if "best_val_fidelity" in model:
        return np.asarray(model["best_val_fidelity"], dtype=float)
    return None


def get_threshold(model):
    if "threshold" in model:
        return np.asarray(model["threshold"], dtype=float)
    if "thresholds" in model:
        return np.asarray(model["thresholds"], dtype=float)
    return None


def collect_records(model_dir):
    records = []

    for path in sorted(model_dir.glob("*.joblib")):
        parsed = parse_model_name(path)
        if parsed is None:
            continue

        exposure, method = parsed
        model = joblib.load(path)

        if "repeat_results" in model:
            method = normalize_method(method)

            if method not in METHODS:
                continue

            records.append(make_repeated_record(path, exposure, method, model))
            continue

        if method not in METHODS:
            continue

        if "s" not in model:
            continue

        threshold = get_threshold(model)
        if threshold is None:
            continue

        s = np.asarray(model["s"], dtype=float)
        fidelity = get_val_fidelity(model)

        record = {
            "path": path,
            "exposure": exposure,
            "method": method,
            "s": s,
            "threshold": threshold,
            "mean_s": float(np.mean(s)),
            "std_s": float(np.std(s)),
            "mean_threshold": float(np.mean(threshold)),
            "std_threshold": float(np.std(threshold)),
            "n_sites": int(s.size),
            "is_repeated": False,
        }

        if fidelity is not None:
            record["mean_fidelity"] = float(np.mean(fidelity))
            record["mean_infidelity"] = float(1.0 - np.mean(fidelity))

        records.append(record)

    return prefer_repeated_records(records)


def prefer_repeated_records(records):
    best_by_method_exposure = {}

    for record in records:
        key = (record["method"], record["exposure"])
        previous = best_by_method_exposure.get(key)

        if previous is None or (record["is_repeated"] and not previous["is_repeated"]):
            best_by_method_exposure[key] = record

    return sorted(
        best_by_method_exposure.values(),
        key=lambda r: (r["method"], r["exposure"]),
    )


def group_by_method(records):
    return {
        method: sorted(
            [r for r in records if r["method"] == method],
            key=lambda r: r["exposure"],
        )
        for method in METHODS
    }


def format_log_x_axis(ax):
    ax.set_xscale("linear")
    ax.xaxis.set_major_formatter(ScalarFormatter())


def plot_mean_s(records_by_method):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for method, records in records_by_method.items():
        if not records:
            continue

        x = np.array([r["exposure"] for r in records])
        y = np.array([r["mean_s"] for r in records])
        yerr = np.array([r["std_s"] for r in records])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

        for r in records:
            xs = np.full_like(r["s"], r["exposure"], dtype=float)
            ax.scatter(
                xs,
                r["s"],
                s=15,
                alpha=0.25,
                color=METHOD_COLORS[method],
            )

    format_log_x_axis(ax)
    ax.set_xlim(*EXPOSURE_XLIM)
    ax.set_xlabel("Exposure time")
    ax.set_ylabel("Selected window size s")
    ax.set_title("Selected s vs Exposure")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = OUTPUT_DIR / "mean_s_vs_exposure_all_methods.png"
    fig.savefig(out, dpi=200)
    print("Saved:", out)


def plot_threshold(records_by_method):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for method in THRESHOLD_METHODS:
        records = records_by_method.get(method, [])
        if not records:
            continue

        x = np.array([r["exposure"] for r in records])
        y = np.array([r["mean_threshold"] for r in records])
        yerr = np.array([r["std_threshold"] for r in records])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

        for r in records:
            xs = np.full_like(r["threshold"], r["exposure"], dtype=float)
            ax.scatter(
                xs,
                r["threshold"],
                s=15,
                alpha=0.25,
                color=METHOD_COLORS[method],
            )

    format_log_x_axis(ax)
    ax.set_xlim(*EXPOSURE_XLIM)
    ax.set_xlabel("Exposure time")
    ax.set_ylabel("Threshold")
    ax.set_title("Threshold vs Exposure")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out = OUTPUT_DIR / "threshold_vs_exposure_all_methods.png"
    fig.savefig(out, dpi=200)
    print("Saved:", out)


def plot_infidelity(records_by_method):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    plotted = False

    for method, records in records_by_method.items():
        records = [r for r in records if "mean_infidelity" in r]
        if not records:
            continue

        x = np.array([r["exposure"] for r in records])
        y = np.array([r["mean_infidelity"] for r in records])

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        plotted = True

    if not plotted:
        return

    format_log_x_axis(ax)
    ax.set_xlim(*EXPOSURE_XLIM)
    ax.set_xlabel("Exposure time")
    ax.set_ylabel("1 - validation fidelity")
    ax.set_title("Validation Infidelity vs Exposure")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    all_y = [
        r["mean_infidelity"]
        for records in records_by_method.values()
        for r in records
        if "mean_infidelity" in r
    ]
    if all_y and np.all(np.array(all_y) > 0):
        ax.set_yscale("linear")

    fig.tight_layout()
    out = OUTPUT_DIR / "infidelity_vs_exposure_all_methods.png"
    fig.savefig(out, dpi=200)
    print("Saved:", out)


def plot_s_usage_heatmaps(records_by_method):
    methods_with_records = [
        method for method, records in records_by_method.items()
        if records
    ]

    if not methods_with_records:
        return

    fig, axes = plt.subplots(
        len(methods_with_records),
        1,
        figsize=(8, 3.2 * len(methods_with_records)),
        sharex=False,
        squeeze=False,
    )

    axes = axes[:, 0]

    for ax, method in zip(axes, methods_with_records):
        records = records_by_method[method]

        exposures = np.array([r["exposure"] for r in records])
        all_s = S_USAGE_VALUES
        y_pos = np.arange(len(all_s))

        counts = np.zeros((len(all_s), len(exposures)), dtype=int)

        for j, r in enumerate(records):
            s_usage = np.rint(r.get("s_usage", r["s"])).astype(int)
            s_usage = np.clip(s_usage, all_s[0], all_s[-1])
            values, value_counts = np.unique(s_usage, return_counts=True)
            for value, count in zip(values, value_counts):
                i = np.where(all_s == int(value))[0][0]
                counts[i, j] = count

        mesh = ax.pcolormesh(
            exposures,
            y_pos,
            counts,
            shading="nearest",
            cmap="viridis",
        )

        format_log_x_axis(ax)
        ax.set_xlim(*EXPOSURE_XLIM)
        ax.set_xlabel("Exposure time")
        ax.set_ylabel("Selected s")
        ax.set_title(f"{METHOD_LABELS[method]}: s Usage")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_s)

        for i, s_value in enumerate(all_s):
            for j, exposure in enumerate(exposures):
                if counts[i, j] > 0:
                    ax.text(
                        exposure,
                        y_pos[i],
                        str(counts[i, j]),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

        fig.colorbar(mesh, ax=ax, label="number of sites")

    fig.tight_layout()
    out = OUTPUT_DIR / "s_usage_heatmaps_all_methods.png"
    fig.savefig(out, dpi=200)
    print("Saved:", out)


def print_table(records):
    print(
        f"{'method':<10} "
        f"{'exposure':>10} "
        f"{'mean s':>10} "
        f"{'std s':>10} "
        f"{'mean thr':>12} "
        f"{'std thr':>12} "
        f"{'mean F':>12} "
        f"model"
    )
    print("-" * 110)

    for r in records:
        mean_f = r.get("mean_fidelity", np.nan)
        print(
            f"{r['method']:<10} "
            f"{r['exposure']:>10.4g} "
            f"{r['mean_s']:>10.4f} "
            f"{r['std_s']:>10.4f} "
            f"{r['mean_threshold']:>12.6f} "
            f"{r['std_threshold']:>12.6f} "
            f"{mean_f:>12.8f} "
            f"{r['path'].name}"
        )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    records = collect_records(MODEL_DIR)

    if not records:
        raise RuntimeError(f"No matching joblib models found in {MODEL_DIR}")

    print_table(records)

    records_by_method = group_by_method(records)

    plot_mean_s(records_by_method)
    plot_threshold(records_by_method)
    plot_infidelity(records_by_method)
    plot_s_usage_heatmaps(records_by_method)

    plt.show()


if __name__ == "__main__":
    main()
