import csv
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt


RESULTS_PATH = Path("output_3x3/cnn_site_quantization_vs_exposure.joblib")
CSV_PATH = None
OUTPUT_PATH = Path("output_3x3/cnn_site_quantization_vs_exposure.png")
DELTA_OUTPUT_PATH = Path("output_3x3/cnn_site_quantization_delta_fidelity.png")
ERROR_OUTPUT_PATH = Path("output_3x3/cnn_site_quantization_error_increase.png")
PLOT_METRIC = "mean_infidelity"  # "mean_infidelity" or "mean_fidelity"
X_MIN_MS = 10.0
X_MAX_MS = 100.0
FIDELITY_TOLERANCE_PERCENT_POINTS = 0.05
SHOW = False


def method_sort_key(method):
    if method == "float32":
        return 0, 0
    match = re.fullmatch(r"fixed([0-9]+)", method)
    if match is not None:
        return 1, -int(match.group(1))
    return 2, method


def load_records(results_path=None, csv_path=None):
    if results_path is not None:
        payload = joblib.load(results_path)
        if isinstance(payload, dict) and "records" in payload:
            return payload["records"]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"unsupported results format in {results_path}")

    records = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for record in csv.DictReader(csv_file):
            records.append(
                {
                    **record,
                    "exposure_s": float(record["exposure_s"]),
                    "exposure_ms": float(record["exposure_ms"]),
                    "mean_fidelity": float(record["mean_fidelity"]),
                    "mean_infidelity": float(record["mean_infidelity"]),
                    "raw_accuracy": float(record["raw_accuracy"]),
                    "total_errors": int(record["total_errors"]),
                    "delta_mean_fidelity": float(record["delta_mean_fidelity"]),
                    "delta_mean_fidelity_percent_points": float(
                        record.get("delta_mean_fidelity_percent_points", 0.0)
                    ),
                    "relative_mean_fidelity_percent": float(
                        record.get("relative_mean_fidelity_percent", 0.0)
                    ),
                    "relative_mean_infidelity_percent": float(
                        record.get("relative_mean_infidelity_percent", 0.0)
                    ),
                    "relative_total_errors_percent": float(
                        record.get("relative_total_errors_percent", 0.0)
                    ),
                }
            )
    return records


def filter_records(records, x_min_ms, x_max_ms):
    filtered = []
    for record in records:
        exposure_ms = record["exposure_ms"]
        if x_min_ms is not None and exposure_ms < x_min_ms:
            continue
        if x_max_ms is not None and exposure_ms > x_max_ms:
            continue
        filtered.append(record)
    return filtered


def plot_records(records, output_path, plot_metric, x_min_ms, x_max_ms):
    if not records:
        raise ValueError("no quantization records to plot")

    fig, ax = plt.subplots(figsize=(7, 5))
    methods = sorted(
        {record["method"] for record in records},
        key=method_sort_key,
    )

    for method in methods:
        group = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )
        x = [record["exposure_ms"] for record in group]
        y = [record[plot_metric] for record in group]
        label = "CNN-site float32" if method == "float32" else f"CNN-site {method}"
        ax.plot(x, y, marker="o", linewidth=2, label=label)

    ax.set_xlabel("Exposure time (ms)")
    if x_min_ms is not None or x_max_ms is not None:
        ax.set_xlim(left=x_min_ms, right=x_max_ms)
    if plot_metric == "mean_infidelity":
        ax.set_ylabel("Infidelity, 1 - F")
        ax.set_title("CNN-site Quantization: Infidelity vs Exposure Time")
    else:
        ax.set_ylabel("Fidelity, F")
        ax.set_title("CNN-site Quantization: Fidelity vs Exposure Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    return fig


def fixed_methods(records):
    return sorted(
        {
            record["method"]
            for record in records
            if record["method"] != "float32"
        },
        key=method_sort_key,
    )


def plot_fidelity_delta(records, output_path, x_min_ms, x_max_ms, tolerance_percent_points):
    if not records:
        raise ValueError("no quantization records to plot")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method in fixed_methods(records):
        group = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )
        x = [record["exposure_ms"] for record in group]
        y = [record["delta_mean_fidelity"] * 100.0 for record in group]
        ax.plot(x, y, marker="o", linewidth=2, label=method)

    ax.axhline(
        tolerance_percent_points,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"+/- {tolerance_percent_points:g} pp band",
    )
    ax.axhline(
        -tolerance_percent_points,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )
    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("F_quantized - F_float (percentage points)")
    ax.set_title("Quantization Fidelity Difference vs Float CNN-site")
    if x_min_ms is not None or x_max_ms is not None:
        ax.set_xlim(left=x_min_ms, right=x_max_ms)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    return fig


def plot_relative_error_increase(records, output_path, x_min_ms, x_max_ms):
    if not records:
        raise ValueError("no quantization records to plot")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method in fixed_methods(records):
        group = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )
        x = [record["exposure_ms"] for record in group]
        y = [record["relative_total_errors_percent"] for record in group]
        ax.plot(x, y, marker="o", linewidth=2, label=method)

    ax.axhline(0.0, color="0.4", linewidth=1.0)
    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Relative change in test errors (%)")
    ax.set_title("Quantization Error-Count Change vs Float CNN-site")
    if x_min_ms is not None or x_max_ms is not None:
        ax.set_xlim(left=x_min_ms, right=x_max_ms)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    return fig


def main(
    results_path=RESULTS_PATH,
    csv_path=CSV_PATH,
    output_path=OUTPUT_PATH,
    delta_output_path=DELTA_OUTPUT_PATH,
    error_output_path=ERROR_OUTPUT_PATH,
    plot_metric=PLOT_METRIC,
    x_min_ms=X_MIN_MS,
    x_max_ms=X_MAX_MS,
    fidelity_tolerance_percent_points=FIDELITY_TOLERANCE_PERCENT_POINTS,
    show=SHOW,
):
    if plot_metric not in {"mean_infidelity", "mean_fidelity"}:
        raise ValueError("plot_metric must be 'mean_infidelity' or 'mean_fidelity'")

    records = load_records(
        results_path=None if csv_path is not None else results_path,
        csv_path=csv_path,
    )
    records = filter_records(records, x_min_ms, x_max_ms)
    plot_records(records, output_path, plot_metric, x_min_ms, x_max_ms)
    plot_fidelity_delta(
        records,
        delta_output_path,
        x_min_ms,
        x_max_ms,
        fidelity_tolerance_percent_points,
    )
    plot_relative_error_increase(records, error_output_path, x_min_ms, x_max_ms)
    print(f"Saved plot: {output_path}")
    print(f"Saved delta plot: {delta_output_path}")
    print(f"Saved error plot: {error_output_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
