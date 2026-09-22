import csv
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
plots_dir = Path(__file__).resolve().parent
for path in (project_root, plots_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt

from plot_fidelity_delta_from_cnn_site import (
    baseline_method,
    collect_delta_rows,
    model_dirs,
    select_records,
    target_methods,
)
from plot_fidelity_vs_exposure import (
    collect_records,
    default_data_dir,
    default_output_dir,
    display_method_label,
)


selected_exposures = None
data_dir = default_data_dir
output_dir = default_output_dir
plot_path = output_dir / "error_rate_improvement.png"
csv_path = output_dir / "error_rate_improvement.csv"

min_infidelity = 1e-3
figsize = (8.8, 5.4)
dpi = 200
y_axis_limits = None
show = False


def collect_improvement_rows(records):
    rows = []
    for row in collect_delta_rows(records):
        model_infidelity = row["mean_infidelity"]
        baseline_infidelity = row["baseline_mean_infidelity"]
        rows.append(
            {
                "exposure_ms": row["exposure_ms"],
                "method": row["method"],
                "label": display_method_label(row["method"]),
                "infidelity": model_infidelity,
                "cnn_site_infidelity": baseline_infidelity,
                "error_rate_improvement_pct": (
                    100.0 * (1.0 - model_infidelity / baseline_infidelity)
                    if baseline_infidelity > 0
                    else None
                ),
                "improvement_factor": (
                    baseline_infidelity / model_infidelity
                    if model_infidelity > 0 and baseline_infidelity > 0
                    else None
                ),
                "plotted": baseline_infidelity >= min_infidelity,
            }
        )

    return rows


def save_improvement_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_improvement_rows(rows, path):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color="black", linewidth=1)

    for method in target_methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method and row["plotted"]],
            key=lambda row: row["exposure_ms"],
        )
        if not method_rows:
            continue

        ax.plot(
            [row["exposure_ms"] for row in method_rows],
            [row["error_rate_improvement_pct"] for row in method_rows],
            marker="o",
            linewidth=2,
            label=display_method_label(method),
        )

    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel(
        f"error rate improvement over {display_method_label(baseline_method)} (%)"
    )
    ax.set_title(
        f"Error Rate Improvement over {display_method_label(baseline_method)}"
    )
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

    return fig, ax


def print_improvement_table(rows):
    print(
        f"{'exposure':>9}  {'method':<42}  "
        f"{'1-F':>12}  {'CNN 1-F':>12}  {'improve %':>10}  plotted"
    )
    for row in sorted(rows, key=lambda item: (item["exposure_ms"], item["method"])):
        improvement = row["error_rate_improvement_pct"]
        improvement_text = "-" if improvement is None else f"{improvement:+.1f}"
        print(
            f"{row['exposure_ms']:>9g}  "
            f"{row['method']:<42}  "
            f"{row['infidelity']:>12.8f}  "
            f"{row['cnn_site_infidelity']:>12.8f}  "
            f"{improvement_text:>10}  "
            f"{row['plotted']}"
        )


def main():
    records = collect_records(
        model_dirs=model_dirs,
        selected_exposures=selected_exposures,
        data_dir=data_dir,
    )
    rows = collect_improvement_rows(select_records(records))
    if not any(row["plotted"] for row in rows):
        raise ValueError(
            f"no exposure has a {baseline_method} infidelity of at least {min_infidelity}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_improvement_csv(rows, csv_path)
    fig, _ax = plot_improvement_rows(rows, plot_path)
    print_improvement_table(rows)

    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
