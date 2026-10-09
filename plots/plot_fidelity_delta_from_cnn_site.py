import csv
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
plots_dir = Path(__file__).resolve().parent
for path in (project_root, plots_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import joblib
import matplotlib.pyplot as plt

import constants
from plot_fidelity_vs_exposure import (
    assign_model_colors,
    collect_records,
    display_method_label,
)


baseline_method = constants.plot_baseline_method
target_methods = constants.plot_target_methods

selected_exposures = None
model_dirs = constants.plot_model_dirs
data_dir = constants.plot_default_data_dir
output_dir = constants.plot_default_output_dir
plot_path = output_dir / "fidelity_delta_from_cnn_site.png"
csv_path = output_dir / "fidelity_delta_from_cnn_site.csv"

figsize = constants.plot_delta_figsize
dpi = constants.plot_delta_dpi
y_axis_limits = constants.plot_delta_y_axis_limits
show = False


def select_records(records):
    records = [
        record
        for record in records
        if record["method"] in {baseline_method, *target_methods}
    ]
    return records


def collect_delta_rows(records):
    baseline_by_exposure = {
        record["exposure_ms"]: record
        for record in records
        if record["method"] == baseline_method
    }

    rows = []
    for method in target_methods:
        method_records = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )
        for record in method_records:
            baseline = baseline_by_exposure.get(record["exposure_ms"])
            if baseline is None:
                print(
                    "Skipping "
                    f"{method} at {record['exposure_ms']:g} ms: "
                    f"missing {baseline_method} baseline"
                )
                continue

            delta_fidelity = record["mean_fidelity"] - baseline["mean_fidelity"]
            rows.append(
                {
                    "exposure_s": record["exposure_s"],
                    "exposure_ms": record["exposure_ms"],
                    "method": method,
                    "baseline_method": baseline_method,
                    "mean_fidelity": record["mean_fidelity"],
                    "baseline_mean_fidelity": baseline["mean_fidelity"],
                    "delta_fidelity": delta_fidelity,
                    "delta_fidelity_pp": 100.0 * delta_fidelity,
                    "mean_infidelity": record["mean_infidelity"],
                    "baseline_mean_infidelity": baseline["mean_infidelity"],
                    "delta_infidelity_pp": -100.0 * delta_fidelity,
                    "n_sites": record["n_sites"],
                    "path": str(record["path"]),
                    "baseline_path": str(baseline["path"]),
                }
            )

    return rows


def one_error_pp(records):
    baseline = next(record for record in records if record["method"] == baseline_method)
    n_test_images = len(joblib.load(baseline["path"])["test_idx"])
    n_decisions = n_test_images * baseline["n_sites"]
    return 100.0 / n_decisions, n_decisions


def one_error_note(error_pp, n_decisions):
    return f"{error_pp:.4f} p.p. = 1 wrong decision out of {n_decisions:,}"


def save_delta_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_delta_rows(rows, path, legend_note):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color="black", linewidth=1)
    colors = assign_model_colors({row["method"] for row in rows})

    for method in target_methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: row["exposure_ms"],
        )
        if not method_rows:
            continue

        ax.plot(
            [row["exposure_ms"] for row in method_rows],
            [row["delta_fidelity_pp"] for row in method_rows],
            marker="o",
            linewidth=2,
            color=colors[method],
            label=display_method_label(method),
        )

    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Δ Fidelity (p.p.)")
    # ax.set_title("Fidelity Difference from CNN Site")
    if y_axis_limits is not None:
        ax.set_ylim(y_axis_limits)
    ax.grid(True, alpha=0.3)
    ax.plot([], [], " ", label=legend_note)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)

    return fig, ax


def print_delta_table(rows):
    print(
        f"{'exposure':>9}  {'method':<42}  "
        f"{'F':>12}  {'CNN F':>12}  {'ΔF (p.p.)':>11}"
    )
    for row in sorted(rows, key=lambda item: (item["exposure_ms"], item["method"])):
        print(
            f"{row['exposure_ms']:>9g}  "
            f"{row['method']:<42}  "
            f"{row['mean_fidelity']:>12.8f}  "
            f"{row['baseline_mean_fidelity']:>12.8f}  "
            f"{row['delta_fidelity_pp']:>+11.4f}"
        )


def main():
    records = collect_records(
        model_dirs=model_dirs,
        selected_exposures=selected_exposures,
        data_dir=data_dir,
    )
    records = select_records(records)
    rows = collect_delta_rows(records)
    if not rows:
        raise ValueError(
            f"no paired records found for {target_methods} against {baseline_method}"
        )

    error_pp, n_decisions = one_error_pp(records)
    for row in rows:
        row["one_error_pp"] = error_pp

    output_dir.mkdir(parents=True, exist_ok=True)
    save_delta_csv(rows, csv_path)
    fig, _ax = plot_delta_rows(
        rows,
        plot_path,
        legend_note=one_error_note(error_pp, n_decisions),
    )
    print_delta_table(rows)

    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
