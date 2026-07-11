import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


MODEL_RE = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")

METHOD_LABELS = {
    "square": "Square",
    "gaussian": "Gaussian",
    "mf_site": "MF site",
    "mf_array": "MF array",
}


def parse_model_name(path):
    match = MODEL_RE.fullmatch(path.name)
    if match is None:
        return None

    exposure_s = float(match.group(1))
    exposure_ms = exposure_s * 1000.0
    method = match.group(2)

    return exposure_s, exposure_ms, method


def get_test_fidelity(model):
    if "test_fidelity" in model:
        return np.asarray(model["test_fidelity"], dtype=float)

    raise KeyError("model has no test_fidelity")


def collect_records(model_dir):
    records = []

    for path in sorted(model_dir.glob("*.joblib")):
        parsed = parse_model_name(path)
        if parsed is None:
            continue

        exposure_s, exposure_ms, method = parsed
        model = joblib.load(path)
        try:
            fidelity = get_test_fidelity(model)
        except KeyError as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        mean_fidelity = float(np.mean(fidelity))
        mean_infidelity = float(1.0 - mean_fidelity)

        records.append(
            {
                "exposure_s": exposure_s,
                "exposure_ms": exposure_ms,
                "method": method,
                "mean_fidelity": mean_fidelity,
                "mean_infidelity": mean_infidelity,
                "plot_infidelity": max(mean_infidelity, 1e-12),
                "std_fidelity": float(np.std(fidelity)),
                "n_sites": int(fidelity.size),
                "path": path,
            }
        )

    return records


def plot_records(records, output_path):
    if not records:
        raise ValueError("no model records found")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = sorted({record["method"] for record in records})

    for method in methods:
        group = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["exposure_ms"],
        )

        x = [record["exposure_ms"] for record in group]
        y = [record["plot_infidelity"] for record in group]
        label = METHOD_LABELS.get(method, method)

        ax.plot(x, y, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Infidelity, 1 - F")
    ax.set_title("Infidelity vs Exposure Time")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    return fig, ax


def print_table(records):
    records = sorted(records, key=lambda record: (record["method"], record["exposure_ms"]))

    print(
        f"{'method':<12} "
        f"{'exposure ms':>12} "
        f"{'mean F':>12} "
        f"{'1-F':>12} "
        f"{'sites':>8}  "
        f"model"
    )
    print("-" * 95)

    for record in records:
        print(
            f"{record['method']:<12} "
            f"{record['exposure_ms']:>12.4g} "
            f"{record['mean_fidelity']:>12.8f} "
            f"{record['mean_infidelity']:>12.8f} "
            f"{record['n_sites']:>8d}  "
            f"{record['path'].name}"
        )


def main():
    model_dir = Path("data")
    output_path = "models/fidelity_vs_exposure.png"

    Path(output_path).parent.mkdir(exist_ok=True)

    records = collect_records(model_dir)

    print_table(records)
    plot_records(records, output_path)

    print(f"\nSaved plot: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
