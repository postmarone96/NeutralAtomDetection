import re
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


MODEL_RE = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")
COLORBAR_MIN = 1e-4
COLORBAR_MAX = 2e-3

METHOD_ORDER = ["square", "gaussian", "mf_site", "mf_array", "mf_array_gaussian"]

METHOD_LABELS = {
    "square": "Square",
    "gaussian": "Gaussian",
    "mf_site": "MF-site",
    "mf_array": "MF-array",
    "mf_array_gaussian": "MF-array Gaussian",
}

METHOD_COLORS = {
    "square": "tab:orange",
    "gaussian": "tab:green",
    "mf_site": "tab:blue",
    "mf_array": "tab:red",
    "mf_array_gaussian": "tab:purple",
}


def normalize_method(method):
    if method.startswith("mf_site_") and method.endswith("_splits"):
        return "mf_site"
    if method.startswith("mf_array_gaussian_") and method.endswith("_splits"):
        return "mf_array_gaussian"
    if method.startswith("mf_array_") and method.endswith("_splits"):
        return "mf_array"
    return method


def parse_model_name(path):
    match = MODEL_RE.fullmatch(path.name)
    if match is None:
        return None

    exposure_s = float(match.group(1))
    method = normalize_method(match.group(2))
    return exposure_s, exposure_s * 1000.0, method


def get_infidelity_record(path):
    parsed = parse_model_name(path)
    if parsed is None:
        return None

    exposure_s, exposure_ms, method = parsed
    if method not in METHOD_ORDER:
        return None

    model = joblib.load(path)
    h5_path = model.get("h5_path")

    if "mean_fidelity_per_site" in model:
        fidelity = np.asarray(model["mean_fidelity_per_site"], dtype=float)
        sem_fidelity = np.asarray(model.get("sem_fidelity_per_site", np.nan), dtype=float)
        if sem_fidelity.ndim == 0:
            sem_fidelity = np.full_like(fidelity, np.nan, dtype=float)
        is_repeated = True
    elif "test_fidelity" in model:
        fidelity = np.asarray(model["test_fidelity"], dtype=float)
        sem_fidelity = np.full_like(fidelity, np.nan, dtype=float)
        is_repeated = False
    else:
        return None

    return {
        "exposure_s": exposure_s,
        "exposure_ms": exposure_ms,
        "method": method,
        "infidelity": 1.0 - fidelity,
        "sem": sem_fidelity,
        "is_repeated": is_repeated,
        "path": path,
        "h5_path": None if h5_path is None else Path(h5_path),
    }


def collect_records(joblib_dir):
    by_exposure = {}

    for path in sorted(joblib_dir.glob("*.joblib")):
        record = get_infidelity_record(path)
        if record is None:
            continue

        exposure_key = record["exposure_s"]
        method = record["method"]
        by_method = by_exposure.setdefault(exposure_key, {})

        previous = by_method.get(method)
        if previous is None or (record["is_repeated"] and not previous["is_repeated"]):
            by_method[method] = record

    return by_exposure


def ordered_records(records_by_method):
    return [records_by_method[m] for m in METHOD_ORDER if m in records_by_method]


def grid_shape(n_sites):
    side = int(np.sqrt(n_sites))
    if side * side == n_sites:
        return side, side
    return 1, n_sites


def site_grid(records, n_sites):
    n_rows, n_cols = grid_shape(n_sites)

    for record in records:
        h5_path = record.get("h5_path")
        if h5_path is None or not h5_path.exists():
            continue

        with h5py.File(h5_path, "r") as f:
            if "sites" not in f:
                continue
            sites = np.asarray(f["sites"][:], dtype=float)

        if sites.shape[0] != n_sites or n_rows * n_cols != n_sites:
            continue

        order_by_y = np.argsort(sites[:, 1])
        grid = np.zeros((n_rows, n_cols), dtype=int)

        for row_idx in range(n_rows):
            row_sites = order_by_y[row_idx * n_cols:(row_idx + 1) * n_cols]
            row_sites = row_sites[np.argsort(sites[row_sites, 0])]
            grid[row_idx] = row_sites

        return grid

    return np.arange(n_sites).reshape(n_rows, n_cols)


def exposure_name(exposure_ms):
    return f"{exposure_ms:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def make_log_norm(records):
    values = np.concatenate([record["infidelity"] for record in records])
    values = values[np.isfinite(values)]
    positive = values[values > 0]

    return LogNorm(vmin=COLORBAR_MIN, vmax=COLORBAR_MAX)


def colorbar_ticks(norm):
    ticks = []
    value = norm.vmin
    while value <= norm.vmax:
        ticks.append(value)
        value *= 10.0

    if ticks[-1] < norm.vmax:
        ticks.append(norm.vmax)

    return ticks


def format_tick(value):
    if value >= 0.01:
        return f"{value:g}"
    return f"{value:.0e}"


def plot_exposure(records_by_method, output_dir):
    records = ordered_records(records_by_method)
    if not records:
        return

    exposure_ms = records[0]["exposure_ms"]
    n_sites = len(records[0]["infidelity"])

    for record in records:
        if len(record["infidelity"]) != n_sites:
            raise ValueError(f"Mismatched site count at {exposure_ms:g} ms")

    grid = site_grid(records, n_sites)
    n_rows, n_cols = grid.shape
    n_methods = len(records)

    fig = plt.figure(figsize=(2.35 * n_methods + 1.0, 3.05))
    grid_spec = fig.add_gridspec(
        1,
        n_methods + 1,
        width_ratios=[1.0] * n_methods + [0.06],
    )
    axes = [fig.add_subplot(grid_spec[0, i]) for i in range(n_methods)]
    colorbar_ax = fig.add_subplot(grid_spec[0, -1])

    norm = make_log_norm(records)

    image = None
    for ax, record in zip(axes, records):
        data = record["infidelity"][grid]
        color_data = np.clip(data, norm.vmin, norm.vmax)
        image = ax.imshow(color_data, cmap="magma", norm=norm)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(METHOD_LABELS[record["method"]])

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                site_idx = int(grid[row_idx, col_idx])
                normalized_value = float(norm(color_data[row_idx, col_idx]))
                color = "black" if normalized_value > 0.62 else "white"
                # ax.text(
                #     col_idx,
                #     row_idx,
                #     str(site_idx + 1),
                #     ha="center",
                #     va="center",
                #     color=color,
                #     fontsize=10,
                #     fontweight="bold",
                # )

    cbar = fig.colorbar(image, cax=colorbar_ax)
    ticks = colorbar_ticks(norm)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([format_tick(tick) for tick in ticks])
    cbar.set_label("Infidelity, 1 - F (log scale)")

    fig.suptitle(f"Classification infidelity by qubit, {exposure_ms:g} ms", y=0.96)
    fig.subplots_adjust(left=0.035, right=0.92, bottom=0.08, top=0.78, wspace=0.18)

    output_path = output_dir / f"fig3_per_site_infidelity_{exposure_name(exposure_ms)}ms.png"
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")


def main(joblib_dir=Path("models_3x3"), output_dir=Path("output_3x3/fig3")):
    joblib_dir = Path(joblib_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    by_exposure = collect_records(joblib_dir)
    if not by_exposure:
        raise RuntimeError(f"No matching joblib files found in {joblib_dir}")

    for exposure_s in sorted(by_exposure):
        plot_exposure(by_exposure[exposure_s], output_dir)


if __name__ == "__main__":
    main(
        joblib_dir=Path("models_20x20"),
        output_dir=Path("output_20x20/fig3"),
    )
