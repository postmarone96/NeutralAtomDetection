import argparse
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from mf_array import calc_best_scores_array
from mf_site import calc_best_scores
from utils import H5Dataset


MODEL_RE = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")
DEFAULT_DATA_DIR = Path(
    "/Users/marone/PycharmProjects/"
    "neutral-atom-imaging-simulation-main/dataset_3x3"
)

default_model = Path("/Users/marone/PycharmProjects/NeutralAtomDetection/models_3x3/qubit_dataset_sim_0.062_mf_site_10_splits.joblib")


def parse_model_name(path):
    match = MODEL_RE.fullmatch(path.name)
    if match is None:
        return None
    return float(match.group(1)), match.group(2)


def normalize_method(method):
    if "mf_array_gaussian" in method:
        return "mf_array_gaussian"
    if "mf_array" in method:
        return "mf_array"
    if "mf_site" in method:
        return "mf_site"
    raise ValueError(f"threshold-shift plot only supports MF-site/MF-array, got {method}")


def method_label(method):
    return {
        "mf_site": "MF-site",
        "mf_array": "MF-array",
        "mf_array_gaussian": "MF-array Gaussian",
    }[normalize_method(method)]


def resolve_h5_path(model, model_path, data_dir):
    h5_path = model.get("h5_path")
    if h5_path is not None:
        h5_path = Path(h5_path)
        if h5_path.exists():
            return h5_path

    parsed = parse_model_name(model_path)
    if parsed is None:
        raise FileNotFoundError(
            "model does not contain h5_path and exposure cannot be parsed from filename"
        )

    exposure_s, _ = parsed
    for path in sorted(data_dir.glob("*.h5")):
        h5_parsed = re.fullmatch(
            r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5",
            path.name,
        )
        if h5_parsed is not None and np.isclose(float(h5_parsed.group(1)), exposure_s):
            return path

    raise FileNotFoundError(f"could not find matching h5 in {data_dir}")


def split_repeat_entries(model):
    if "repeat_results" in model:
        return model["repeat_results"]

    return [
        {
            "seed": model.get("seed", 42),
            "test_idx": model["test_idx"],
            "mean_image": model.get("mean_image"),
            "scale": model.get("scale"),
            "trained_model": model,
        }
    ]


def calc_scores_for_repeat(h5_path, repeat, method, batch_size, num_workers, device):
    dataloader = DataLoader(
        H5Dataset(
            h5_path,
            indices=repeat["test_idx"],
            mean_image=repeat.get("mean_image"),
            scale=repeat.get("scale"),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    trained = repeat["trained_model"]
    if normalize_method(method) == "mf_site":
        return calc_best_scores(dataloader, trained, device=device)
    return calc_best_scores_array(dataloader, trained, device=device)


def thresholds_for_repeat(repeat):
    trained = repeat["trained_model"]
    if "threshold" in trained:
        return np.asarray(trained["threshold"], dtype=np.float64)
    if "thresholds" in repeat:
        return np.asarray(repeat["thresholds"], dtype=np.float64)
    raise KeyError("repeat has no threshold information")


def sweep_one_repeat(scores, labels, thresholds, shifts):
    target = labels > 0.5
    dark = ~target
    bright = target

    total_dark_per_site = np.maximum(dark.sum(axis=0), 1)
    total_bright_per_site = np.maximum(bright.sum(axis=0), 1)
    total = max(int(labels.size), 1)

    false_bright_rate = np.zeros(shifts.size, dtype=np.float64)
    false_dark_rate = np.zeros(shifts.size, dtype=np.float64)
    fidelity = np.zeros(shifts.size, dtype=np.float64)
    raw_accuracy = np.zeros(shifts.size, dtype=np.float64)
    false_bright_count = np.zeros(shifts.size, dtype=np.int64)
    false_dark_count = np.zeros(shifts.size, dtype=np.int64)

    for shift_idx, shift in enumerate(shifts):
        pred = scores >= (thresholds[None, :] + shift)

        false_bright = np.logical_and(pred, dark)
        false_dark = np.logical_and(~pred, bright)

        false_bright_count[shift_idx] = int(false_bright.sum())
        false_dark_count[shift_idx] = int(false_dark.sum())

        fb_per_site = false_bright.sum(axis=0) / total_dark_per_site
        fd_per_site = false_dark.sum(axis=0) / total_bright_per_site

        false_bright_rate[shift_idx] = float(fb_per_site.mean())
        false_dark_rate[shift_idx] = float(fd_per_site.mean())
        fidelity[shift_idx] = 1.0 - 0.5 * (
            false_bright_rate[shift_idx] + false_dark_rate[shift_idx]
        )
        raw_accuracy[shift_idx] = 1.0 - (
            (false_bright_count[shift_idx] + false_dark_count[shift_idx]) / total
        )

    return {
        "false_bright_rate": false_bright_rate,
        "false_dark_rate": false_dark_rate,
        "fidelity": fidelity,
        "raw_accuracy": raw_accuracy,
        "false_bright_count": false_bright_count,
        "false_dark_count": false_dark_count,
        "margins": (scores - thresholds[None, :]).ravel(),
    }


def default_shifts(margins, n_shifts):
    margins = margins[np.isfinite(margins)]
    if margins.size == 0:
        return np.linspace(-0.25, 0.25, n_shifts)

    lo, hi = np.percentile(margins, [1.0, 99.0])
    max_abs = max(abs(float(lo)), abs(float(hi)))
    if max_abs == 0.0:
        max_abs = 0.25

    return np.linspace(-max_abs, max_abs, n_shifts)


def average_sweeps(sweeps):
    rate_keys = [
        "false_bright_rate",
        "false_dark_rate",
        "fidelity",
        "raw_accuracy",
    ]
    count_keys = ["false_bright_count", "false_dark_count"]

    averaged = {
        key: np.mean([sweep[key] for sweep in sweeps], axis=0)
        for key in rate_keys
    }
    averaged.update(
        {
            key: np.sum([sweep[key] for sweep in sweeps], axis=0)
            for key in count_keys
        }
    )
    return averaged


def collect_scores(model_path, data_dir, batch_size, num_workers, device, max_repeats):
    model = joblib.load(model_path)
    parsed = parse_model_name(model_path)
    if parsed is None:
        method = normalize_method(model.get("method", "mf_site"))
        exposure_ms = None
    else:
        exposure_s, method = parsed
        exposure_ms = exposure_s * 1000.0
        method = normalize_method(method)

    h5_path = resolve_h5_path(model, model_path, data_dir)
    repeats = split_repeat_entries(model)
    if max_repeats is not None:
        repeats = repeats[:max_repeats]

    repeats_data = []
    all_margins = []
    for repeat in repeats:
        scores, labels = calc_scores_for_repeat(
            h5_path=h5_path,
            repeat=repeat,
            method=method,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        thresholds = thresholds_for_repeat(repeat)
        repeats_data.append((scores, labels, thresholds))
        all_margins.append((scores - thresholds[None, :]).ravel())

    return {
        "method": method,
        "exposure_ms": exposure_ms,
        "h5_path": h5_path,
        "repeats_data": repeats_data,
        "margins": np.concatenate(all_margins),
        "n_repeats": len(repeats_data),
    }


def plot_tradeoff(model_path, collected, shifts, output_path):
    sweeps = [
        sweep_one_repeat(scores, labels, thresholds, shifts)
        for scores, labels, thresholds in collected["repeats_data"]
    ]
    sweep = average_sweeps(sweeps)

    best_idx = int(np.argmax(sweep["fidelity"]))
    zero_idx = int(np.argmin(np.abs(shifts)))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(
        shifts,
        sweep["false_bright_rate"],
        label="false bright rate: dark -> bright",
        color="tab:blue",
        linewidth=1.9,
    )
    ax.plot(
        shifts,
        sweep["false_dark_rate"],
        label="false dark rate: bright -> dark",
        color="tab:orange",
        linewidth=1.9,
    )
    ax.plot(
        shifts,
        1.0 - sweep["fidelity"],
        label="balanced error, 1 - F",
        color="black",
        linewidth=1.5,
    )

    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.2, label="saved threshold")
    ax.axvline(
        shifts[best_idx],
        color="tab:green",
        linestyle=":",
        linewidth=1.8,
        label=f"best shift = {shifts[best_idx]:+.4g}",
    )

    title = f"{method_label(collected['method'])} threshold-shift tradeoff"
    if collected["exposure_ms"] is not None:
        title += f", {collected['exposure_ms']:g} ms"
    ax.set_title(title)
    ax.set_xlabel("threshold shift added to every site threshold")
    ax.set_ylabel("rate")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax.text(
        0.98,
        0.96,
        (
            f"saved threshold\n"
            f"false bright: {sweep['false_bright_rate'][zero_idx]:.5f} "
            f"({sweep['false_bright_count'][zero_idx]:,})\n"
            f"false dark: {sweep['false_dark_rate'][zero_idx]:.5f} "
            f"({sweep['false_dark_count'][zero_idx]:,})\n"
            f"F: {sweep['fidelity'][zero_idx]:.6f}\n\n"
            f"best plotted shift\n"
            f"false bright: {sweep['false_bright_rate'][best_idx]:.5f}\n"
            f"false dark: {sweep['false_dark_rate'][best_idx]:.5f}\n"
            f"F: {sweep['fidelity'][best_idx]:.6f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.92,
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    return {
        "output_path": output_path,
        "best_shift": float(shifts[best_idx]),
        "saved_threshold_fidelity": float(sweep["fidelity"][zero_idx]),
        "best_shift_fidelity": float(sweep["fidelity"][best_idx]),
        "saved_false_bright_rate": float(sweep["false_bright_rate"][zero_idx]),
        "saved_false_dark_rate": float(sweep["false_dark_rate"][zero_idx]),
        "best_false_bright_rate": float(sweep["false_bright_rate"][best_idx]),
        "best_false_dark_rate": float(sweep["false_dark_rate"][best_idx]),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot how false-bright and false-dark errors change when the saved "
            "MF threshold is shifted."
        )
    )
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_3x3/threshold_shift_tradeoffs"),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-shifts", type=int, default=201)
    parser.add_argument("--shift-min", type=float, default=None)
    parser.add_argument("--shift-max", type=float, default=None)
    parser.add_argument(
        "--max-repeats",
        type=int,
        default=None,
        help="Limit repeated-split models for a quick diagnostic.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    collected = collect_scores(
        model_path=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_repeats=args.max_repeats,
    )

    if args.shift_min is None or args.shift_max is None:
        shifts = default_shifts(collected["margins"], args.n_shifts)
    else:
        shifts = np.linspace(args.shift_min, args.shift_max, args.n_shifts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.model.stem}_threshold_shift_tradeoff.png"

    summary = plot_tradeoff(args.model, collected, shifts, output_path)
    print(f"Saved plot: {summary['output_path']}")
    print(
        "saved threshold: "
        f"FB={summary['saved_false_bright_rate']:.6f}, "
        f"FD={summary['saved_false_dark_rate']:.6f}, "
        f"F={summary['saved_threshold_fidelity']:.6f}"
    )
    print(
        f"best plotted shift {summary['best_shift']:+.6g}: "
        f"FB={summary['best_false_bright_rate']:.6f}, "
        f"FD={summary['best_false_dark_rate']:.6f}, "
        f"F={summary['best_shift_fidelity']:.6f}"
    )

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
