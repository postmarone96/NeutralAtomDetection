import argparse
import re
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from mf_site import make_random_split
from utils import H5Dataset, extract_patches, get_device


MODEL_RE = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_(.+)\.joblib$")
H5_RE = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5$")

TARGET_GRID_LABEL = "20x20"
SOURCE_METHODS = ("mf_site", "mf_array")

METHOD_LABELS = {
    "mf_site": "MF-site",
    "mf_array": "MF-array",
}


def transfer_method_name(source_method):
    return f"{source_method}_3x3_transfer_to_{TARGET_GRID_LABEL}"


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

    return float(match.group(1)), normalize_method(match.group(2))


def parse_h5_name(path):
    match = H5_RE.fullmatch(path.name)
    if match is None:
        return None

    return float(match.group(1))


def exposure_matches(exposure, selected_exposures):
    if not selected_exposures:
        return True
    return any(np.isclose(exposure, selected) for selected in selected_exposures)


def collect_source_models(model_dir, source_method, selected_exposures=None):
    models = {}

    for path in sorted(model_dir.glob("*.joblib")):
        parsed = parse_model_name(path)
        if parsed is None:
            continue

        exposure_s, method = parsed
        if method != source_method or not exposure_matches(exposure_s, selected_exposures):
            continue

        previous = models.get(exposure_s)
        is_repeated = "_splits" in path.stem
        previous_is_repeated = previous is not None and "_splits" in previous.stem
        if previous is None or (is_repeated and not previous_is_repeated):
            models[exposure_s] = path

    return models


def collect_h5_files(data_dir, selected_exposures=None):
    datasets = {}

    for path in sorted(data_dir.glob("*.h5")):
        exposure_s = parse_h5_name(path)
        if exposure_s is None or not exposure_matches(exposure_s, selected_exposures):
            continue
        datasets[exposure_s] = path

    return datasets


def load_sites(h5_path):
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["sites"][:], dtype=np.float32)


def sample_count(h5_path):
    with h5py.File(h5_path, "r") as f:
        return int(f["images"].shape[0])


def site_region_map(sites, n_regions_per_axis=3):
    """
    Map target sites to 3x3 regions in row-major order.

    The target grid can be any square size. For a 20x20 target grid this
    creates column groups of 7, 7, 6 sites and row groups of 7, 7, 6 sites.
    The region numbering matches the 3x3 source site order: top-left is 0,
    top-middle is 1, and so on.
    """
    sites = np.asarray(sites, dtype=np.float32)
    x_values = np.sort(np.unique(sites[:, 0]))
    y_values = np.sort(np.unique(sites[:, 1]))

    x_groups = np.array_split(x_values, n_regions_per_axis)
    y_groups = np.array_split(y_values, n_regions_per_axis)

    x_to_group = {
        float(x): group_idx
        for group_idx, group in enumerate(x_groups)
        for x in group
    }
    y_to_group = {
        float(y): group_idx
        for group_idx, group in enumerate(y_groups)
        for y in group
    }

    col = np.asarray([x_to_group[float(x)] for x in sites[:, 0]], dtype=np.int64)
    row = np.asarray([y_to_group[float(y)] for y in sites[:, 1]], dtype=np.int64)
    return row * n_regions_per_axis + col


def region_counts(region_for_site, n_regions=9):
    return np.bincount(region_for_site, minlength=n_regions).astype(np.int64)


def split_repeat_entries(model):
    if "repeat_results" in model:
        return model["repeat_results"]

    required = ["W", "s", "threshold", "mean_image", "scale"]
    missing = [key for key in required if key not in model]
    if missing:
        raise KeyError(
            "single-split source model is missing required keys: "
            + ", ".join(missing)
        )

    return [
        {
            "seed": 42,
            "trained_model": model,
            "mean_image": model["mean_image"],
            "scale": model["scale"],
        }
    ]


def source_mean_patches(mean_image, source_sites, s_values, device):
    mean_image_t = torch.as_tensor(mean_image, dtype=torch.float32, device=device)[None]
    source_sites_t = torch.as_tensor(source_sites, dtype=torch.float32, device=device)

    patches = {}
    for s in sorted(set(int(v) for v in s_values)):
        patches_s = extract_patches(mean_image_t, source_sites_t, s)[0]
        patches[int(s)] = {
            int(region_idx): patches_s[int(region_idx)].clone()
            for region_idx in range(source_sites.shape[0])
        }

    return patches


@torch.no_grad()
def calc_transferred_scores(
    dataloader,
    source_repeat,
    source_sites,
    region_for_site,
    source_method,
    device,
):
    trained = source_repeat["trained_model"]
    source_s = np.asarray(trained["s"], dtype=np.int64)
    source_w = trained["W"]

    if source_s.shape[0] != 9:
        raise ValueError(f"expected 9 source sites, got {source_s.shape[0]}")
    if len(source_w) != 9:
        raise ValueError(f"expected 9 source weight vectors, got {len(source_w)}")

    scale = float(source_repeat["scale"])
    if scale == 0.0:
        scale = 1.0

    mean_patches = source_mean_patches(
        mean_image=source_repeat["mean_image"],
        source_sites=source_sites,
        s_values=source_s,
        device=device,
    )

    region_for_site = np.asarray(region_for_site, dtype=np.int64)
    target_s = source_s[region_for_site]

    all_scores = []
    all_labels = []

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()
        target_sites = sites[0].to(device)

        batch_size = images.shape[0]
        n_sites = target_sites.shape[0]
        batch_scores = torch.empty(
            (batch_size, n_sites),
            dtype=torch.float32,
            device=device,
        )

        for s in sorted(set(int(v) for v in target_s)):
            patches = extract_patches(images, target_sites, s)
            region_context = None
            normalized_patches_by_region = {}

            if source_method == "mf_array":
                region_context = torch.zeros(
                    (batch_size, source_s.shape[0]),
                    dtype=torch.float32,
                    device=device,
                )

                for source_region in range(source_s.shape[0]):
                    target_indices_np = np.where(region_for_site == source_region)[0]
                    if target_indices_np.size == 0:
                        continue

                    target_indices = torch.as_tensor(
                        target_indices_np,
                        dtype=torch.long,
                        device=device,
                    )
                    region_patches = patches.index_select(1, target_indices)
                    region_patches = (
                        region_patches
                        - mean_patches[int(s)][int(source_region)][None, None, :]
                    ) / scale
                    normalized_patches_by_region[int(source_region)] = region_patches
                    region_context[:, source_region] = region_patches.mean(dim=2).mean(dim=1)

            for source_region in np.where(source_s == s)[0]:
                target_indices_np = np.where(region_for_site == source_region)[0]
                if target_indices_np.size == 0:
                    continue

                target_indices = torch.as_tensor(
                    target_indices_np,
                    dtype=torch.long,
                    device=device,
                )
                region_patches = normalized_patches_by_region.get(int(source_region))
                if region_patches is None:
                    region_patches = patches.index_select(1, target_indices)
                    region_patches = (
                        region_patches
                        - mean_patches[int(s)][int(source_region)][None, None, :]
                    ) / scale

                ones = torch.ones(
                    (*region_patches.shape[:2], 1),
                    dtype=region_patches.dtype,
                    device=device,
                )

                if source_method == "mf_site":
                    features = torch.cat([region_patches, ones], dim=2)
                elif source_method == "mf_array":
                    other_regions = [
                        region_idx
                        for region_idx in range(source_s.shape[0])
                        if region_idx != int(source_region)
                    ]
                    context = region_context[:, other_regions]
                    context = context[:, None, :].expand(
                        -1,
                        region_patches.shape[1],
                        -1,
                    )
                    features = torch.cat([region_patches, context, ones], dim=2)
                else:
                    raise ValueError(f"unknown source method: {source_method}")

                weights = source_w[int(source_region)].to(device).float().squeeze(-1)
                batch_scores[:, target_indices] = torch.einsum(
                    "bqd,d->bq",
                    features,
                    weights,
                )

        all_scores.append(batch_scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def evaluate_scores(scores, labels, thresholds):
    target = labels > 0.5
    pred = scores >= thresholds[None, :]

    dark = ~target
    bright = target

    false_bright = np.logical_and(pred, dark).sum(axis=0)
    total_dark = np.maximum(dark.sum(axis=0), 1)

    false_dark = np.logical_and(~pred, bright).sum(axis=0)
    total_bright = np.maximum(bright.sum(axis=0), 1)

    p_false_bright = false_bright / total_dark
    p_false_dark = false_dark / total_bright
    fidelity = 1.0 - 0.5 * (p_false_bright + p_false_dark)
    errors = pred != target

    return {
        "fidelity": fidelity.astype(np.float32),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


def resolve_source_h5(source_model, source_h5_by_exposure, exposure_s):
    h5_path = source_model.get("h5_path")
    if h5_path is not None and Path(h5_path).exists():
        return Path(h5_path)

    if exposure_s in source_h5_by_exposure:
        return source_h5_by_exposure[exposure_s]

    raise FileNotFoundError(
        f"could not resolve source 3x3 h5 file for exposure {exposure_s:g}"
    )


def evaluate_transfer(
    source_model_path,
    target_h5_path,
    source_h5_by_exposure,
    source_method,
    exposure_s,
    batch_size,
    num_workers,
    device,
    max_repeats=None,
):
    source_model = joblib.load(source_model_path)
    source_h5_path = resolve_source_h5(
        source_model=source_model,
        source_h5_by_exposure=source_h5_by_exposure,
        exposure_s=exposure_s,
    )
    source_sites = load_sites(source_h5_path)
    target_sites = load_sites(target_h5_path)
    region_for_site = site_region_map(target_sites)

    repeats = split_repeat_entries(source_model)
    if max_repeats is not None:
        repeats = repeats[:max_repeats]

    n_target_samples = sample_count(target_h5_path)
    per_site_fidelity = []
    per_repeat_mean_fidelity = []
    repeat_results = []

    for repeat_idx, repeat in enumerate(repeats):
        seed = int(repeat.get("seed", 42 + repeat_idx))
        _, _, test_idx = make_random_split(n_samples=n_target_samples, seed=seed)

        dataloader = DataLoader(
            H5Dataset(target_h5_path, indices=test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )

        scores, labels = calc_transferred_scores(
            dataloader=dataloader,
            source_repeat=repeat,
            source_sites=source_sites,
            region_for_site=region_for_site,
            source_method=source_method,
            device=device,
        )

        source_threshold = np.asarray(
            repeat["trained_model"]["threshold"],
            dtype=np.float32,
        )
        target_threshold = source_threshold[region_for_site]

        metrics = evaluate_scores(
            scores=scores,
            labels=labels,
            thresholds=target_threshold,
        )
        fidelity = metrics["fidelity"]
        mean_fidelity = float(np.mean(fidelity))

        per_site_fidelity.append(fidelity)
        per_repeat_mean_fidelity.append(mean_fidelity)
        repeat_results.append(
            {
                "seed": seed,
                "test_idx": test_idx,
                "source_s": np.asarray(
                    repeat["trained_model"]["s"],
                    dtype=np.int64,
                ).copy(),
                "source_threshold": source_threshold.copy(),
                "target_s": np.asarray(
                    repeat["trained_model"]["s"],
                    dtype=np.int64,
                )[region_for_site].copy(),
                "target_threshold": target_threshold.copy(),
                "test_fidelity": fidelity.copy(),
                "test_mean_fidelity": mean_fidelity,
                "test_errors_per_site": metrics["errors_per_site"],
                "test_total_errors": metrics["total_errors"],
                "test_raw_accuracy": metrics["raw_accuracy"],
            }
        )

        print(
            f"  repeat {repeat_idx + 1:02d}/{len(repeats)} "
            f"seed={seed} mean F={mean_fidelity:.8f}"
        )

    per_site_fidelity = np.stack(per_site_fidelity, axis=0)
    per_repeat_mean_fidelity = np.asarray(
        per_repeat_mean_fidelity,
        dtype=np.float64,
    )

    n_repeats = int(per_site_fidelity.shape[0])
    mean_fidelity_per_site = per_site_fidelity.mean(axis=0)

    if n_repeats > 1:
        std_fidelity_per_site = per_site_fidelity.std(axis=0, ddof=1)
        sem_fidelity_per_site = std_fidelity_per_site / np.sqrt(n_repeats)
        std_mean_fidelity = float(per_repeat_mean_fidelity.std(ddof=1))
        sem_mean_fidelity = float(std_mean_fidelity / np.sqrt(n_repeats))
    else:
        std_fidelity_per_site = np.zeros_like(mean_fidelity_per_site)
        sem_fidelity_per_site = np.zeros_like(mean_fidelity_per_site)
        std_mean_fidelity = 0.0
        sem_mean_fidelity = 0.0

    return {
        "method": transfer_method_name(source_method),
        "source_method": source_method,
        "exposure_s": float(exposure_s),
        "n_repeats": n_repeats,
        "n_qubits": int(target_sites.shape[0]),
        "source_model_path": str(source_model_path),
        "source_h5_path": str(source_h5_path),
        "h5_path": str(target_h5_path),
        "region_for_site": region_for_site.astype(np.int64),
        "region_counts": region_counts(region_for_site),
        "per_site_fidelity": per_site_fidelity,
        "per_repeat_mean_fidelity": per_repeat_mean_fidelity,
        "mean_fidelity_per_site": mean_fidelity_per_site,
        "std_fidelity_per_site": std_fidelity_per_site,
        "sem_fidelity_per_site": sem_fidelity_per_site,
        "mean_fidelity": float(per_repeat_mean_fidelity.mean()),
        "std_mean_fidelity": std_mean_fidelity,
        "sem_mean_fidelity": sem_mean_fidelity,
        "repeat_results": repeat_results,
    }


def transfer_record_from_model(path, model):
    exposure_s = float(model.get("exposure_s", parse_model_name(path)[0]))
    mean_fidelity = float(model["mean_fidelity"])
    return {
        "path": path,
        "exposure_s": exposure_s,
        "exposure_ms": exposure_s * 1000.0,
        "mean_fidelity": mean_fidelity,
        "mean_infidelity": float(1.0 - mean_fidelity),
        "plot_infidelity": max(float(1.0 - mean_fidelity), 1e-12),
        "sem_mean_fidelity": float(model.get("sem_mean_fidelity", np.nan)),
        "n_sites": int(model["n_qubits"]),
        "n_repeats": int(model["n_repeats"]),
    }


def collect_native_records(model_dir, source_method, selected_exposures=None):
    records = []

    for path in sorted(model_dir.glob("*.joblib")):
        parsed = parse_model_name(path)
        if parsed is None:
            continue

        exposure_s, method = parsed
        if method != source_method or not exposure_matches(exposure_s, selected_exposures):
            continue

        model = joblib.load(path)
        if "mean_fidelity" in model:
            mean_fidelity = float(model["mean_fidelity"])
            sem_mean_fidelity = float(model.get("sem_mean_fidelity", np.nan))
            n_repeats = int(model.get("n_repeats", 1))
            n_sites = int(model.get("n_qubits", len(model["mean_fidelity_per_site"])))
            is_repeated = "repeat_results" in model
        elif "test_fidelity" in model:
            fidelity = np.asarray(model["test_fidelity"], dtype=float)
            mean_fidelity = float(np.mean(fidelity))
            sem_mean_fidelity = np.nan
            n_repeats = 1
            n_sites = int(fidelity.size)
            is_repeated = False
        else:
            continue

        records.append(
            {
                "path": path,
                "exposure_s": exposure_s,
                "exposure_ms": exposure_s * 1000.0,
                "mean_fidelity": mean_fidelity,
                "mean_infidelity": float(1.0 - mean_fidelity),
                "plot_infidelity": max(float(1.0 - mean_fidelity), 1e-12),
                "sem_mean_fidelity": sem_mean_fidelity,
                "n_sites": n_sites,
                "n_repeats": n_repeats,
                "is_repeated": is_repeated,
            }
        )

    best_by_exposure = {}
    for record in records:
        previous = best_by_exposure.get(record["exposure_s"])
        if previous is None or (record["is_repeated"] and not previous["is_repeated"]):
            best_by_exposure[record["exposure_s"]] = record

    return [best_by_exposure[k] for k in sorted(best_by_exposure)]


def plot_records(transfer_records, native_records, source_method, output_path):
    if not transfer_records:
        raise ValueError("no transfer records to plot")

    method_label = METHOD_LABELS[source_method]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    transfer_records = sorted(transfer_records, key=lambda r: r["exposure_ms"])
    ax.plot(
        [r["exposure_ms"] for r in transfer_records],
        [r["plot_infidelity"] for r in transfer_records],
        marker="o",
        linewidth=1.8,
        label=f"3x3 {method_label} on {TARGET_GRID_LABEL}",
    )

    if native_records:
        native_records = sorted(native_records, key=lambda r: r["exposure_ms"])
        ax.plot(
            [r["exposure_ms"] for r in native_records],
            [r["plot_infidelity"] for r in native_records],
            marker="s",
            linewidth=1.8,
            label=f"{TARGET_GRID_LABEL} {method_label} trained",
        )

    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Infidelity, 1 - F")
    ax.set_title(f"3x3 {method_label} Transfer to {TARGET_GRID_LABEL}")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return fig, ax


def print_table(transfer_records, native_records, source_method):
    method_label = METHOD_LABELS[source_method]
    native_by_exposure = {
        record["exposure_s"]: record
        for record in native_records
    }

    print(f"\n{method_label}")
    print(
        f"{'exposure ms':>12} "
        f"{'transfer F':>12} "
        f"{'transfer 1-F':>12} "
        f"{'native F':>12} "
        f"{'delta F':>12} "
        f"{'repeats':>8}"
    )
    print("-" * 76)

    for record in sorted(transfer_records, key=lambda r: r["exposure_ms"]):
        native = native_by_exposure.get(record["exposure_s"])
        if native is None:
            native_f = " " * 12
            delta = " " * 12
        else:
            native_f = f"{native['mean_fidelity']:>12.8f}"
            delta = f"{record['mean_fidelity'] - native['mean_fidelity']:>12.8f}"

        print(
            f"{record['exposure_ms']:>12.4g} "
            f"{record['mean_fidelity']:>12.8f} "
            f"{record['mean_infidelity']:>12.8f} "
            f"{native_f} "
            f"{delta} "
            f"{record['n_repeats']:>8d}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f"Apply 3x3 MF-site and MF-array models to {TARGET_GRID_LABEL} datasets by "
            "mapping each target site to one of 9 large spatial regions."
        )
    )
    parser.add_argument("--source-model-dir", type=Path, default=Path("models_3x3"))
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=Path(
            "/Users/marone/PycharmProjects/"
            "neutral-atom-imaging-simulation-main/dataset_3x3"
        ),
    )
    parser.add_argument(
        "--target-data-dir",
        type=Path,
        default=Path(
            "/Users/marone/PycharmProjects/"
            "neutral-atom-imaging-simulation-main/datasets_20x20"
        ),
    )
    parser.add_argument("--native-model-dir", type=Path, default=Path("models_20x20"))
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        default=Path("models_3x3_transfer_to_20x20"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_3x3_transfer_to_20x20"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--source-method",
        action="append",
        choices=SOURCE_METHODS,
        default=None,
        help="Source method to transfer. Defaults to both mf_site and mf_array.",
    )
    parser.add_argument(
        "--exposure",
        action="append",
        type=float,
        default=None,
        help="Exposure in seconds to run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-repeats",
        type=int,
        default=None,
        help="Limit repeats for a quick smoke test.",
    )
    parser.add_argument(
        "--no-native-comparison",
        action="store_true",
        help=f"Do not add existing {TARGET_GRID_LABEL}-trained native curves.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or get_device()
    source_methods = args.source_method or list(SOURCE_METHODS)

    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_h5_by_exposure = collect_h5_files(args.target_data_dir, args.exposure)
    source_h5_by_exposure = collect_h5_files(args.source_data_dir, args.exposure)

    if not target_h5_by_exposure:
        raise RuntimeError(f"no target h5 files found in {args.target_data_dir}")

    any_transfer_records = False

    for source_method in source_methods:
        method_label = METHOD_LABELS[source_method]
        source_models = collect_source_models(
            args.source_model_dir,
            source_method,
            args.exposure,
        )
        if not source_models:
            print(f"No source {method_label} models found in {args.source_model_dir}")
            continue

        transfer_records = []
        method_transfer_name = transfer_method_name(source_method)

        for exposure_s in sorted(source_models):
            target_h5_path = target_h5_by_exposure.get(exposure_s)
            if target_h5_path is None:
                print(f"Skipping {exposure_s:g}: no matching {TARGET_GRID_LABEL} h5 file")
                continue

            source_model_path = source_models[exposure_s]
            output_model_path = (
                args.output_model_dir
                / f"{target_h5_path.stem}_{method_transfer_name}.joblib"
            )

            print(
                f"\n{method_label} exposure {exposure_s * 1000.0:g} ms\n"
                f"  source: {source_model_path}\n"
                f"  target: {target_h5_path}"
            )

            result = evaluate_transfer(
                source_model_path=source_model_path,
                target_h5_path=target_h5_path,
                source_h5_by_exposure=source_h5_by_exposure,
                source_method=source_method,
                exposure_s=exposure_s,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                max_repeats=args.max_repeats,
            )
            joblib.dump(result, output_model_path)
            print(
                f"  saved: {output_model_path} | "
                f"mean F={result['mean_fidelity']:.8f}"
            )

            transfer_records.append(
                transfer_record_from_model(output_model_path, result)
            )

        if not transfer_records:
            continue

        any_transfer_records = True
        native_records = []
        if not args.no_native_comparison:
            native_records = collect_native_records(
                args.native_model_dir,
                source_method,
                args.exposure,
            )

        print_table(transfer_records, native_records, source_method)

        output_plot = args.output_dir / (
            f"fidelity_vs_exposure_3x3_{source_method}_transfer_to_{TARGET_GRID_LABEL}.png"
        )
        plot_records(transfer_records, native_records, source_method, output_plot)
        print(f"\nSaved plot: {output_plot}")

    if not any_transfer_records:
        raise RuntimeError("no transfer records were generated")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
