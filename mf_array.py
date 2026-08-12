import h5py
import joblib
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from baselines import fit_sigmas_from_average_image
from mf_site import (
    best_threshold_by_fidelity,
    classification_fidelity,
)
from utils import (
    H5Dataset, get_device, compute_preprocessing,
    learn_W, calc_score
)


def coordinate_neighbor_indices(sites, n_neighbors=8):
    sites = np.asarray(sites, dtype=np.float32)
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError(f"expected sites with shape (n_sites, 2), got {sites.shape}")

    n_sites = sites.shape[0]
    if n_sites < 2:
        raise ValueError("MF-array neighbor context requires at least 2 sites")

    n_neighbors = min(int(n_neighbors), n_sites - 1)
    distances = np.sum((sites[:, None, :] - sites[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)

    return np.argsort(distances, axis=1)[:, :n_neighbors].astype(np.int64)


def mf_array_with_s_sweep(
    h5_path,
    train_idx,
    val_idx,
    s_values=range(2, 15),
    alpha=0.0,
    batch_size=256,
    num_workers=4,
    device="cpu",
    mean_image=None,
    scale=None,
    gaussian=False,
    gaussian_sigmas=None,
    gaussian_normalize=False,
    neighbor_indices=None,
    neighbor_context=None,
):
    train_loader = DataLoader(
        H5Dataset(h5_path, indices=train_idx, mean_image=mean_image, scale=scale),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        H5Dataset(h5_path, indices=val_idx, mean_image=mean_image, scale=scale),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    n_qubits = H5Dataset(h5_path, indices=[0]).sites.shape[0]
    if gaussian and gaussian_sigmas is None:
        raise ValueError("gaussian=True requires fitted gaussian_sigmas")

    best = {
        "method": "mf_array",
        "W": [None] * n_qubits,
        "s": np.zeros(n_qubits, dtype=int),
        "threshold": np.zeros(n_qubits, dtype=np.float32),
        "val_fidelity": np.full(n_qubits, -np.inf, dtype=np.float32),
        "history": [],
        "alpha": alpha,
        "mf_array_gaussian": bool(gaussian),
        "mf_array_context": "gaussian" if gaussian else "mean",
        "mf_array_gaussian_normalize": bool(gaussian_normalize),
    }
    if neighbor_indices is not None:
        best["mf_array_neighbor_indices"] = np.asarray(
            neighbor_indices,
            dtype=np.int64,
        ).copy()
        best["mf_array_neighbor_context"] = neighbor_context or "custom"
    if gaussian:
        best["mf_array_sigmas"] = np.asarray(gaussian_sigmas, dtype=np.float32).copy()

    progress = tqdm(s_values, desc="MF-Array s sweep", unit="s", leave=False)
    for s in progress:
        learned_W = learn_W(
            train_loader,
            s=s,
            mf_array=True,
            alpha=alpha,
            device=device,
            mf_array_gaussian=gaussian,
            mf_array_sigmas=gaussian_sigmas,
            mf_array_gaussian_normalize=gaussian_normalize,
            mf_array_neighbor_indices=neighbor_indices,
        )

        val_scores, val_labels = calc_score(
            val_loader,
            s=s,
            W=learned_W,
            mf_array=True,
            device=device,
            mf_array_gaussian=gaussian,
            mf_array_sigmas=gaussian_sigmas,
            mf_array_gaussian_normalize=gaussian_normalize,
            mf_array_neighbor_indices=neighbor_indices,
        )

        thresholds = np.zeros(n_qubits, dtype=np.float32)
        fidelities = np.zeros(n_qubits, dtype=np.float32)

        for q in range(n_qubits):
            threshold, fid = best_threshold_by_fidelity(
                val_scores[:, q],
                val_labels[:, q],
            )
            thresholds[q] = threshold
            fidelities[q] = fid

            if fid > best["val_fidelity"][q]:
                best["W"][q] = learned_W[q].cpu()
                best["s"][q] = s
                best["threshold"][q] = threshold
                best["val_fidelity"][q] = fid

        best["history"].append({
            "s": int(s),
            "thresholds": thresholds.copy(),
            "fidelities": fidelities.copy(),
            "mean_fidelity": float(np.mean(fidelities)),
        })

        # progress.set_postfix(mean_val_fidelity=f"{np.mean(fidelities):.8f}")

    return best


def calc_best_scores_array(dataloader, best, device="cpu"):
    n_qubits = len(best["s"])
    best_s = np.asarray(best["s"])
    gaussian = bool(best.get("mf_array_gaussian", False))
    gaussian_sigmas = best.get("mf_array_sigmas")
    gaussian_normalize = bool(best.get("mf_array_gaussian_normalize", False))
    neighbor_indices = best.get("mf_array_neighbor_indices")

    all_scores = None
    all_labels = None

    for s in sorted(set(int(v) for v in best_s)):
        site_indices = np.where(best_s == s)[0]
        feature_dim = int(best["W"][site_indices[0]].shape[0])
        W = torch.zeros((n_qubits, feature_dim, 1), dtype=best["W"][site_indices[0]].dtype)

        for q in site_indices:
            W[q] = best["W"][q]

        scores_s, labels = calc_score(
            dataloader,
            s=s,
            W=W,
            mf_array=True,
            device=device,
            mf_array_gaussian=gaussian,
            mf_array_sigmas=gaussian_sigmas,
            mf_array_gaussian_normalize=gaussian_normalize,
            mf_array_neighbor_indices=neighbor_indices,
        )

        if all_scores is None:
            all_scores = np.zeros((labels.shape[0], n_qubits), dtype=np.float32)
            all_labels = labels

        all_scores[:, site_indices] = scores_s[:, site_indices]

    return all_scores, all_labels


def evaluate_best_model_array(dataloader, best, device="cpu"):
    scores, labels = calc_best_scores_array(dataloader, best, device=device)
    thresholds = np.asarray(best["threshold"])

    fids = np.zeros(labels.shape[1], dtype=np.float32)
    for q in range(labels.shape[1]):
        fids[q] = classification_fidelity(
            scores[:, q],
            labels[:, q],
            thresholds[q],
        )

    preds = scores >= thresholds[None, :]
    errors = preds != (labels > 0.5)

    return {
        "scores": scores,
        "labels": labels,
        "fidelity": fids,
        "mean_fidelity": float(np.mean(fids)),
        "errors_per_site": errors.sum(axis=0).astype(np.int64),
        "total_errors": int(errors.sum()),
        "raw_accuracy": float(1.0 - errors.mean()),
    }


def make_random_split(
    n_samples: int,
    seed: int,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
):
    if n_samples < 5:
        raise ValueError("At least 5 samples are required.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(train_fraction * n_samples)
    n_val = int(val_fraction * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


def run_one_split(
    h5_path,
    split_seed,
    s_values,
    alpha,
    batch_size,
    num_workers,
    device,
    gaussian=False,
    gaussian_normalize=False,
    neighbor_indices=None,
    neighbor_context=None,
):
    with h5py.File(h5_path, "r") as f:
        n_samples = f["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )

    mean_image, scale = compute_preprocessing(h5_path, train_idx)
    gaussian_sigmas = None
    if gaussian:
        sites = H5Dataset(h5_path, indices=[0]).sites
        gaussian_sigmas = fit_sigmas_from_average_image(
            avg_image=mean_image,
            sites=sites,
            fit_size=9,
        )
    if neighbor_indices is None and neighbor_context == "coordinate_8_nearest":
        sites = H5Dataset(h5_path, indices=[0]).sites.numpy()
        neighbor_indices = coordinate_neighbor_indices(sites, n_neighbors=8)

    trained = mf_array_with_s_sweep(
        h5_path=h5_path,
        train_idx=train_idx,
        val_idx=val_idx,
        s_values=s_values,
        alpha=alpha,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        mean_image=mean_image,
        scale=scale,
        gaussian=gaussian,
        gaussian_sigmas=gaussian_sigmas,
        gaussian_normalize=gaussian_normalize,
        neighbor_indices=neighbor_indices,
        neighbor_context=neighbor_context,
    )

    test_loader = DataLoader(
        H5Dataset(
            h5_path,
            indices=test_idx,
            mean_image=mean_image,
            scale=scale,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    test_metrics = evaluate_best_model_array(
        test_loader,
        trained,
        device=device,
    )

    return {
        "seed": split_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean_image": mean_image,
        "scale": scale,
        "mf_array_gaussian": bool(gaussian),
        "mf_array_context": "gaussian" if gaussian else "mean",
        "mf_array_sigmas": None if gaussian_sigmas is None else gaussian_sigmas.copy(),
        "mf_array_gaussian_normalize": bool(gaussian_normalize),
        "mf_array_neighbor_indices": (
            None
            if neighbor_indices is None
            else np.asarray(neighbor_indices, dtype=np.int64).copy()
        ),
        "mf_array_neighbor_context": neighbor_context,
        "best_s": np.asarray(trained["s"]).copy(),
        "thresholds": np.asarray(trained["threshold"]).copy(),
        "val_fidelity": np.asarray(trained["val_fidelity"]).copy(),
        "test_fidelity": np.asarray(test_metrics["fidelity"]).copy(),
        "test_mean_fidelity": float(test_metrics["mean_fidelity"]),
        "test_raw_accuracy": float(test_metrics["raw_accuracy"]),
        "test_errors_per_site": np.asarray(
            test_metrics["errors_per_site"]
        ).copy(),
        "test_total_errors": int(test_metrics["total_errors"]),
        "trained_model": trained,
    }


def repeated_split_evaluation(
    h5_path,
    n_repeats=10,
    base_seed=42,
    s_values=range(2, 15, 2),
    alpha=1e-4,
    batch_size=500,
    num_workers=2,
    device="cpu",
    gaussian=False,
    gaussian_normalize=False,
):
    repeat_results = []

    progress = tqdm(
        range(n_repeats),
        desc=f"{h5_path.stem} MF-Array repeats",
        unit="split",
        leave=False,
    )
    for repeat_idx in progress:
        split_seed = base_seed + repeat_idx

        result = run_one_split(
            h5_path=h5_path,
            split_seed=split_seed,
            s_values=s_values,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            gaussian=gaussian,
            gaussian_normalize=gaussian_normalize,
        )
        repeat_results.append(result)

        # progress.set_postfix(mean_test_fidelity=f"{result['test_mean_fidelity']:.8f}")

    per_site_fidelity = np.stack(
        [r["test_fidelity"] for r in repeat_results],
        axis=0,
    )
    per_repeat_mean_fidelity = np.asarray(
        [r["test_mean_fidelity"] for r in repeat_results],
        dtype=np.float64,
    )

    n_qubits = per_site_fidelity.shape[1]
    mean_fidelity_per_site = per_site_fidelity.mean(axis=0)
    std_fidelity_per_site = per_site_fidelity.std(axis=0, ddof=1)
    sem_fidelity_per_site = std_fidelity_per_site / np.sqrt(n_repeats)

    std_repeat_mean_fidelity = per_repeat_mean_fidelity.std(ddof=1)
    sem_repeat_mean_fidelity = std_repeat_mean_fidelity / np.sqrt(n_repeats)

    return {
        "n_repeats": n_repeats,
        "n_qubits": n_qubits,
        "base_seed": base_seed,
        "per_site_fidelity": per_site_fidelity,
        "per_repeat_mean_fidelity": per_repeat_mean_fidelity,
        "mean_fidelity_per_site": mean_fidelity_per_site,
        "std_fidelity_per_site": std_fidelity_per_site,
        "sem_fidelity_per_site": sem_fidelity_per_site,
        "mean_fidelity": float(per_repeat_mean_fidelity.mean()),
        "std_mean_fidelity": float(std_repeat_mean_fidelity),
        "sem_mean_fidelity": float(sem_repeat_mean_fidelity),
        "repeat_results": repeat_results,
        "mf_array_gaussian": bool(gaussian),
        "mf_array_context": "gaussian" if gaussian else "mean",
        "mf_array_gaussian_normalize": bool(gaussian_normalize),
    }


def main(data_dir, model_dir):
    model_dir.mkdir(exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))

    s_values = range(2, 15, 2)
    alpha = 1e-4
    batch_size = 500
    num_workers = 2
    base_seed = 42
    n_repeats = 10
    gaussian = True
    gaussian_normalize = True
    device = get_device()

    for h5_path in h5_paths:
        method_name = "mf_array_gaussian" if gaussian else "mf_array"
        output_path = model_dir / f"{h5_path.stem}_{method_name}_{n_repeats}_splits.joblib"

        with h5py.File(h5_path, "r") as f:
            n_samples = f["images"].shape[0]

        if n_samples < 5:
            print(f"Skipping {h5_path}: dataset has only {n_samples} images")
            continue

        summary = repeated_split_evaluation(
            h5_path=h5_path,
            n_repeats=n_repeats,
            base_seed=base_seed,
            s_values=s_values,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            gaussian=gaussian,
            gaussian_normalize=gaussian_normalize,
        )

        save_obj = {
            **summary,
            "method": f"{method_name}_repeated_splits",
            "h5_path": str(h5_path),
            "s_values": list(s_values),
            "alpha": alpha,
            "batch_size": batch_size,
            "mf_array_gaussian": bool(gaussian),
            "mf_array_context": "gaussian" if gaussian else "mean",
            "mf_array_gaussian_normalize": bool(gaussian_normalize),
            "preprocessing": (
                "subtract_repeat_train_mean_image_"
                "divide_repeat_train_max_abs"
            ),
        }

        joblib.dump(save_obj, output_path)

        print(
            f"Saved {output_path} | "
            f"mean F={summary['mean_fidelity']:.8f} | "
            f"SEM={summary['sem_mean_fidelity']:.8f}"
        )


if __name__ == "__main__":
    main(
        data_dir=Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/datasets_20x20"),
        model_dir=Path("models_20x20"),
        # data_dir=Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/dataset_3x3"),
        # model_dir=Path("models_3x3"),
    )
