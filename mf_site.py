import h5py
import torch
import numpy as np
import joblib
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import (
    get_device, H5Dataset, compute_preprocessing,
    learn_W, calc_score
)


def classification_fidelity(scores, y, threshold):
    pred = scores >= threshold
    target = y > 0.5

    false_bright = np.logical_and(pred == 1, target == 0).sum()
    total_dark = max((target == 0).sum(), 1)

    false_dark = np.logical_and(pred == 0, target == 1).sum()
    total_bright = max((target == 1).sum(), 1)

    fb = false_bright / total_dark
    fd = false_dark / total_bright
    fidelity = 1.0 - 0.5 * (fb + fd)

    return fidelity

def best_threshold_by_fidelity(scores, y):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(y).reshape(-1) > 0.5

    finite = np.isfinite(scores)
    scores = scores[finite]
    target = target[finite]
    if scores.size == 0:
        raise ValueError("cannot choose a threshold from empty/non-finite scores")

    unique_scores, inverse = np.unique(scores, return_inverse=True)
    n_unique = unique_scores.size

    dark = ~target
    bright = target
    n_dark = int(dark.sum())
    n_bright = int(bright.sum())
    total_dark = max(n_dark, 1)
    total_bright = max(n_bright, 1)

    dark_by_score = np.bincount(
        inverse,
        weights=dark.astype(np.float64),
        minlength=n_unique,
    )
    bright_by_score = np.bincount(
        inverse,
        weights=bright.astype(np.float64),
        minlength=n_unique,
    )

    dark_below_threshold = np.concatenate([[0.0], np.cumsum(dark_by_score)])
    bright_below_threshold = np.concatenate([[0.0], np.cumsum(bright_by_score)])

    false_bright = n_dark - dark_below_threshold
    false_dark = bright_below_threshold
    fidelities = 1.0 - 0.5 * (
        false_bright / total_dark + false_dark / total_bright
    )

    score_span = float(unique_scores[-1] - unique_scores[0])
    edge_pad = max(score_span * 1e-6, abs(float(unique_scores[0])) * 1e-6, 1e-12)

    if n_unique == 1:
        candidate_thresholds = np.array(
            [unique_scores[0] - edge_pad, unique_scores[0] + edge_pad],
            dtype=np.float64,
        )
    else:
        midpoint_thresholds = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidate_thresholds = np.concatenate(
            [
                [unique_scores[0] - edge_pad],
                midpoint_thresholds,
                [unique_scores[-1] + edge_pad],
            ]
        )

    best_fid = float(np.max(fidelities))
    best_indices = np.flatnonzero(
        np.isclose(fidelities, best_fid, rtol=1e-12, atol=1e-12)
    )

    run_starts = np.r_[0, np.flatnonzero(np.diff(best_indices) > 1) + 1]
    run_ends = np.r_[run_starts[1:], best_indices.size]
    best_run = None
    best_run_width = -np.inf

    for start, end in zip(run_starts, run_ends):
        run = best_indices[start:end]
        width = candidate_thresholds[run[-1]] - candidate_thresholds[run[0]]
        if width > best_run_width:
            best_run_width = width
            best_run = run

    best_t = 0.5 * (
        candidate_thresholds[best_run[0]]
        + candidate_thresholds[best_run[-1]]
    )

    return float(best_t), best_fid

def mf_site_with_s_sweep(
        h5_path,
        train_idx, val_idx,
        s_values=range(2, 15),
        alpha=0.0, batch_size=256,
        num_workers=4, device="cpu",
        mean_image=None, scale=None):

    train_loader = DataLoader(
        H5Dataset(h5_path, indices=train_idx, mean_image=mean_image, scale=scale),
        batch_size, True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        H5Dataset(h5_path, indices=val_idx, mean_image=mean_image, scale=scale),
        batch_size, False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0
    )

    n_qubits = H5Dataset(h5_path, indices=[0]).sites.shape[0]

    best = {
        "W": [None] * n_qubits,
        "s": np.zeros(n_qubits, dtype=int),
        "threshold": np.zeros(n_qubits, dtype=np.float32),
        "val_fidelity": np.full(n_qubits, -np.inf, dtype=np.float32),
        "history": [],
        "alpha": alpha,
    }

    progress = tqdm(s_values, desc="MF-Site s sweep", unit="s", leave=False)
    for s in progress:
        learned_W = learn_W(train_loader, s=s, alpha=alpha, device=device)

        val_scores_all, val_labels_all = calc_score(
            val_loader, s=s, W=learned_W, device=device
        )

        fidelities = np.zeros(n_qubits, dtype=np.float32)

        for site_idx in range(n_qubits):
            threshold, fid = best_threshold_by_fidelity(
                val_scores_all[:, site_idx],
                val_labels_all[:, site_idx],
            )
            fidelities[site_idx] = fid

            if fid > best["val_fidelity"][site_idx]:
                best["W"][site_idx] = learned_W[site_idx].cpu()
                best["s"][site_idx] = s
                best["threshold"][site_idx] = threshold
                best["val_fidelity"][site_idx] = fid

        # progress.set_postfix(mean_val_fidelity=f"{np.mean(fidelities):.8f}")

    return best

def calc_best_scores(dataloader, best, device="cpu"):
    n_qubits = len(best["s"])
    best_s = np.asarray(best["s"])

    all_scores = None
    all_labels = None

    for s in sorted(set(int(v) for v in best_s)):
        site_indices = np.where(best_s == s)[0]
        W = torch.zeros((n_qubits, s * s + 1, 1), dtype=best["W"][site_indices[0]].dtype)
        for q in site_indices:
            W[q] = best["W"][q]

        scores_s, labels = calc_score(
            dataloader,
            s=s,
            W=W,
            device=device,
        )

        if all_scores is None:
            all_scores = np.zeros((labels.shape[0], n_qubits), dtype=np.float32)
            all_labels = labels

        all_scores[:, site_indices] = scores_s[:, site_indices]

    return all_scores, all_labels

def evaluate_best_model(dataloader, best, device="cpu"):
    scores, labels = calc_best_scores(dataloader, best, device=device)
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

# def main(data_dir):
#     # data_dir = Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/test_dataset_1")
#     model_dir = Path("models_3x3")
#     model_dir.mkdir(exist_ok=True)
#
#     h5_paths = sorted(data_dir.glob("*.h5"))
#
#     s_values = range(2, 15, 2)
#     alpha = 1e-4
#     batch_size = 500
#     num_workers = 2
#     seed = 42
#
#     for h5_path in h5_paths:
#         out_model = model_dir / f"{h5_path.stem}_mf_site.joblib"
#
#         print("\n" + "=" * 80)
#         print("Dataset:", h5_path)
#         print("Output:", out_model)
#         print("=" * 80)
#
#         with h5py.File(h5_path, "r") as f:
#             N = f["images"].shape[0]
#             print("images:", f["images"].shape)
#             print("labels:", f["labels"].shape)
#             print("sites:", f["sites"].shape)
#
#         if N < 5:
#             print(f"Skipping {h5_path}: dataset has only {N} images")
#             continue
#
#         rng = np.random.default_rng(seed)
#         idx = rng.permutation(N)
#
#         n_train = int(0.6 * N)
#         n_val = int(0.2 * N)
#
#         train_idx = idx[:n_train]
#         val_idx = idx[n_train:n_train + n_val]
#         test_idx = idx[n_train + n_val:]
#
#         device = get_device()
#
#         print("train/val/test:", len(train_idx), len(val_idx), len(test_idx))
#         print("device:", device)
#
#         mean_image, scale = compute_preprocessing(h5_path, train_idx)
#         print("preprocessing mean image:", mean_image.shape, mean_image.dtype)
#         print("preprocessing scale:", scale)
#
#         trained = mf_site_with_s_sweep(
#             h5_path=h5_path,
#             train_idx=train_idx,
#             val_idx=val_idx,
#             s_values=s_values,
#             alpha=alpha,
#             batch_size=batch_size,
#             num_workers=num_workers,
#             device=device,
#             mean_image=mean_image,
#             scale=scale,
#         )
#
#         test_loader = DataLoader(
#             H5Dataset(h5_path, indices=test_idx, mean_image=mean_image, scale=scale),
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=False,
#             persistent_workers=num_workers > 0
#         )
#
#         test_metrics = evaluate_best_model(
#             test_loader,
#             trained,
#             device=device,
#         )
#
#         trained["test_fidelity"] = test_metrics["fidelity"]
#         trained["test_mean_fidelity"] = test_metrics["mean_fidelity"]
#         trained["test_errors_per_site"] = test_metrics["errors_per_site"]
#         trained["test_total_errors"] = test_metrics["total_errors"]
#         trained["test_raw_accuracy"] = test_metrics["raw_accuracy"]
#
#         save_obj = {
#             **trained,
#             "method": "mf_site",
#             "h5_path": h5_path,
#             "train_idx": train_idx,
#             "val_idx": val_idx,
#             "test_idx": test_idx,
#             "mean_image": mean_image,
#             "scale": scale,
#             "preprocessing": "subtract_train_mean_image_divide_train_max_abs",
#         }
#
#         joblib.dump(save_obj, out_model)
#
#         print("\nBest s per site:")
#         print(trained["s"])
#
#         print("\nThreshold per site:")
#         print(trained["threshold"])
#
#         print("\nValidation fidelity per site:")
#         print(trained["val_fidelity"])
#
#         print("\nTest fidelity per site:")
#         print(trained["test_fidelity"])
#         print("Mean test fidelity:", trained["test_mean_fidelity"])
#         print("Test errors per site:", trained["test_errors_per_site"])
#         print("Total test errors:", trained["test_total_errors"])
#
#         print("\nSaved model:", out_model)
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
):
    with h5py.File(h5_path, "r") as f:
        n_samples = f["images"].shape[0]

    train_idx, val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=split_seed,
    )

    mean_image, scale = compute_preprocessing(h5_path, train_idx)

    trained = mf_site_with_s_sweep(
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

    test_metrics = evaluate_best_model(
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
    n_repeats=100,
    base_seed=42,
    s_values=range(2, 15, 2),
    alpha=1e-4,
    batch_size=500,
    num_workers=2,
    device="cpu",
):
    repeat_results = []

    progress = tqdm(
        range(n_repeats),
        desc=f"{h5_path.stem} MF-Site repeats",
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
        )
        repeat_results.append(result)

        # progress.set_postfix(mean_test_fidelity=f"{result['test_mean_fidelity']:.8f}")

    # Shape: [repeats, qubits]
    per_site_fidelity = np.stack(
        [r["test_fidelity"] for r in repeat_results],
        axis=0,
    )

    # Shape: [repeats]
    per_repeat_mean_fidelity = np.asarray(
        [r["test_mean_fidelity"] for r in repeat_results],
        dtype=np.float64,
    )

    n_qubits = per_site_fidelity.shape[1]

    # ddof=1 gives the sample standard deviation.
    mean_fidelity_per_site = per_site_fidelity.mean(axis=0)
    std_fidelity_per_site = per_site_fidelity.std(axis=0, ddof=1)
    sem_fidelity_per_site = (
        std_fidelity_per_site / np.sqrt(n_repeats)
    )

    mean_fidelity_across_sites_and_repeats = (
        per_repeat_mean_fidelity.mean()
    )
    std_repeat_mean_fidelity = per_repeat_mean_fidelity.std(ddof=1)
    sem_repeat_mean_fidelity = (
        std_repeat_mean_fidelity / np.sqrt(n_repeats)
    )

    summary = {
        "n_repeats": n_repeats,
        "n_qubits": n_qubits,
        "base_seed": base_seed,
        "per_site_fidelity": per_site_fidelity,
        "per_repeat_mean_fidelity": per_repeat_mean_fidelity,
        "mean_fidelity_per_site": mean_fidelity_per_site,
        "std_fidelity_per_site": std_fidelity_per_site,
        "sem_fidelity_per_site": sem_fidelity_per_site,
        "mean_fidelity": float(
            mean_fidelity_across_sites_and_repeats
        ),
        "std_mean_fidelity": float(std_repeat_mean_fidelity),
        "sem_mean_fidelity": float(sem_repeat_mean_fidelity),
        "repeat_results": repeat_results,
    }

    return summary

def main(data_dir, model_dir):
    model_dir.mkdir(exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))

    s_values = range(2, 15, 2)
    alpha = 1e-4
    batch_size = 500
    num_workers = 2
    base_seed = 42
    n_repeats = 10

    device = get_device()

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            n_samples = f["images"].shape[0]

        if n_samples < 5:
            print(
                f"Skipping {h5_path}: only {n_samples} images."
            )
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
        )

        output_path = (
            model_dir
            / f"{h5_path.stem}_mf_site_{n_repeats}_splits.joblib"
        )

        save_obj = {
            **summary,
            "method": "mf_site_repeated_splits",
            "h5_path": str(h5_path),
            "s_values": list(s_values),
            "alpha": alpha,
            "batch_size": batch_size,
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
        # data_dir = Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/datasets_40x40"),
        # data_dir = Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/dataset_3x3"),
        # model_dir = Path("models_3x3"),
        data_dir=Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/datasets_20x20"),
        model_dir=Path("models_20x20"),
)
