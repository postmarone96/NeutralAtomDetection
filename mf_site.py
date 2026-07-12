import h5py
import torch
import numpy as np
import joblib
from pathlib import Path

from torch.utils.data import DataLoader
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
    best_t = 0.5
    best_fid = -1.0

    for t in np.linspace(0.01, 0.99, 99):
        fid = classification_fidelity(scores, y, t)
        if fid > best_fid:
            best_fid = fid
            best_t = float(t)

    return best_t, best_fid

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

    for s in s_values:
        learned_W = learn_W(train_loader, s=s, alpha=alpha, device=device)

        val_scores_all, val_labels_all = calc_score(
            val_loader, s=s, W=learned_W, device=device
        )

        for site_idx in range(n_qubits):
            threshold, fid = best_threshold_by_fidelity(
                val_scores_all[:, site_idx],
                val_labels_all[:, site_idx],
            )

            if fid > best["val_fidelity"][site_idx]:
                best["W"][site_idx] = learned_W[site_idx].cpu()
                best["s"][site_idx] = s
                best["threshold"][site_idx] = threshold
                best["val_fidelity"][site_idx] = fid

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
#     model_dir = Path("data")
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

    for repeat_idx in range(n_repeats):
        split_seed = base_seed + repeat_idx

        print("\n" + "#" * 80)
        print(
            f"Dataset: {h5_path.name} | "
            f"repeat {repeat_idx + 1}/{n_repeats} | "
            f"seed {split_seed}"
        )
        print("#" * 80)

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

        print("Per-site test fidelity:")
        print(result["test_fidelity"])
        print("Mean across sites:", result["test_mean_fidelity"])

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

def main(data_dir):
    model_dir = Path("data")
    model_dir.mkdir(exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))

    s_values = range(2, 15, 2)
    alpha = 1e-4
    batch_size = 500
    num_workers = 2
    base_seed = 42
    n_repeats = 10

    device = get_device()
    print("Device:", device)

    for h5_path in h5_paths:
        print("\n" + "=" * 80)
        print("Dataset:", h5_path)
        print("=" * 80)

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

        print("\nRepeated-split summary")
        print("----------------------")
        print("Mean fidelity per site:")
        print(summary["mean_fidelity_per_site"])

        print("\nStandard deviation per site:")
        print(summary["std_fidelity_per_site"])

        print("\nStandard error per site:")
        print(summary["sem_fidelity_per_site"])

        print(
            "\nMean across sites and repeats:",
            summary["mean_fidelity"],
        )
        print(
            "SEM of repeat-level site means:",
            summary["sem_mean_fidelity"],
        )
        print("\nSaved:", output_path)


if __name__ == "__main__":
    main(
        data_dir = Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/dataset_3x3")
)
