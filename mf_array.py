import h5py
import joblib
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mf_site import (
    best_threshold_by_fidelity,
    classification_fidelity,
)
from utils import (
    H5Dataset, get_device, compute_preprocessing,
    learn_W, calc_score
)


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

    best = {
        "method": "mf_array",
        "W": [None] * n_qubits,
        "s": np.zeros(n_qubits, dtype=int),
        "threshold": np.zeros(n_qubits, dtype=np.float32),
        "val_fidelity": np.full(n_qubits, -np.inf, dtype=np.float32),
        "history": [],
        "alpha": alpha,
    }

    progress = tqdm(s_values, desc="MF-Array s sweep", unit="s")
    for s in progress:
        learned_W = learn_W(
            train_loader,
            s=s,
            mf_array=True,
            alpha=alpha,
            device=device,
        )

        val_scores, val_labels = calc_score(
            val_loader,
            s=s,
            W=learned_W,
            mf_array=True,
            device=device,
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

        progress.set_postfix(mean_val_fidelity=f"{np.mean(fidelities):.8f}")

    return best


def calc_best_scores_array(dataloader, best, device="cpu"):
    n_qubits = len(best["s"])
    best_s = np.asarray(best["s"])

    all_scores = None
    all_labels = None

    for s in sorted(set(int(v) for v in best_s)):
        site_indices = np.where(best_s == s)[0]
        feature_dim = s * s + (n_qubits - 1) + 1
        W = torch.zeros((n_qubits, feature_dim, 1), dtype=best["W"][site_indices[0]].dtype)

        for q in site_indices:
            W[q] = best["W"][q]

        scores_s, labels = calc_score(
            dataloader,
            s=s,
            W=W,
            mf_array=True,
            device=device,
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


def main():
    data_dir = Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/test_dataset")
    model_dir = Path("data")
    model_dir.mkdir(exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))

    s_values = range(2, 15, 2)
    alpha = 1e-4
    batch_size = 500
    num_workers = 2
    seed = 42

    for h5_path in h5_paths:
        out_model = model_dir / f"{h5_path.stem}_mf_array.joblib"

        print("\n" + "=" * 80)
        print("Dataset:", h5_path)
        print("Output:", out_model)
        print("=" * 80)

        with h5py.File(h5_path, "r") as f:
            N = f["images"].shape[0]
            print("images:", f["images"].shape)
            print("labels:", f["labels"].shape)
            print("sites:", f["sites"].shape)

        if N < 5:
            print(f"Skipping {h5_path}: dataset has only {N} images")
            continue

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)

        n_train = int(0.6 * N)
        n_val = int(0.2 * N)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        device = get_device()

        print("train/val/test:", len(train_idx), len(val_idx), len(test_idx))
        print("device:", device)

        mean_image, scale = compute_preprocessing(h5_path, train_idx)
        print("preprocessing mean image:", mean_image.shape, mean_image.dtype)
        print("preprocessing scale:", scale)

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
        )

        test_loader = DataLoader(
            H5Dataset(h5_path, indices=test_idx, mean_image=mean_image, scale=scale),
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

        trained["test_fidelity"] = test_metrics["fidelity"]
        trained["test_mean_fidelity"] = test_metrics["mean_fidelity"]
        trained["test_errors_per_site"] = test_metrics["errors_per_site"]
        trained["test_total_errors"] = test_metrics["total_errors"]
        trained["test_raw_accuracy"] = test_metrics["raw_accuracy"]

        save_obj = {
            **trained,
            "h5_path": h5_path,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "mean_image": mean_image,
            "scale": scale,
            "preprocessing": "subtract_train_mean_image_divide_train_max_abs",
        }

        joblib.dump(save_obj, out_model)

        print("\nBest s per site:")
        print(trained["s"])

        print("\nThreshold per site:")
        print(trained["threshold"])

        print("\nValidation fidelity per site:")
        print(trained["val_fidelity"])

        print("\nTest fidelity per site:")
        print(trained["test_fidelity"])
        print("Mean test fidelity:", trained["test_mean_fidelity"])
        print("Test errors per site:", trained["test_errors_per_site"])
        print("Total test errors:", trained["test_total_errors"])

        print("\nSaved model:", out_model)


if __name__ == "__main__":
    main()
