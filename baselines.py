import h5py
import torch
import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import curve_fit, minimize_scalar
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mf_site import (
    best_threshold_by_fidelity,
    classification_fidelity,
)

from utils import H5Dataset, get_device, extract_patches, path_from_root
import constants

# ---------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------

def tune_thresholds(scores, labels):
    n_qubits = labels.shape[1]

    thresholds = np.zeros(n_qubits, dtype=np.float32)
    fidelities = np.zeros(n_qubits, dtype=np.float32)

    for q in range(n_qubits):
        threshold, fid = best_threshold_by_fidelity(
            scores[:, q],
            labels[:, q],
        )
        thresholds[q] = threshold
        fidelities[q] = fid

    return thresholds, fidelities


# ---------------------------------------------------------------------
# Average image and Gaussian fitting
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_average_image(h5_path, indices, batch_size=256, num_workers=4):
    loader = DataLoader(
        H5Dataset(h5_path, indices=indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    sum_image = None
    count = 0
    sites0 = None

    for images, labels, sites in loader:
        images = images.float()

        if sum_image is None:
            sum_image = images.sum(dim=0).double()
            sites0 = sites[0].cpu().numpy()
        else:
            sum_image += images.sum(dim=0).double()

        count += images.shape[0]

    avg_image = (sum_image / count).cpu().numpy()
    return avg_image, sites0


def circular_gaussian(coords, amp, x0, y0, sigma, offset):
    y, x = coords
    return offset + amp * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    )


def fit_sigmas_from_average_image(avg_image, sites, fit_size=constants.baselines_fit_size):
    """
    Fit one circular 2D Gaussian around each qubit site.

    sites are assumed to be stored as (x, y).
    avg_image is indexed as avg_image[y, x].
    """
    sigmas = []

    H, W = avg_image.shape
    r = fit_size // 2

    for site in sites:
        x0_guess = float(site[0])
        y0_guess = float(site[1])

        xc = int(round(x0_guess))
        yc = int(round(y0_guess))

        y1 = max(0, yc - r)
        y2 = min(H, yc + r + 1)
        x1 = max(0, xc - r)
        x2 = min(W, xc + r + 1)

        patch = avg_image[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]

        amp_guess = float(patch.max() - patch.min())
        offset_guess = float(patch.min())
        sigma_guess = 2.0

        p0 = [
            amp_guess,
            x0_guess,
            y0_guess,
            sigma_guess,
            offset_guess,
        ]

        bounds = (
            [0.0, x1, y1, 0.2, -np.inf],
            [np.inf, x2, y2, 10.0, np.inf],
        )

        try:
            popt, _ = curve_fit(
                circular_gaussian,
                (yy.ravel(), xx.ravel()),
                patch.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
            sigma = float(popt[3])
        except RuntimeError:
            sigma = sigma_guess

        sigmas.append(sigma)

    return np.asarray(sigmas, dtype=np.float32)


def fit_shared_sigma_from_average_image(avg_image, sites):
    """One PSF sigma for all sites, fitted to the whole average image at once.

    Every site is a Gaussian on its site position with its own brightness, plus one
    flat background. Only sigma is searched; brightnesses and background are solved
    linearly for each candidate sigma.
    """
    height, width = avg_image.shape
    yy, xx = np.mgrid[0:height, 0:width]
    sites = np.asarray(sites, dtype=np.float64)
    squared = (xx.reshape(-1, 1) - sites[:, 0]) ** 2 + (yy.reshape(-1, 1) - sites[:, 1]) ** 2
    image = np.asarray(avg_image, dtype=np.float64).ravel()
    background = np.ones((image.size, 1))

    def residual(sigma):
        design = np.hstack([np.exp(-squared / (2 * sigma ** 2)), background])
        coefficients = np.linalg.lstsq(design, image, rcond=None)[0]
        return float(np.sum((design @ coefficients - image) ** 2))

    return float(minimize_scalar(residual, bounds=(0.2, 10.0), method="bounded").x)
# ---------------------------------------------------------------------
# Square scoring
# ---------------------------------------------------------------------

@torch.no_grad()
def calc_square_scores(dataloader, s, device="cpu"):
    all_scores = []
    all_labels = []

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()

        # Assumes sites are constant across the dataset.
        sites = sites[0].to(device)

        X = extract_patches(images, sites, s)  # (B, Q, s*s)
        scores = X.sum(dim=2)                 # (B, Q)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def square_with_s_sweep(
        h5_path,
        val_idx,
        s_values=range(2, 15),
        batch_size=256,
        num_workers=4,
        device="cpu"):

    val_loader = DataLoader(
        H5Dataset(h5_path, indices=val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    n_qubits = H5Dataset(h5_path, indices=[0]).sites.shape[0]

    best = {
        "method": "square",
        "s": np.zeros(n_qubits, dtype=int),
        "threshold": np.zeros(n_qubits, dtype=np.float32),
        "val_fidelity": np.full(n_qubits, -np.inf, dtype=np.float32),
        "history": [],
    }

    progress = tqdm(s_values, desc="Square s sweep", unit="s", leave=True)
    for s in progress:
        scores, labels = calc_square_scores(
            val_loader,
            s=s,
            device=device,
        )

        thresholds, fidelities = tune_thresholds(scores, labels)

        for q in range(n_qubits):
            if fidelities[q] > best["val_fidelity"][q]:
                best["s"][q] = s
                best["threshold"][q] = thresholds[q]
                best["val_fidelity"][q] = fidelities[q]

        best["history"].append({
            "s": s,
            "thresholds": thresholds.copy(),
            "fidelities": fidelities.copy(),
            "mean_fidelity": float(np.mean(fidelities)),
        })

        # progress.set_postfix(mean_val_fidelity=f"{np.mean(fidelities):.8f}")

    return best


# ---------------------------------------------------------------------
# Gaussian scoring
# ---------------------------------------------------------------------

def gaussian_kernel(s, sigma, device="cpu", normalize=False):
    offsets = torch.arange(s, device=device).float() - (s - 1) / 2
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")

    g = torch.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2)))

    if normalize:
        g = g / g.sum()

    return g.reshape(-1)  # (s*s,)


@torch.no_grad()
def calc_gaussian_scores(dataloader, s, sigmas, device="cpu", normalize=False):
    all_scores = []
    all_labels = []

    sigmas = np.asarray(sigmas)
    n_qubits = len(sigmas)

    kernels = [
        gaussian_kernel(s, float(sigmas[q]), device=device, normalize=normalize)
        for q in range(n_qubits)
    ]

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()

        # Assumes sites are constant across the dataset.
        sites = sites[0].to(device)

        X = extract_patches(images, sites, s)  # (B, Q, s*s)

        B = images.shape[0]
        scores = torch.zeros((B, n_qubits), device=device)

        for q in range(n_qubits):
            scores[:, q] = torch.einsum("bd,d->b", X[:, q, :], kernels[q])

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def gaussian_with_s_sweep(
        h5_path,
        val_idx,
        sigmas,
        s_values=range(2, 15),
        batch_size=256,
        num_workers=4,
        device="cpu",
        normalize=False):

    val_loader = DataLoader(
        H5Dataset(h5_path, indices=val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    sigmas = np.asarray(sigmas, dtype=np.float32)
    n_qubits = len(sigmas)

    best = {
        "method": "gaussian",
        "s": np.zeros(n_qubits, dtype=int),
        "sigma": sigmas.copy(),
        "threshold": np.zeros(n_qubits, dtype=np.float32),
        "val_fidelity": np.full(n_qubits, -np.inf, dtype=np.float32),
        "history": [],
        "normalize_kernel": normalize,
    }

    progress = tqdm(s_values, desc="Gaussian s sweep", unit="s", leave=True)
    for s in progress:
        scores, labels = calc_gaussian_scores(
            val_loader,
            s=s,
            sigmas=sigmas,
            device=device,
            normalize=normalize,
        )

        thresholds, fidelities = tune_thresholds(scores, labels)

        for q in range(n_qubits):
            if fidelities[q] > best["val_fidelity"][q]:
                best["s"][q] = s
                best["threshold"][q] = thresholds[q]
                best["val_fidelity"][q] = fidelities[q]

        best["history"].append({
            "s": s,
            "sigma": sigmas.copy(),
            "thresholds": thresholds.copy(),
            "fidelities": fidelities.copy(),
            "mean_fidelity": float(np.mean(fidelities)),
        })

        # progress.set_postfix(mean_val_fidelity=f"{np.mean(fidelities):.8f}")

    return best


# ---------------------------------------------------------------------
# Evaluation using best per-qubit s and thresholds
# ---------------------------------------------------------------------

@torch.no_grad()
def calc_square_scores_best(dataloader, best, device="cpu"):
    all_scores = []
    all_preds = []
    all_labels = []

    n_qubits = len(best["s"])

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()
        sites = sites[0].to(device)

        B = images.shape[0]
        batch_scores = torch.zeros((B, n_qubits), device=device)

        unique_s_values = sorted(set(int(s) for s in best["s"]))

        for s in unique_s_values:
            X = extract_patches(images, sites, s)

            qs = np.where(best["s"] == s)[0]
            for q in qs:
                batch_scores[:, q] = X[:, q, :].sum(dim=1)

        thresholds = torch.tensor(best["threshold"], device=device)
        batch_preds = batch_scores >= thresholds

        all_scores.append(batch_scores.cpu())
        all_preds.append(batch_preds.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_scores).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


@torch.no_grad()
def calc_gaussian_scores_best(dataloader, best, device="cpu"):
    all_scores = []
    all_preds = []
    all_labels = []

    n_qubits = len(best["s"])
    normalize = bool(best.get("normalize_kernel", False))

    kernel_cache = {}
    for q in range(n_qubits):
        s = int(best["s"][q])
        sigma = float(best["sigma"][q])
        kernel_cache[q] = gaussian_kernel(
            s,
            sigma,
            device=device,
            normalize=normalize,
        )

    for images, labels, sites in dataloader:
        images = images.to(device).float()
        labels = labels.float()
        sites = sites[0].to(device)

        B = images.shape[0]
        batch_scores = torch.zeros((B, n_qubits), device=device)

        unique_s_values = sorted(set(int(s) for s in best["s"]))

        for s in unique_s_values:
            X = extract_patches(images, sites, s)

            qs = np.where(best["s"] == s)[0]
            for q in qs:
                batch_scores[:, q] = torch.einsum(
                    "bd,d->b",
                    X[:, q, :],
                    kernel_cache[q],
                )

        thresholds = torch.tensor(best["threshold"], device=device)
        batch_preds = batch_scores >= thresholds

        all_scores.append(batch_scores.cpu())
        all_preds.append(batch_preds.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_scores).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(data_dir, model_dir):
    model_dir.mkdir(exist_ok=True)

    h5_paths = sorted(data_dir.glob("*.h5"))

    s_values = range(2, 15, 2)

    batch_size = 1
    num_workers = 8
    seed = 42

    normalize_gaussian_kernel = False

    for h5_path in h5_paths:
        square_out_model = model_dir / f"{h5_path.stem}_square.joblib"
        gaussian_out_model = model_dir / f"{h5_path.stem}_gaussian.joblib"

        with h5py.File(h5_path, "r") as f:
            N = f["images"].shape[0]

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)

        n_train = int(constants.split_train_fraction * N)
        n_val = int(constants.split_val_fraction * N)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        device = get_device()

        avg_image, sites = compute_average_image(
            h5_path=h5_path,
            indices=train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        fitted_sigmas = fit_sigmas_from_average_image(
            avg_image=avg_image,
            sites=sites,
            fit_size=constants.baselines_fit_size,
        )

        square_trained = square_with_s_sweep(
            h5_path=h5_path,
            val_idx=val_idx,
            s_values=s_values,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        square_save_obj = {
            **square_trained,
            "h5_path": path_from_root(h5_path),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "s_values": list(s_values),
        }

        gaussian_trained = gaussian_with_s_sweep(
            h5_path=h5_path,
            val_idx=val_idx,
            sigmas=fitted_sigmas,
            s_values=s_values,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            normalize=normalize_gaussian_kernel,
        )

        gaussian_save_obj = {
            **gaussian_trained,
            "h5_path": path_from_root(h5_path),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "s_values": list(s_values),
            "fitted_sigmas": fitted_sigmas,
        }

        test_loader = DataLoader(
            H5Dataset(h5_path, indices=test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )

        square_scores, square_preds, square_labels = calc_square_scores_best(
            test_loader,
            square_trained,
            device=device,
        )

        gaussian_scores, gaussian_preds, gaussian_labels = calc_gaussian_scores_best(
            test_loader,
            gaussian_trained,
            device=device,
        )

        square_test_fids = np.array([
            classification_fidelity(
                square_scores[:, q],
                square_labels[:, q],
                square_trained["threshold"][q],
            )
            for q in range(square_labels.shape[1])
        ])

        gaussian_test_fids = np.array([
            classification_fidelity(
                gaussian_scores[:, q],
                gaussian_labels[:, q],
                gaussian_trained["threshold"][q],
            )
            for q in range(gaussian_labels.shape[1])
        ])

        square_errors = square_preds != (square_labels > 0.5)
        gaussian_errors = gaussian_preds != (gaussian_labels > 0.5)

        square_save_obj["test_fidelity"] = square_test_fids
        square_save_obj["test_mean_fidelity"] = float(np.mean(square_test_fids))
        square_save_obj["test_errors_per_site"] = square_errors.sum(axis=0).astype(np.int64)
        square_save_obj["test_total_errors"] = int(square_errors.sum())
        square_save_obj["test_raw_accuracy"] = float(1.0 - square_errors.mean())

        gaussian_save_obj["test_fidelity"] = gaussian_test_fids
        gaussian_save_obj["test_mean_fidelity"] = float(np.mean(gaussian_test_fids))
        gaussian_save_obj["test_errors_per_site"] = gaussian_errors.sum(axis=0).astype(np.int64)
        gaussian_save_obj["test_total_errors"] = int(gaussian_errors.sum())
        gaussian_save_obj["test_raw_accuracy"] = float(1.0 - gaussian_errors.mean())

        joblib.dump(square_save_obj, square_out_model)
        joblib.dump(gaussian_save_obj, gaussian_out_model)

        print(
            f"Saved {square_out_model} | "
            f"mean F={square_save_obj['test_mean_fidelity']:.8f}"
        )
        print(
            f"Saved {gaussian_out_model} | "
            f"mean F={gaussian_save_obj['test_mean_fidelity']:.8f}"
        )


if __name__ == "__main__":
    main(
        data_dir = Path("/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/dataset_3x3"),
        model_dir = Path("models_3x3"),
    )
