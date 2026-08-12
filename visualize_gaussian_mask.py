from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit


H5_PATH = Path(
    "/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/"
    "test_dataset_1/qubit_dataset_sim_0.01.h5"
)
OUT_PATH = Path("output_3x3/gaussian_mask_explanation.png")
IMAGE_INDEX = 0
SITE_INDEX = 4
S = 8
FIT_SIZE = 9
SEED = 42
NORMALIZE_KERNEL = False


def circular_gaussian(coords, amp, x0, y0, sigma, offset):
    y, x = coords
    return offset + amp * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    )


def split_indices(n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def compute_average_image(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        images = f["images"]
        total = np.zeros(images.shape[1:], dtype=np.float64)

        for i in indices:
            total += images[int(i)]

    return (total / len(indices)).astype(np.float32)


def fit_sigma_for_site(avg_image, site, fit_size=9):
    h, w = avg_image.shape
    r = fit_size // 2

    x0_guess = float(site[0])
    y0_guess = float(site[1])

    xc = int(round(x0_guess))
    yc = int(round(y0_guess))

    y1 = max(0, yc - r)
    y2 = min(h, yc + r + 1)
    x1 = max(0, xc - r)
    x2 = min(w, xc + r + 1)

    patch = avg_image[y1:y2, x1:x2]
    yy, xx = np.mgrid[y1:y2, x1:x2]

    amp_guess = float(patch.max() - patch.min())
    offset_guess = float(patch.min())
    sigma_guess = 2.0

    p0 = [amp_guess, x0_guess, y0_guess, sigma_guess, offset_guess]
    bounds = (
        [0.0, x1, y1, 0.2, -np.inf],
        [np.inf, x2, y2, 10.0, np.inf],
    )

    popt, _ = curve_fit(
        circular_gaussian,
        (yy.ravel(), xx.ravel()),
        patch.ravel(),
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    fitted = circular_gaussian((yy, xx), *popt)
    sigma = float(popt[3])

    return sigma, patch, fitted, (x1, x2, y1, y2)


def gaussian_kernel(s, sigma, normalize=False):
    offsets = torch.arange(s).float() - (s - 1) / 2
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    kernel = torch.exp(-((xx ** 2 + yy ** 2) / (2 * sigma ** 2)))

    if normalize:
        kernel = kernel / kernel.sum()

    return kernel.numpy()


def extract_patch_grid_sample(image, site, s):
    h, w = image.shape
    x = float(site[0])
    y = float(site[1])

    offsets = torch.arange(s).float() - (s - 1) / 2
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")

    sample_x = x + dx
    sample_y = y + dy

    grid_x = 2.0 * sample_x / (w - 1) - 1.0
    grid_y = 2.0 * sample_y / (h - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]

    img = torch.from_numpy(image.astype(np.float32))[None, None]
    patch = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return patch[0, 0].numpy()


def score_images_for_site(images, site, s, kernel):
    scores = np.zeros(images.shape[0], dtype=np.float32)

    for i in range(images.shape[0]):
        patch = extract_patch_grid_sample(images[i], site, s)
        scores[i] = float(np.sum(patch * kernel))

    return scores


def classification_fidelity(scores, labels, threshold):
    pred = scores >= threshold
    target = labels > 0.5

    dark = target == 0
    bright = target == 1

    false_bright = np.mean(pred[dark] == 1) if np.any(dark) else 0.0
    false_dark = np.mean(pred[bright] == 0) if np.any(bright) else 0.0

    return 1.0 - 0.5 * (false_bright + false_dark)


def best_threshold(scores, labels):
    thresholds = np.linspace(float(scores.min()), float(scores.max()), 400)

    best_t = thresholds[0]
    best_f = -1.0

    for t in thresholds:
        fid = classification_fidelity(scores, labels, t)
        if fid > best_f:
            best_f = fid
            best_t = float(t)

    return best_t, best_f


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)

    with h5py.File(H5_PATH, "r") as f:
        n = f["images"].shape[0]
        image = f["images"][IMAGE_INDEX].astype(np.float32)
        sites = f["sites"][:].astype(np.float32)
        labels_all = f["labels"][:].astype(np.float32)
        images_all = f["images"][:].astype(np.float32)

    train_idx, val_idx, test_idx = split_indices(n, seed=SEED)

    site = sites[SITE_INDEX]
    avg_image = compute_average_image(H5_PATH, train_idx)
    sigma, fit_patch, fitted_patch, fit_bounds = fit_sigma_for_site(
        avg_image,
        site,
        fit_size=FIT_SIZE,
    )

    kernel = gaussian_kernel(S, sigma, normalize=NORMALIZE_KERNEL)
    patch = extract_patch_grid_sample(image, site, S)
    weighted_patch = patch * kernel
    score = float(weighted_patch.sum())
    image_vmin = float(avg_image.min())
    image_vmax = float(avg_image.max())

    val_scores = score_images_for_site(images_all[val_idx], site, S, kernel)
    val_labels = labels_all[val_idx, SITE_INDEX]
    threshold, val_fidelity = best_threshold(val_scores, val_labels)

    x, y = site
    half_boundary = S / 2

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()

    image_cmap_kwargs = {"cmap": "jet", "vmin": image_vmin, "vmax": image_vmax}

    axes[0].imshow(avg_image, **image_cmap_kwargs)
    axes[0].scatter([x], [y], c="white", s=40, marker="x")
    x1, x2, y1, y2 = fit_bounds
    axes[0].add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
        )
    )
    axes[0].set_title("Training-set average image + fit window")

    axes[1].imshow(fit_patch, **image_cmap_kwargs)
    axes[1].set_title(f"Average-image fit patch, site {SITE_INDEX}")

    axes[2].imshow(fitted_patch, **image_cmap_kwargs)
    axes[2].set_title(f"Fitted circular Gaussian, sigma={sigma:.3f}")

    axes[3].imshow(image, **image_cmap_kwargs)
    axes[3].scatter([x], [y], c="white", s=40, marker="x")
    axes[3].add_patch(
        plt.Rectangle(
            (x - half_boundary, y - half_boundary),
            S,
            S,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
        )
    )
    axes[3].set_title(f"Input image {IMAGE_INDEX}, Gaussian window s={S}")

    axes[4].imshow(kernel, cmap="jet")
    axes[4].set_title("Gaussian mask")

    axes[5].imshow(weighted_patch, **image_cmap_kwargs)
    axes[5].set_title(f"patch * mask, score={score:.4g}")

    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.05, top=0.92, wspace=0.12, hspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_ticks([])
    cbar.ax.text(0.5, 1.02, "higher", ha="center", va="bottom", transform=cbar.ax.transAxes)
    cbar.ax.text(0.5, -0.02, "lower", ha="center", va="top", transform=cbar.ax.transAxes)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(OUT_PATH, dpi=200)
    print("Saved:", OUT_PATH)

    dark = val_scores[val_labels == 0]
    bright = val_scores[val_labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dark, bins=50, alpha=0.6, label="dark")
    ax.hist(bright, bins=50, alpha=0.6, label="bright")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4g}")
    ax.set_title(
        f"Gaussian-mask validation scores, site {SITE_INDEX}\n"
        f"s={S}, sigma={sigma:.3f}, val F={val_fidelity:.6f}"
    )
    ax.set_xlabel("score = sum(patch * Gaussian mask)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()

    hist_path = OUT_PATH.with_name("gaussian_mask_score_histogram.png")
    fig.savefig(hist_path, dpi=200)
    print("Saved:", hist_path)

    plt.show()


if __name__ == "__main__":
    main()
