from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

from baselines import (
    circular_gaussian,
    compute_average_image,
    fit_sigmas_from_average_image,
)


default_h5_path = Path(
    "/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/"
    "test_dataset_1/qubit_dataset_sim_0.01.h5"
)

site_index = 0
fit_size = 6
seed = 42
batch_size = 500
num_workers = 2
output_dir = Path("output_3x3")


def split_indices(n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def fit_gaussian_for_site(avg_image, site, fit_size=9):
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

    try:
        popt, _ = curve_fit(
            circular_gaussian,
            (yy.ravel(), xx.ravel()),
            patch.ravel(),
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        failed = False
    except RuntimeError:
        popt = np.asarray(p0, dtype=float)
        failed = True

    fitted = circular_gaussian((yy, xx), *popt)

    return {
        "patch": patch,
        "fitted": fitted,
        "residual": patch - fitted,
        "bounds": (x1, x2, y1, y2),
        "params": popt,
        "sigma": float(popt[3]),
        "failed": failed,
    }


def plot_overview(avg_image, sites, sigmas, fit_size, site_index, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    im = axes[0].imshow(avg_image, cmap="viridis")
    scatter = axes[0].scatter(
        sites[:, 0],
        sites[:, 1],
        c=sigmas,
        cmap="magma",
        s=35,
        edgecolors="white",
        linewidths=0.5,
    )

    r = fit_size // 2
    for x, y in sites:
        xc = int(round(float(x)))
        yc = int(round(float(y)))
        axes[0].add_patch(
            Rectangle(
                (xc - r - 0.5, yc - r - 0.5),
                fit_size,
                fit_size,
                fill=False,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.35,
            )
        )

    axes[0].scatter(
        [sites[site_index, 0]],
        [sites[site_index, 1]],
        marker="x",
        s=80,
        c="cyan",
        linewidths=2.0,
    )
    axes[0].set_title("Training average image with fitted sigma per site")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="average intensity")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="fitted sigma")

    axes[1].hist(sigmas, bins=min(20, max(5, sigmas.size // 2)), color="tab:blue", alpha=0.8)
    axes[1].axvline(np.mean(sigmas), color="black", linestyle="--", label=f"mean={np.mean(sigmas):.3f}")
    axes[1].set_xlabel("fitted sigma")
    axes[1].set_ylabel("number of sites")
    axes[1].set_title("Fitted sigma distribution")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print("Saved:", output_path)


def plot_site_detail(site_index, fit, output_path):
    patch = fit["patch"]
    fitted = fit["fitted"]
    residual = fit["residual"]
    amp, x0, y0, sigma, offset = fit["params"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()

    axes[0].imshow(patch, cmap="viridis")
    axes[0].set_title("average-image patch")

    axes[1].imshow(fitted, cmap="viridis")
    axes[1].set_title(f"fitted Gaussian, sigma={sigma:.3f}")

    im = axes[2].imshow(residual, cmap="coolwarm")
    axes[2].set_title("models_3x3 - fit")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    row = patch.shape[0] // 2
    axes[3].plot(patch[row], marker="o", label="models_3x3")
    axes[3].plot(fitted[row], marker="o", label="fit")
    axes[3].set_title("center row")
    axes[3].set_xlabel("local x pixel")
    axes[3].set_ylabel("intensity")
    axes[3].legend()

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    status = "fallback fit" if fit["failed"] else "curve_fit"
    fig.suptitle(
        f"Site {site_index}: {status}, amp={amp:.3g}, x0={x0:.3f}, "
        f"y0={y0:.3f}, sigma={sigma:.3f}, offset={offset:.3g}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print("Saved:", output_path)


def main():
    h5_path = Path(default_h5_path)
    output_dir.mkdir(exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]

    train_idx, _, _ = split_indices(n, seed=seed)

    avg_image, sites = compute_average_image(
        h5_path=h5_path,
        indices=train_idx,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if not 0 <= site_index < sites.shape[0]:
        raise IndexError(f"--site-index must be in [0, {sites.shape[0] - 1}]")

    sigmas = fit_sigmas_from_average_image(
        avg_image=avg_image,
        sites=sites,
        fit_size=fit_size,
    )

    selected_fit = fit_gaussian_for_site(
        avg_image=avg_image,
        site=sites[site_index],
        fit_size=fit_size,
    )

    stem = h5_path.stem
    overview_path = output_dir / f"{stem}_fit_sigmas_overview.png"
    detail_path = output_dir / f"{stem}_fit_sigma_site_{site_index}.png"

    plot_overview(
        avg_image=avg_image,
        sites=sites,
        sigmas=sigmas,
        fit_size=fit_size,
        site_index=site_index,
        output_path=overview_path,
    )
    plot_site_detail(site_index, selected_fit, detail_path)

    print("Fitted sigmas:")
    print(sigmas)
    print("Mean sigma:", float(np.mean(sigmas)))

    plt.show()


if __name__ == "__main__":
    main()
