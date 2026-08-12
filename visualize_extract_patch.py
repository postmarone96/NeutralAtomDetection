import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from mf_site import extract_patches


DEFAULT_H5_PATH = Path(
    "/Users/marone/PycharmProjects/neutral-atom-imaging-simulation-main/"
    "test_dataset_1/qubit_dataset_sim_0.036.h5"
)


def preprocess_image_for_display(
    raw_image,
    preprocess,
    all_images=None,
    unattenuated_image=None,
):
    if preprocess == "none":
        return raw_image.astype(np.float32), "raw image", {}

    if preprocess == "mean-scale":
        if all_images is None:
            raise ValueError("mean-scale preprocessing needs all images from the H5 file.")
        mean_value = float(all_images.mean())
        centered = raw_image - mean_value
        scale = float(np.max(np.abs(all_images - mean_value)))
        if scale == 0:
            scale = 1.0
        return (centered / scale).astype(np.float32), "mean-scale preprocessed image", {
            "mean": mean_value,
            "scale": scale,
        }

    if preprocess == "image-mean-scale":
        mean_value = float(raw_image.mean())
        centered = raw_image - mean_value
        scale = float(np.max(np.abs(centered)))
        if scale == 0:
            scale = 1.0
        return (centered / scale).astype(np.float32), "single-image mean-scale preprocessed image", {
            "mean": mean_value,
            "scale": scale,
        }

    if preprocess == "unattenuated-minus-image":
        if unattenuated_image is None:
            raise ValueError(
                "unattenuated-minus-image preprocessing needs an H5 dataset named by --unattenuated-key."
            )
        return (unattenuated_image - raw_image).astype(np.float32), "unattenuated - image", {}

    raise ValueError(f"Unknown preprocessing mode: {preprocess}")


def build_sample_points(sites, s, H, W, device):
    x = sites[:, 0].to(device)
    y = sites[:, 1].to(device)

    offsets = torch.arange(s, device=device).float() - (s - 1) / 2
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")

    sample_x = x[:, None, None] + dx[None, :, :]
    sample_y = y[:, None, None] + dy[None, :, :]

    grid_x = 2.0 * sample_x / (W - 1) - 1.0
    grid_y = 2.0 * sample_y / (H - 1) - 1.0

    return offsets, dx, dy, sample_x, sample_y, grid_x, grid_y


def extract_square_window_patch(image, site_x, site_y, s):
    """
    Match train_square_window_sweep.extract_site_patches_padded for one image/site.

    This method rounds the site to the nearest integer pixel, pads the image with
    zeros, and cuts an s x s block of actual pixel values.
    """
    half = s // 2
    pad = half + 1
    padded = np.pad(image, pad_width=((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    rounded_x = int(round(float(site_x)))
    rounded_y = int(round(float(site_y)))

    padded_x = rounded_x + pad
    padded_y = rounded_y + pad

    x0_padded = padded_x - half
    x1_padded = x0_padded + s
    y0_padded = padded_y - half
    y1_padded = y0_padded + s

    patch = padded[y0_padded:y1_padded, x0_padded:x1_padded]

    x0 = rounded_x - half
    x1 = x0 + s
    y0 = rounded_y - half
    y1 = y0 + s

    cols = np.arange(x0, x1)
    rows = np.arange(y0, y1)
    cut_x, cut_y = np.meshgrid(cols, rows)

    return patch, rounded_x, rounded_y, cut_x, cut_y, x0, y0


def draw_bilinear_geometry(ax, sample_x, sample_y, site_x, site_y):
    for row in range(sample_x.shape[0]):
        ax.plot(sample_x[row], sample_y[row], color="yellow", linewidth=1.0, alpha=0.8)
    for col in range(sample_x.shape[1]):
        ax.plot(sample_x[:, col], sample_y[:, col], color="yellow", linewidth=1.0, alpha=0.8)

    ax.scatter(sample_x.ravel(), sample_y.ravel(), s=36, c="yellow", edgecolors="black")
    ax.scatter([site_x], [site_y], s=90, c="red", marker="+")
    ax.add_patch(
        Rectangle(
            (sample_x.min() - 0.5, sample_y.min() - 0.5),
            sample_x.max() - sample_x.min() + 1.0,
            sample_y.max() - sample_y.min() + 1.0,
            fill=False,
            edgecolor="red",
            linewidth=1.6,
        )
    )


def draw_square_window_geometry(ax, cut_x, cut_y, rounded_x, rounded_y, x0, y0, s):
    for row in range(cut_x.shape[0]):
        ax.plot(cut_x[row], cut_y[row], color="cyan", linewidth=1.0, alpha=0.8)
    for col in range(cut_x.shape[1]):
        ax.plot(cut_x[:, col], cut_y[:, col], color="cyan", linewidth=1.0, alpha=0.8)

    ax.scatter(cut_x.ravel(), cut_y.ravel(), s=30, c="cyan", edgecolors="black")
    ax.scatter([rounded_x], [rounded_y], s=90, c="magenta", marker="+")
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            s,
            s,
            fill=False,
            edgecolor="cyan",
            linewidth=1.6,
            linestyle="--",
        )
    )


def annotate_patch_values(ax, patch):
    if patch.shape[0] > 12:
        return

    midpoint = 0.5 * (float(patch.min()) + float(patch.max()))
    for row in range(patch.shape[0]):
        for col in range(patch.shape[1]):
            color = "white" if patch[row, col] < midpoint else "black"
            ax.text(
                col,
                row,
                f"{patch[row, col]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize how mf_site.extract_patches samples one patch from one image."
    )
    parser.add_argument("h5_path", nargs="?", default=str(DEFAULT_H5_PATH))
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--site-index", type=int, default=2)
    parser.add_argument("--s", type=int, default=2, help="Patch side length.")
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--unattenuated-key", default="unattenuated_images")
    parser.add_argument("--label-key", default="labels")
    parser.add_argument("--site-key", default="sites")
    parser.add_argument(
        "--preprocess",
        choices=["none", "mean-scale", "image-mean-scale", "unattenuated-minus-image"],
        default="none",
        help=(
            "Image values to extract patches from. mean-scale matches the sweep scripts "
            "except it estimates mean/scale from the whole H5 file for visualization."
        ),
    )
    parser.add_argument(
        "--patch-scale",
        choices=["image", "patch"],
        default="image",
        help="Use image-wide color limits for patches, or autoscale each patch for contrast.",
    )
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} does not exist. Pass the H5 path as the first argument.")

    with h5py.File(h5_path, "r") as f:
        raw_image = f[args.image_key][args.image_index].astype(np.float32)
        sites = f[args.site_key][:].astype(np.float32)
        all_images = None
        if args.preprocess == "mean-scale":
            all_images = f[args.image_key][:].astype(np.float32)
        unattenuated_image = None
        if args.preprocess == "unattenuated-minus-image":
            if args.unattenuated_key not in f:
                raise KeyError(
                    f"{args.unattenuated_key!r} was not found in {h5_path}. "
                    "Use --preprocess none or pass the correct --unattenuated-key."
                )
            unattenuated_image = f[args.unattenuated_key][args.image_index].astype(np.float32)
        label = None
        if args.label_key in f:
            label = f[args.label_key][args.image_index, args.site_index]

    image, image_title, preprocess_info = preprocess_image_for_display(
        raw_image,
        args.preprocess,
        all_images=all_images,
        unattenuated_image=unattenuated_image,
    )

    image_t = torch.from_numpy(image)[None]
    sites_t = torch.from_numpy(sites)
    Batch, H, W = image_t.shape

    if not 0 <= args.site_index < sites.shape[0]:
        raise IndexError(f"--site-index must be in [0, {sites.shape[0] - 1}]")

    offsets, dx, dy, sample_x, sample_y, grid_x, grid_y = build_sample_points(
        sites_t, args.s, H, W, image_t.device
    )

    site_x, site_y = sites[args.site_index]
    patch = extract_patches(image_t, sites_t, args.s)[0, args.site_index].reshape(args.s, args.s)
    square_patch, rounded_x, rounded_y, cut_x, cut_y, square_x0, square_y0 = extract_square_window_patch(
        image, site_x, site_y, args.s
    )

    sx = sample_x[args.site_index].cpu().numpy()
    sy = sample_y[args.site_index].cpu().numpy()
    patch_np = patch.cpu().numpy()
    zoom = args.zoom if args.zoom is not None else max(args.s * 2.0, 8.0)
    image_vmin = float(np.min(image))
    image_vmax = float(np.max(image))
    if image_vmin == image_vmax:
        image_vmax = image_vmin + 1.0
    patch_vmin = image_vmin if args.patch_scale == "image" else None
    patch_vmax = image_vmax if args.patch_scale == "image" else None

    print(f"H5: {h5_path}")
    print(f"image shape: {image.shape}")
    print(f"preprocess: {args.preprocess} ({image_title})")
    for key, value in preprocess_info.items():
        print(f"preprocess {key}: {value}")
    print(f"display color scale: vmin={image_vmin}, vmax={image_vmax}")
    print(f"patch color scale: {args.patch_scale}")
    print(f"site {args.site_index}: x={site_x:.3f}, y={site_y:.3f}, label={label}")
    print(f"s: {args.s}")
    print("offsets:", offsets.cpu().numpy())
    print("sample_x:")
    print(sx)
    print("sample_y:")
    print(sy)
    print("normalized grid_x range:", float(grid_x[args.site_index].min()), float(grid_x[args.site_index].max()))
    print("normalized grid_y range:", float(grid_y[args.site_index].min()), float(grid_y[args.site_index].max()))
    print("bilinear extract_patches values:")
    print(patch_np)
    print(f"square window rounded site: x={rounded_x}, y={rounded_y}")
    print("square window x pixel indices:")
    print(cut_x)
    print("square window y pixel indices:")
    print(cut_y)
    print("square window pixel values:")
    print(square_patch)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax_full, ax_zoom, ax_offsets, ax_bilinear_patch, ax_square_pixels, ax_square_patch = axes.ravel()

    ax_full.imshow(image, cmap=args.cmap, origin="upper", vmin=image_vmin, vmax=image_vmax)
    ax_full.scatter(sites[:, 0], sites[:, 1], s=14, c="white", marker="x")
    draw_bilinear_geometry(ax_full, sx, sy, site_x, site_y)
    draw_square_window_geometry(ax_full, cut_x, cut_y, rounded_x, rounded_y, square_x0, square_y0, args.s)
    ax_full.set_title(f"full {image_title}: yellow=bilinear, cyan=integer square")
    ax_full.set_xlim(-0.5, W - 0.5)
    ax_full.set_ylim(H - 0.5, -0.5)

    ax_zoom.imshow(image, cmap=args.cmap, origin="upper", vmin=image_vmin, vmax=image_vmax)
    draw_bilinear_geometry(ax_zoom, sx, sy, site_x, site_y)
    draw_square_window_geometry(ax_zoom, cut_x, cut_y, rounded_x, rounded_y, square_x0, square_y0, args.s)
    ax_zoom.set_title("zoom: yellow sampled, cyan cut pixels")
    ax_zoom.set_xlim(site_x - zoom, site_x + zoom)
    ax_zoom.set_ylim(site_y + zoom, site_y - zoom)

    ax_offsets.scatter(dx.cpu().numpy().ravel(), dy.cpu().numpy().ravel(), s=50, c=np.arange(args.s * args.s))
    ax_offsets.scatter([0], [0], s=90, c="red", marker="+")
    ax_offsets.set_title("bilinear offsets before adding site")
    ax_offsets.set_xlabel("dx")
    ax_offsets.set_ylabel("dy")
    ax_offsets.invert_yaxis()
    ax_offsets.grid(True, alpha=0.3)

    im_bilinear = ax_bilinear_patch.imshow(
        patch_np,
        cmap=args.cmap,
        origin="upper",
        vmin=patch_vmin,
        vmax=patch_vmax,
    )
    annotate_patch_values(ax_bilinear_patch, patch_np)
    ax_bilinear_patch.set_title("extract_patches: bilinear values")
    fig.colorbar(im_bilinear, ax=ax_bilinear_patch, fraction=0.046, pad=0.04)

    ax_square_pixels.scatter(cut_x.ravel(), cut_y.ravel(), s=50, c=np.arange(args.s * args.s))
    ax_square_pixels.scatter([rounded_x], [rounded_y], s=90, c="magenta", marker="+")
    ax_square_pixels.add_patch(
        Rectangle(
            (square_x0 - 0.5, square_y0 - 0.5),
            args.s,
            args.s,
            fill=False,
            edgecolor="cyan",
            linewidth=1.6,
            linestyle="--",
        )
    )
    ax_square_pixels.set_title("integer square window pixels")
    ax_square_pixels.set_xlabel("x pixel index")
    ax_square_pixels.set_ylabel("y pixel index")
    ax_square_pixels.invert_yaxis()
    ax_square_pixels.grid(True, alpha=0.3)

    im_square = ax_square_patch.imshow(
        square_patch,
        cmap=args.cmap,
        origin="upper",
        vmin=patch_vmin,
        vmax=patch_vmax,
    )
    annotate_patch_values(ax_square_patch, square_patch)
    ax_square_patch.set_title("square window: cut pixel values")
    fig.colorbar(im_square, ax=ax_square_patch, fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_aspect("equal", adjustable="box")

    label_text = "" if label is None else f", label={label}"
    fig.suptitle(
        f"{image_title}: image {args.image_index}, site {args.site_index}, "
        f"s={args.s}{label_text}"
    )
    fig.tight_layout()

    if args.save is not None:
        fig.savefig(args.save, dpi=200)
        print(f"saved: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
