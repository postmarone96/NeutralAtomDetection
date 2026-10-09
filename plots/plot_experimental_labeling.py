from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.stats import norm

from real_data.label_real_data import (
    choose_site_peaks,
    find_peak_candidates,
    fit_two_gaussian_threshold,
)
import constants


h5_path = project_root / "real_data" / "output" / "runs_4043_fast_imaging_labeled.h5"
plot_path = project_root / "report" / "figures" / "experimental_labeling.png"

peak_min_distance = constants.labeling_peak_min_distance
peak_smooth_sigma = constants.labeling_peak_smooth_sigma

figsize = (10.0, 8.4)
dpi = 200
crop_radius = 12
n_score_bins = 45
n_peak_bins = 40
show = False

ink = constants.figure_ink
ink_secondary = constants.figure_ink_secondary
grid_color = constants.figure_grid
site_color = constants.figure_site
site_dark_color = constants.figure_site_dark
fit_color = constants.figure_fit
neutral_color = constants.figure_neutral


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink_secondary)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=ink_secondary, labelsize=9, width=0.8)
    ax.grid(axis="y", color=grid_color, linewidth=0.7)
    ax.set_axisbelow(True)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=11, color=ink)


def load_dataset():
    with h5py.File(h5_path, "r") as f:
        return {
            "avg_image": f["average_label_image"][:],
            "sites": f["sites"][:],
            "fit": f["site_fit_parameters"][:],
            "scores": f["label_scores"][:],
            "labels": f["labels"][:],
            "thresholds": f["label_thresholds"][:],
        }


def plot_site_detection(ax, avg_image, candidates, kept, sites):
    low, high = np.percentile(avg_image, (1.0, 99.7))
    ax.imshow(avg_image, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    ax.scatter(
        candidates[~kept, 0],
        candidates[~kept, 1],
        s=14,
        marker="x",
        color="white",
        linewidths=0.6,
    )
    ax.scatter(
        sites[:, 0],
        sites[:, 1],
        s=46,
        facecolors="none",
        edgecolors=site_color,
        linewidths=1.0,
    )
    ax.set_xlabel("x (pixel)", fontsize=9, color=ink_secondary)
    ax.set_ylabel("y (pixel)", fontsize=9, color=ink_secondary)
    ax.tick_params(colors=ink_secondary, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Line2D([], [], marker="x", linestyle="", color=neutral_color, markersize=5, markeredgewidth=0.9),
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor=site_color, markersize=7, markeredgewidth=1.0,
        ),
    ]
    labels = [f"rejected candidate ({int((~kept).sum())})", f"site ({sites.shape[0]})"]
    legend = ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9
    )
    for text in legend.get_texts():
        text.set_color(ink)
    panel_title(ax, "a", "Peaks in the averaged first image")


def plot_peak_heights(ax, values, kept, threshold, params):
    edges = np.linspace(values.min(), values.max(), n_peak_bins + 1)
    width = edges[1] - edges[0]
    ax.hist(values[~kept], bins=edges, color=neutral_color, edgecolor="white", linewidth=0.6, label="no site")
    ax.hist(values[kept], bins=edges, color=site_color, edgecolor="white", linewidth=0.6, label="site")

    grid = np.linspace(edges[0], edges[-1], 400)
    for mean, sigma, weight, color in zip(
        params["means"], params["sigmas"], params["weights"], (ink_secondary, site_dark_color)
    ):
        ax.plot(
            grid,
            values.size * width * weight * norm.pdf(grid, mean, sigma),
            color=color,
            linewidth=1.5,
        )

    ax.axvline(threshold, color=ink, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.55,
        " threshold",
        color=ink,
        fontsize=9,
        ha="left",
        va="center",
    )
    ax.set_xlabel("smoothed peak height (counts)", fontsize=9, color=ink_secondary)
    ax.set_ylabel("candidates", fontsize=9, color=ink_secondary)
    style_axis(ax)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(ink)
    panel_title(ax, "b", "Site selection")


def plot_site_fit(ax, avg_image, x0, y0, sigma, site_index):
    xc, yc = int(round(x0)), int(round(y0))
    x1, x2 = xc - crop_radius, xc + crop_radius + 1
    y1, y2 = yc - crop_radius, yc + crop_radius + 1
    patch = avg_image[y1:y2, x1:x2]
    ax.imshow(
        patch,
        cmap="gray",
        interpolation="nearest",
        extent=(x1 - 0.5, x2 - 0.5, y2 - 0.5, y1 - 0.5),
    )
    for k in (1, 2):
        ax.add_patch(
            Circle((x0, y0), k * sigma, fill=False, edgecolor=fit_color, linewidth=1.4)
        )
    ax.plot([x0], [y0], marker="+", color=fit_color, markersize=9, markeredgewidth=1.4)
    for k, text in ((1, r"$\sigma$"), (2, r"$2\sigma$")):
        ax.text(
            x0,
            y0 - k * sigma - 0.2,
            text,
            color=fit_color,
            fontsize=10,
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("x (pixel)", fontsize=9, color=ink_secondary)
    ax.set_ylabel("y (pixel)", fontsize=9, color=ink_secondary)
    ax.tick_params(colors=ink_secondary, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_title(ax, "c", f"Gaussian fit of site {site_index}")


def plot_score_histogram(ax, scores, threshold, params):
    edges = np.linspace(scores.min(), scores.max(), n_score_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    empty = scores < threshold
    occupied = ~empty
    ax.hist(scores[empty], bins=edges, color=neutral_color, edgecolor="white", linewidth=0.6, label="empty")
    ax.hist(scores[occupied], bins=edges, color=site_color, edgecolor="white", linewidth=0.6, label="occupied")

    grid = np.linspace(edges[0], edges[-1], 400)
    for mean, sigma, weight, color in zip(
        params["means"], params["sigmas"], params["weights"], (ink_secondary, site_dark_color)
    ):
        ax.plot(
            grid,
            scores.size * width * weight * norm.pdf(grid, mean, sigma),
            color=color,
            linewidth=1.5,
        )

    ax.axvline(threshold, color=ink, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.55,
        " threshold",
        color=ink,
        fontsize=9,
        ha="left",
        va="center",
    )
    ax.set_xlabel("score: Gaussian-weighted sum of the first channel", fontsize=9, color=ink_secondary)
    ax.set_ylabel("images", fontsize=9, color=ink_secondary)
    style_axis(ax)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(ink)
    panel_title(ax, "d", f"Occupation labels of site {site_index}")


data = load_dataset()
avg_image = data["avg_image"]
sites = data["sites"]

candidates, peak_values = find_peak_candidates(
    avg_image=avg_image,
    min_distance=peak_min_distance,
    smooth_sigma=peak_smooth_sigma,
)
selected, site_detection = choose_site_peaks(candidates=candidates, values=peak_values)
kept = peak_values >= site_detection["selected_peak_min"]
if int(kept.sum()) != sites.shape[0]:
    raise ValueError(
        f"peak selection gives {int(kept.sum())} sites, the dataset has {sites.shape[0]}"
    )

peak_threshold, peak_params = fit_two_gaussian_threshold(peak_values)
if not np.array_equal(peak_values >= peak_threshold, kept):
    raise ValueError("the mixture threshold on the peak heights does not reproduce the retained sites")

center = np.array([avg_image.shape[1] / 2, avg_image.shape[0] / 2])
site_index = int(np.argmin(np.linalg.norm(sites - center, axis=1)))
x0, y0, sigma = data["fit"][site_index, :3]

site_scores = data["scores"][:, site_index]
threshold, params = fit_two_gaussian_threshold(site_scores)
if not np.isclose(threshold, data["thresholds"][site_index], rtol=1e-4):
    raise ValueError(
        f"refit threshold {threshold} differs from stored {data['thresholds'][site_index]}"
    )
occupancy = float((site_scores >= threshold).mean())

fig = plt.figure(figsize=figsize)
grid_spec = fig.add_gridspec(
    2, 2, width_ratios=(1.15, 1.0), height_ratios=(1.0, 1.0), hspace=0.42, wspace=0.28
)
ax_sites = fig.add_subplot(grid_spec[0, 0])
ax_peaks = fig.add_subplot(grid_spec[0, 1])
ax_fit = fig.add_subplot(grid_spec[1, 0])
ax_scores = fig.add_subplot(grid_spec[1, 1])

plot_site_detection(ax_sites, avg_image, candidates, kept, sites)
plot_peak_heights(ax_peaks, peak_values, kept, peak_threshold, peak_params)
plot_site_fit(ax_fit, avg_image, x0, y0, sigma, site_index)
plot_score_histogram(ax_scores, site_scores, threshold, params)

plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
print(f"Saved {plot_path}")
print(
    f"site {site_index}: center ({x0:.1f}, {y0:.1f}), sigma {sigma:.2f}, "
    f"threshold {threshold:.1f}, occupancy {occupancy:.3f}"
)
if show:
    plt.show()
plt.close(fig)
