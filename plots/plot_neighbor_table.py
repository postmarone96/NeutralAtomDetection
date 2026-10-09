from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Bbox

from baselines import compute_average_image
from lightweight_cnn_site import neighbor_table
from utils import make_random_split
import constants


h5_path = project_root / "dataset_3x3" / "qubit_dataset_sim_0.01.h5"
plot_path = project_root / "report" / "figures" / "neighbor_table_site_0.png"

site_index = 0
split_seed = 42
crop_size = 10
figsize = (4.4, 8.4)
dpi = 250
view_min = -1.5
view_max = 24

ink = constants.figure_ink
ink_secondary = constants.figure_ink_secondary
site_color = constants.figure_site
own_color = constants.figure_fit
outside_color = constants.figure_ink_secondary
floor_fill = "#d9d9d6"

slot_names = {
    (-1, -1): "up-left",
    (0, -1): "up",
    (1, -1): "up-right",
    (-1, 0): "left",
    (1, 0): "right",
    (-1, 1): "down-left",
    (0, 1): "down",
    (1, 1): "down-right",
}


def crop_around(image, x, y):
    x0, y0 = int(round(x)), int(round(y))
    half = crop_size // 2
    return image[y0 - half:y0 + half, x0 - half:x0 + half]


def draw_noise_floor(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=floor_fill, edgecolor=outside_color, linewidth=2.5, linestyle=(0, (4, 2))))
    ax.text(0.5, 0.5, "noise floor", ha="center", va="center", fontsize=9.5, color=ink_secondary)
    ax.set_title(title, fontsize=10, color=ink)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_crop(ax, image, x, y, title, edge_color, title_color, vmin, vmax):
    ax.imshow(crop_around(image, x, y), cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=title_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(2.5)


with h5py.File(h5_path, "r") as f:
    n_images = int(f["images"].shape[0])
train_idx, _val_idx, _test_idx = make_random_split(n_samples=n_images, seed=split_seed)
avg_image, sites = compute_average_image(h5_path=h5_path, indices=train_idx, num_workers=0)

indices, _ = neighbor_table(sites)
directions = [tuple(direction) for direction in constants.tinycnn_neighbor_directions]
spacing = np.array([np.diff(np.unique(np.round(sites[:, axis], 3))).mean() for axis in (0, 1)])
own = sites[site_index]
vmin, vmax = float(avg_image.min()), float(avg_image.max())

fig = plt.figure(figsize=figsize)
outer = fig.add_gridspec(2, 1, height_ratios=(3.7, 4.1), hspace=0.27, left=0.13, right=0.98, top=0.965, bottom=0.005)
ax_map = fig.add_subplot(outer[0])
slot_grid = outer[1].subgridspec(3, 3, wspace=0.05, hspace=0.62)

ax_map.add_patch(Rectangle((view_min, view_min), view_max - view_min, view_max - view_min, facecolor=floor_fill, edgecolor="none", zorder=0))
ax_map.imshow(avg_image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", zorder=1)
ax_map.set_xlim(view_min, view_max)
ax_map.set_ylim(view_max, view_min)
ax_map.set_xticks(range(0, int(view_max), 5))
ax_map.set_yticks(range(0, int(view_max), 5))
for number, (x, y) in enumerate(sites):
    if view_min <= x < view_max and view_min <= y < view_max:
        ax_map.text(x + 1.2, y - 0.6, str(number), color="white", fontsize=10)

half = crop_size / 2
ax_map.add_patch(Rectangle((own[0] - half - 0.5, own[1] - half - 0.5), crop_size, crop_size, fill=False, edgecolor=own_color, linewidth=2.5))
for slot, (dx, dy) in enumerate(directions):
    target = own + np.array([dx, dy]) * spacing
    neighbor = int(indices[site_index, slot])
    if neighbor >= 0:
        ax_map.plot([own[0], target[0]], [own[1], target[1]], color=site_color, linewidth=1.6)
        ax_map.add_patch(Circle(target, 0.75, facecolor=site_color, edgecolor="white", linewidth=1.0, zorder=5))
    else:
        ax_map.plot([own[0], target[0]], [own[1], target[1]], color=outside_color, linewidth=1.2, linestyle=(0, (3, 3)))
        ax_map.add_patch(Circle(target, 0.75, facecolor="none", edgecolor=outside_color, linewidth=1.6, zorder=5))
    offset_y = 2.0 if dy <= 0 else 2.0
    ax_map.text(
        target[0], target[1] + offset_y, f"slot {slot}", fontsize=9.5, ha="center", va="center", color=ink,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=site_color if neighbor >= 0 else outside_color, linewidth=1.2), zorder=6,
    )
ax_map.plot([own[0]], [own[1]], marker="+", color=own_color, markersize=12, markeredgewidth=2.4, zorder=6)
n_real = int((indices[site_index] >= 0).sum())
# ax_map.set_title(
#     f"site {site_index}(orange box): {n_real} real neighbors (filled), {len(directions) - n_real} outside the grid (hollow)\n",    fontsize=11, color=ink, loc="left",
# )
ax_map.set_xlabel("x (pixel)", fontsize=10.5, color=ink_secondary)
ax_map.set_ylabel("y (pixel)", fontsize=10.5, color=ink_secondary)
ax_map.tick_params(colors=ink_secondary, labelsize=10)

slot_axes = {(-1, -1): (0, 0), (0, -1): (0, 1), (1, -1): (0, 2), (-1, 0): (1, 0), (1, 0): (1, 2), (-1, 1): (2, 0), (0, 1): (2, 1), (1, 1): (2, 2)}
for slot, direction in enumerate(directions):
    row, column = slot_axes[direction]
    ax = fig.add_subplot(slot_grid[row, column])
    neighbor = int(indices[site_index, slot])
    caption = f"slot {slot}"
    if neighbor >= 0:
        draw_crop(ax, avg_image, sites[neighbor, 0], sites[neighbor, 1], caption, site_color, ink, vmin, vmax)
        ax.text(0.06, 0.94, f"site {neighbor}", color="white", fontsize=9, ha="left", va="top", transform=ax.transAxes)
    else:
        draw_noise_floor(ax, caption)
ax = fig.add_subplot(slot_grid[1, 1])
draw_crop(ax, avg_image, own[0], own[1], f"site {site_index}", own_color, own_color, vmin, vmax)

fig.suptitle(
    f"The 8 neighbor slots that site {site_index} reads",
    fontsize=11, color=ink, x=0.555, y=0.998,
)
plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
tight = fig.get_tightbbox(renderer)
inches = fig.dpi_scale_trans.inverted()
axes_left = min(ax.get_window_extent(renderer).transformed(inches).x0 for ax in fig.axes)
axes_right = max(ax.get_window_extent(renderer).transformed(inches).x1 for ax in fig.axes)
right_padding = (axes_left - tight.x0) - (tight.x1 - axes_right)
fig.savefig(
    plot_path,
    dpi=dpi,
    bbox_inches=Bbox([[tight.x0, tight.y0], [tight.x1 + right_padding, tight.y1]]),
    facecolor="white",
)
print(f"Saved {plot_path}")
print(f"neighbors {indices[site_index]}")
plt.close(fig)
