import re
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle


h5_re = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5$")
default_target_data_dir = Path(
    "/Users/marone/PycharmProjects/"
    "neutral-atom-imaging-simulation-main/datasets_20x20"
)

target_h5 = None
exposure = None
output_dir = Path("output_3x3_transfer_to_20x20")
output_name = "transfer_region_map_3x3_to_20x20.png"
max_images = 200
no_image = False
show = False


def parse_h5_exposure(path):
    match = h5_re.fullmatch(path.name)
    if match is None:
        return None
    return float(match.group(1))


def find_target_h5(target_data_dir, exposure=None):
    candidates = []
    for path in sorted(target_data_dir.glob("*.h5")):
        exposure_s = parse_h5_exposure(path)
        if exposure_s is None:
            continue
        if exposure is not None and not np.isclose(exposure_s, exposure):
            continue
        candidates.append((exposure_s, path))

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: item[0])[0][1]


def load_sites_and_mean_image(h5_path, max_images):
    with h5py.File(h5_path, "r") as f:
        sites = np.asarray(f["sites"][:], dtype=np.float32)
        images = np.asarray(f["images"][:max_images], dtype=np.float32)
    return sites, images.mean(axis=0)


def synthetic_20x20_sites():
    xs, ys = np.meshgrid(np.arange(20, dtype=np.float32), np.arange(20, dtype=np.float32))
    return np.column_stack([xs.ravel(), ys.ravel()])


def site_region_map(sites, n_regions_per_axis=3):
    sites = np.asarray(sites, dtype=np.float32)
    x_values = np.sort(np.unique(sites[:, 0]))
    y_values = np.sort(np.unique(sites[:, 1]))

    x_groups = np.array_split(x_values, n_regions_per_axis)
    y_groups = np.array_split(y_values, n_regions_per_axis)

    x_to_group = {
        float(x): group_idx
        for group_idx, group in enumerate(x_groups)
        for x in group
    }
    y_to_group = {
        float(y): group_idx
        for group_idx, group in enumerate(y_groups)
        for y in group
    }

    col = np.asarray([x_to_group[float(x)] for x in sites[:, 0]], dtype=np.int64)
    row = np.asarray([y_to_group[float(y)] for y in sites[:, 1]], dtype=np.int64)
    return row * n_regions_per_axis + col, x_groups, y_groups


def coordinate_edges(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5], dtype=np.float32)

    midpoints = (values[:-1] + values[1:]) / 2.0
    first_edge = values[0] - (midpoints[0] - values[0])
    last_edge = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first_edge], midpoints, [last_edge]]).astype(np.float32)


def grid_region_array(x_values, y_values, x_groups, y_groups):
    x_to_group = {
        float(x): group_idx
        for group_idx, group in enumerate(x_groups)
        for x in group
    }
    y_to_group = {
        float(y): group_idx
        for group_idx, group in enumerate(y_groups)
        for y in group
    }

    region_grid = np.zeros((len(y_values), len(x_values)), dtype=np.int64)
    for row_idx, y in enumerate(y_values):
        for col_idx, x in enumerate(x_values):
            region_grid[row_idx, col_idx] = (
                y_to_group[float(y)] * len(x_groups) + x_to_group[float(x)]
            )
    return region_grid


def group_rect(group_x, group_y, x_edges, y_edges, x_index, y_index):
    x0 = x_edges[x_index[float(group_x[0])]]
    x1 = x_edges[x_index[float(group_x[-1])] + 1]
    y0 = y_edges[y_index[float(group_y[0])]]
    y1 = y_edges[y_index[float(group_y[-1])] + 1]
    return x0, y0, x1 - x0, y1 - y0


def draw_source_3x3(ax, cmap, norm):
    source_grid = np.arange(9, dtype=np.int64).reshape(3, 3)
    edges = np.arange(4)
    ax.pcolormesh(edges, edges, source_grid, cmap=cmap, norm=norm, shading="flat")

    for region in range(9):
        row, col = divmod(region, 3)
        ax.text(
            col + 0.5,
            row + 0.5,
            f"site {region + 1}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_title("3x3 source model")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()


def draw_target_20x20(ax, sites, mean_image, cmap, norm):
    region_for_site, x_groups, y_groups = site_region_map(sites)
    x_values = np.sort(np.unique(sites[:, 0]))
    y_values = np.sort(np.unique(sites[:, 1]))
    x_edges = coordinate_edges(x_values)
    y_edges = coordinate_edges(y_values)
    region_grid = grid_region_array(x_values, y_values, x_groups, y_groups)

    if mean_image is not None:
        ax.imshow(mean_image, cmap="gray", origin="upper")
        alpha = 0.38
    else:
        alpha = 0.78

    ax.pcolormesh(
        x_edges,
        y_edges,
        region_grid,
        cmap=cmap,
        norm=norm,
        shading="flat",
        alpha=alpha,
    )

    ax.scatter(
        sites[:, 0],
        sites[:, 1],
        s=12,
        c="black",
        linewidths=0,
        alpha=0.8,
        zorder=3,
    )

    x_index = {float(x): idx for idx, x in enumerate(x_values)}
    y_index = {float(y): idx for idx, y in enumerate(y_values)}
    counts = np.bincount(region_for_site, minlength=9)

    for region in range(9):
        row, col = divmod(region, 3)
        group_x = x_groups[col]
        group_y = y_groups[row]
        x0, y0, width, height = group_rect(group_x, group_y, x_edges, y_edges, x_index, y_index)

        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
                zorder=4,
            )
        )
        ax.text(
            x0 + width / 2.0,
            y0 + height / 2.0,
            f"site {region + 1}\n{counts[region]} targets",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=5,
        )

    ax.set_title("20x20 target regions")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[-1], y_edges[0])


def plot_transfer_region_map(sites, mean_image, output_path):
    colors = plt.get_cmap("tab10").colors[:9]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(10) - 0.5, cmap.N)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.4),
        gridspec_kw={"width_ratios": [1.0, 2.15]},
    )

    draw_source_3x3(axes[0], cmap, norm)
    draw_target_20x20(axes[1], sites, mean_image, cmap, norm)

    fig.suptitle("Transfer mapping from 3x3 source sites to 20x20 target regions", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    return fig


def main():
    h5_path = target_h5 or find_target_h5(default_target_data_dir, exposure)

    if h5_path is not None:
        sites, mean_image = load_sites_and_mean_image(h5_path, max_images)
        if no_image:
            mean_image = None
        print(f"Using target h5: {h5_path}")
    else:
        sites = synthetic_20x20_sites()
        mean_image = None
        print("No target h5 found; using a synthetic 20x20 grid.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    plot_transfer_region_map(sites, mean_image, output_path)
    print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
