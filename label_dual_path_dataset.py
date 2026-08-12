from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm


INPUT_PATH = Path("/Users/marone/Downloads/runs_4043_fast_imaging.h5")
OUTPUT_PATH = Path("dataset_labeled/runs_4043_fast_imaging_labeled.h5")
SITE_DIAGNOSTIC_PATH = Path("dataset_labeled/runs_4043_fast_imaging_sites.png")

INPUT_IMAGE_KEY = "image_2"
LABEL_IMAGE_KEY = "image_1"

AVERAGE_BATCH_SIZE = 64
SCORE_BATCH_SIZE = 32
FIT_SIZE = 17
PEAK_MIN_DISTANCE = 18
PEAK_SMOOTH_SIGMA = 2.0
SITE_PEAK_GMM_COMPONENTS = 2
SITE_PEAK_COMPONENT = "bright"
ROW_CLUSTER_GAP_FACTOR = 0.55
THRESHOLD_MODE = "per_site"


def circular_gaussian(coords, amp, x0, y0, sigma, offset):
    y, x = coords
    return offset + amp * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    )


def compute_average_image(h5_path, image_key, batch_size):
    with h5py.File(h5_path, "r") as f:
        images = f[image_key]
        total = np.zeros(images.shape[1:], dtype=np.float64)
        count = images.shape[0]

        for start in tqdm(range(0, count, batch_size), desc="Average image"):
            stop = min(start + batch_size, count)
            total += images[start:stop].astype(np.float64).sum(axis=0)

    return (total / count).astype(np.float32)


def find_peak_candidates(avg_image, min_distance, smooth_sigma):
    smoothed = gaussian_filter(avg_image, sigma=smooth_sigma)
    footprint = np.ones((min_distance, min_distance), dtype=bool)
    local_max = smoothed == maximum_filter(smoothed, footprint=footprint)

    margin = min_distance // 2
    local_max[:margin, :] = False
    local_max[-margin:, :] = False
    local_max[:, :margin] = False
    local_max[:, -margin:] = False

    ys, xs = np.nonzero(local_max)
    if ys.size == 0:
        raise ValueError("found no peak candidates")

    values = smoothed[ys, xs]
    return np.column_stack([xs, ys]).astype(np.float32), values.astype(np.float32)


def choose_site_peaks(candidates, values):
    gmm = GaussianMixture(
        n_components=SITE_PEAK_GMM_COMPONENTS,
        covariance_type="full",
        random_state=0,
        n_init=10,
    )
    gmm.fit(values.reshape(-1, 1))

    means = gmm.means_.reshape(-1)
    bright_component = int(np.argmax(means))
    labels = gmm.predict(values.reshape(-1, 1))

    if SITE_PEAK_COMPONENT != "bright":
        raise ValueError(f"unsupported SITE_PEAK_COMPONENT: {SITE_PEAK_COMPONENT}")

    selected = candidates[labels == bright_component]
    selected_values = values[labels == bright_component]
    if selected.size == 0:
        raise ValueError("site peak GMM selected no peaks")

    order = np.lexsort((selected[:, 0], selected[:, 1]))
    return selected[order], {
        "n_candidates": int(candidates.shape[0]),
        "n_selected": int(selected.shape[0]),
        "component_means": means.astype(np.float32),
        "component_weights": gmm.weights_.astype(np.float32),
        "selected_peak_min": float(selected_values.min()),
        "selected_peak_max": float(selected_values.max()),
    }


def fit_site_gaussians(avg_image, initial_sites, fit_size):
    height, width = avg_image.shape
    radius = fit_size // 2
    fitted = []

    for x_guess, y_guess in initial_sites:
        xc = int(round(float(x_guess)))
        yc = int(round(float(y_guess)))

        x1 = max(0, xc - radius)
        x2 = min(width, xc + radius + 1)
        y1 = max(0, yc - radius)
        y2 = min(height, yc + radius + 1)

        patch = avg_image[y1:y2, x1:x2]
        yy, xx = np.mgrid[y1:y2, x1:x2]

        amp_guess = float(patch.max() - patch.min())
        offset_guess = float(patch.min())
        sigma_guess = max(1.0, fit_size / 6.0)

        bounds = (
            [0.0, x1, y1, 0.2, -np.inf],
            [np.inf, x2, y2, fit_size, np.inf],
        )

        try:
            popt, _ = curve_fit(
                circular_gaussian,
                (yy.ravel(), xx.ravel()),
                patch.ravel(),
                p0=[amp_guess, x_guess, y_guess, sigma_guess, offset_guess],
                bounds=bounds,
                maxfev=20000,
            )
            amp, x0, y0, sigma, offset = popt
        except RuntimeError:
            amp, x0, y0, sigma, offset = (
                amp_guess,
                x_guess,
                y_guess,
                sigma_guess,
                offset_guess,
            )

        fitted.append((x0, y0, sigma, amp, offset))

    fitted = np.asarray(fitted, dtype=np.float32)
    return sort_sites_row_major(fitted)


def sort_sites_row_major(fitted_sites):
    ordered = fitted_sites[np.argsort(fitted_sites[:, 1])]
    if ordered.shape[0] <= 1:
        return ordered

    y_diffs = np.diff(ordered[:, 1])
    positive_diffs = y_diffs[y_diffs > 1e-6]
    if positive_diffs.size == 0:
        rows = [ordered]
    else:
        row_gap = np.median(positive_diffs) / max(ROW_CLUSTER_GAP_FACTOR, 1e-6)
        split_indices = np.flatnonzero(y_diffs > row_gap) + 1
        rows = np.split(ordered, split_indices)

    sorted_rows = [row[np.argsort(row[:, 0])] for row in rows]
    return np.concatenate(sorted_rows, axis=0)


def gaussian_masks(image_shape, fitted_sites):
    height, width = image_shape
    yy, xx = np.mgrid[0:height, 0:width]
    masks = []

    for x0, y0, sigma, _, _ in fitted_sites:
        sigma = max(float(sigma), 1e-6)
        mask = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
        masks.append(mask.astype(np.float32))

    return np.stack(masks, axis=0)


def score_label_images(h5_path, image_key, masks, batch_size):
    masks_flat = masks.reshape(masks.shape[0], -1).astype(np.float32)

    with h5py.File(h5_path, "r") as f:
        images = f[image_key]
        n_images = images.shape[0]
        scores = np.empty((n_images, masks.shape[0]), dtype=np.float32)

        for start in tqdm(range(0, n_images, batch_size), desc="Gaussian-mask scores"):
            stop = min(start + batch_size, n_images)
            batch = images[start:stop].astype(np.float32).reshape(stop - start, -1)
            scores[start:stop] = batch @ masks_flat.T

    return scores


def fit_two_gaussian_threshold(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
        n_init=10,
    )
    gmm.fit(values)

    means = gmm.means_.reshape(-1)
    sigmas = np.sqrt(gmm.covariances_.reshape(-1))
    weights = gmm.weights_.reshape(-1)

    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]

    m0, m1 = means
    s0, s1 = np.maximum(sigmas, 1e-12)
    w0, w1 = np.maximum(weights, 1e-12)

    a = 1.0 / (2.0 * s1 ** 2) - 1.0 / (2.0 * s0 ** 2)
    b = m0 / (s0 ** 2) - m1 / (s1 ** 2)
    c = (
        m1 ** 2 / (2.0 * s1 ** 2)
        - m0 ** 2 / (2.0 * s0 ** 2)
        + np.log((w1 * s0) / (w0 * s1))
    )

    if abs(a) < 1e-12:
        threshold = -c / b
    else:
        roots = np.roots([a, b, c])
        real_roots = roots[np.isreal(roots)].real
        between = real_roots[(real_roots > m0) & (real_roots < m1)]
        threshold = float(between[0]) if between.size else float(np.mean(means))

    return float(threshold), {
        "means": means.astype(np.float32),
        "sigmas": sigmas.astype(np.float32),
        "weights": weights.astype(np.float32),
    }


def choose_thresholds(scores, mode):
    if mode == "global":
        threshold, params = fit_two_gaussian_threshold(scores.reshape(-1))
        thresholds = np.full(scores.shape[1], threshold, dtype=np.float32)
        return thresholds, {"mode": mode, "global": params}

    if mode == "per_site":
        thresholds = np.empty(scores.shape[1], dtype=np.float32)
        params = []
        for site_idx in range(scores.shape[1]):
            thresholds[site_idx], site_params = fit_two_gaussian_threshold(scores[:, site_idx])
            params.append(site_params)
        return thresholds, {"mode": mode, "per_site": params}

    raise ValueError(f"unknown threshold mode: {mode}")


def write_labeled_dataset(
    input_path,
    output_path,
    labels,
    fitted_sites,
    avg_image,
    scores,
    thresholds,
    site_detection,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")

    with h5py.File(input_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        src.copy(INPUT_IMAGE_KEY, dst, name="images")
        dst.create_dataset("labels", data=labels.astype(np.uint8), compression="gzip")
        dst.create_dataset("sites", data=fitted_sites[:, :2].astype(np.float32))
        dst.create_dataset("label_scores", data=scores.astype(np.float32), compression="gzip")
        dst.create_dataset("label_thresholds", data=thresholds.astype(np.float32))
        dst.create_dataset("average_label_image", data=avg_image.astype(np.float32), compression="gzip")
        dst.create_dataset(
            "site_fit_parameters",
            data=fitted_sites.astype(np.float32),
        )

        dst.attrs["source_file"] = str(input_path)
        dst.attrs["input_image_key"] = INPUT_IMAGE_KEY
        dst.attrs["label_image_key"] = LABEL_IMAGE_KEY
        dst.attrs["label_method"] = "average_image_site_fit_gaussian_mask_two_gaussian_threshold"
        dst.attrs["threshold_mode"] = THRESHOLD_MODE
        dst.attrs["n_sites"] = int(fitted_sites.shape[0])
        dst.attrs["site_detection_n_candidates"] = site_detection["n_candidates"]
        dst.attrs["site_detection_n_selected"] = site_detection["n_selected"]

    tmp_path.replace(output_path)


def save_site_diagnostic(avg_image, fitted_sites, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(avg_image, cmap="jet")
    ax.scatter(
        fitted_sites[:, 0],
        fitted_sites[:, 1],
        s=100,
        facecolors="none",
        edgecolors="white",
        linewidths=0.8,
    )
    for site_idx, (x, y, *_rest) in enumerate(fitted_sites):
        ax.text(
            x + 2,
            y + 2,
            str(site_idx),
            color="white",
            fontsize=5,
        )

    ax.set_title(f"Detected sites: {fitted_sites.shape[0]}")
    ax.set_xlim(0, avg_image.shape[1])
    ax.set_ylim(avg_image.shape[0], 0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    avg_image = compute_average_image(
        h5_path=INPUT_PATH,
        image_key=LABEL_IMAGE_KEY,
        batch_size=AVERAGE_BATCH_SIZE,
    )
    candidates, peak_values = find_peak_candidates(
        avg_image=avg_image,
        min_distance=PEAK_MIN_DISTANCE,
        smooth_sigma=PEAK_SMOOTH_SIGMA,
    )
    initial_sites, site_detection = choose_site_peaks(
        candidates=candidates,
        values=peak_values,
    )
    fitted_sites = fit_site_gaussians(
        avg_image=avg_image,
        initial_sites=initial_sites,
        fit_size=FIT_SIZE,
    )
    save_site_diagnostic(
        avg_image=avg_image,
        fitted_sites=fitted_sites,
        output_path=SITE_DIAGNOSTIC_PATH,
    )
    masks = gaussian_masks(
        image_shape=avg_image.shape,
        fitted_sites=fitted_sites,
    )
    scores = score_label_images(
        h5_path=INPUT_PATH,
        image_key=LABEL_IMAGE_KEY,
        masks=masks,
        batch_size=SCORE_BATCH_SIZE,
    )
    thresholds, threshold_fit = choose_thresholds(scores, THRESHOLD_MODE)
    labels = scores >= thresholds[None, :]

    write_labeled_dataset(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        labels=labels,
        fitted_sites=fitted_sites,
        avg_image=avg_image,
        scores=scores,
        thresholds=thresholds,
        site_detection=site_detection,
    )

    print("Saved:", OUTPUT_PATH)
    print("Saved site diagnostic:", SITE_DIAGNOSTIC_PATH)
    print(f"Detected {fitted_sites.shape[0]} sites from {site_detection['n_candidates']} peak candidates")
    print("Site detection:")
    print(site_detection)
    print("Sites, row-major x y sigma:")
    print(fitted_sites[:, :3])
    print("Occupancy per site:")
    print(labels.mean(axis=0))
    print("Thresholds:")
    print(thresholds)
    print("Threshold fit:")
    print(threshold_fit)


if __name__ == "__main__":
    main()
