import csv
import pickle
import re
from pathlib import Path

import h5py
import joblib
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnn_site import make_random_split
from lightweight_cnn_site import calc_lightweight_cnn_site_scores
from lightweight_cnn_site_spacing_conditioned import (
    calc_lightweight_spacing_cnn_site_scores_from_loader,
    load_lightweight_spacing_cnn_site_model,
)
from mf_site import classification_fidelity
from utils import H5Dataset


spacing_data_dir = Path("spacing_datasets_3x3")
data_dir = Path("dataset_3x3")
model_dir = Path("models_3x3")
output_dir = Path("output_3x3")

output_path = output_dir / "lightweight_spacing_generalization_vs_exposure.png"
csv_output_path = output_dir / "lightweight_spacing_generalization_vs_exposure.csv"

fixed_spacing_px = 7.0
spacing_center_px = 16.0
spacing_scale_px = 12.0
base_seed = 42
batch_size = 2048
num_workers = 0
device = "cpu"

model_re = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5$")

methods = (
    {
        "method": "lightweight_cnn_site",
        "label": "Tiny CNN-site",
        "suffix": "lightweight_cnn_site",
        "kind": "standard",
    },
    {
        "method": "lightweight_cnn_site_pooled",
        "label": "Tiny CNN-site pooled",
        "suffix": "lightweight_cnn_site_pooled",
        "kind": "spacing_model",
    },
    {
        "method": "lightweight_cnn_site_spacing_conditioned",
        "label": "Tiny CNN-site spacing-conditioned",
        "suffix": "lightweight_cnn_site_spacing_conditioned",
        "kind": "spacing_model",
    },
)


def parse_exposure_from_h5(path):
    match = model_re.fullmatch(path.name)
    if match is None:
        return None
    exposure_s = float(match.group(1))
    return exposure_s, exposure_s * 1000.0


def resolve_h5_path(data_dir, exposure_s):
    expected = data_dir / f"qubit_dataset_sim_{exposure_s:g}.h5"
    if expected.exists():
        return expected

    for path in sorted(data_dir.glob("qubit_dataset_sim_*.h5")):
        parsed = parse_exposure_from_h5(path)
        if parsed is not None and np.isclose(parsed[0], exposure_s):
            return path

    raise FileNotFoundError(f"could not find dataset_3x3 HDF5 for exposure {exposure_s:g}")


def resolve_model_path(model_dir, h5_path, suffix):
    path = model_dir / f"{h5_path.stem}_{suffix}.joblib"
    if path.exists():
        return path
    return None


def get_test_indices(h5_path, model_object=None, seed=base_seed):
    if model_object is not None and "test_idx" in model_object:
        return np.asarray(model_object["test_idx"], dtype=np.int64)

    with h5py.File(h5_path, "r") as f:
        n_samples = f["images"].shape[0]

    _train_idx, _val_idx, test_idx = make_random_split(
        n_samples=n_samples,
        seed=seed,
    )
    return np.asarray(test_idx, dtype=np.int64)


def model_thresholds(model_object):
    if "threshold" in model_object:
        return np.asarray(model_object["threshold"], dtype=np.float32)
    if "thresholds" in model_object:
        return np.asarray(model_object["thresholds"], dtype=np.float32)
    raise KeyError("model has no threshold or thresholds")


def evaluate_scores(scores, labels, thresholds):
    fids = np.zeros(labels.shape[1], dtype=np.float32)
    for site_idx in range(labels.shape[1]):
        fids[site_idx] = classification_fidelity(
            scores[:, site_idx],
            labels[:, site_idx],
            thresholds[site_idx],
        )

    preds = scores >= thresholds[None, :]
    errors = preds != (labels > 0.5)

    return {
        "test_fidelity": fids,
        "test_mean_fidelity": float(np.mean(fids)),
        "test_raw_accuracy": float(1.0 - errors.mean()),
        "test_errors_per_site": errors.sum(axis=0).astype(np.int64),
        "test_total_errors": int(errors.sum()),
    }


class FixedSpacingGlobalSitesDataset(Dataset):
    """
    Adapter for old dataset_3x3 files:
      images: (N,H,W)
      labels: (N,9)
      sites:  (9,2)

    It emits the spacing-model loader format with a constant known spacing.
    """
    def __init__(
        self,
        h5_path,
        indices,
        crop_size,
        fixed_spacing_px=fixed_spacing_px,
        spacing_center_px=spacing_center_px,
        spacing_scale_px=spacing_scale_px,
        image_key="images",
        label_key="labels",
        site_key="sites",
        mean_image=None,
        scale=None,
    ):
        self.h5_path = h5_path
        self.indices = list(indices)
        self.crop_size = int(crop_size)
        self.fixed_spacing_px = np.float32(fixed_spacing_px)
        self.spacing_center_px = float(spacing_center_px)
        self.spacing_scale_px = float(spacing_scale_px)
        self.image_key = image_key
        self.label_key = label_key
        self.site_key = site_key
        self.mean_image = None if mean_image is None else np.asarray(mean_image, dtype=np.float32)
        self.scale = None if scale is None else float(scale)
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            self.n_images = int(f[image_key].shape[0])
            self.sites = f[site_key][:].astype(np.float32)
            self.n_sites = int(self.sites.shape[0])

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _extract_site_crop(self, image, site):
        height, width = image.shape
        half = self.crop_size // 2
        pad = half + 1

        padded = np.pad(
            image,
            pad_width=((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0.0,
        )

        center_x = int(np.rint(site[0])) + pad
        center_y = int(np.rint(site[1])) + pad

        x0 = center_x - half
        y0 = center_y - half
        crop = padded[y0:y0 + self.crop_size, x0:x0 + self.crop_size]

        if crop.shape != (self.crop_size, self.crop_size):
            raise ValueError(
                f"expected crop shape {(self.crop_size, self.crop_size)}, "
                f"got {crop.shape} from image shape {(height, width)}"
            )

        return crop.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        file_index = int(self.indices[idx])
        f = self._file()

        image = f[self.image_key][file_index].astype(np.float32)
        if self.mean_image is not None:
            image = image - self.mean_image
        if self.scale is not None:
            image = image / self.scale

        crops = np.stack(
            [
                self._extract_site_crop(
                    image=image,
                    site=site,
                )
                for site in self.sites
            ],
            axis=0,
        )
        labels = (f[self.label_key][file_index] > 0.5).astype(np.int64)
        spacing_norm = np.float32(
            (self.fixed_spacing_px - self.spacing_center_px) / self.spacing_scale_px
        )

        return (
            torch.from_numpy(crops[:, None, :, :]).float(),
            torch.from_numpy(labels).long(),
            torch.full((self.n_sites,), spacing_norm, dtype=torch.float32),
            torch.full((self.n_sites,), self.fixed_spacing_px, dtype=torch.float32),
        )


def make_fixed_spacing_loader(
    h5_path,
    test_idx,
    model_object,
    batch_size,
    num_workers,
):
    spacing_normalization = model_object.get("spacing_normalization", {})
    spacing_center_px = spacing_normalization.get("center_px", spacing_center_px)
    spacing_scale_px = spacing_normalization.get("scale_px", spacing_scale_px)

    dataset = FixedSpacingGlobalSitesDataset(
        h5_path=h5_path,
        indices=test_idx,
        crop_size=int(model_object.get("crop_size", 20)),
        fixed_spacing_px=fixed_spacing_px,
        spacing_center_px=spacing_center_px,
        spacing_scale_px=spacing_scale_px,
        mean_image=model_object.get("mean_image"),
        scale=model_object.get("scale"),
    )
    image_batch_size = max(1, batch_size // dataset.n_sites)

    return DataLoader(
        dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def evaluate_standard_lightweight(h5_path, model_object, test_idx, device):
    loader = DataLoader(
        H5Dataset(
            h5_path,
            indices=test_idx,
            mean_image=model_object.get("mean_image"),
            scale=model_object.get("scale"),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    scores, labels = calc_lightweight_cnn_site_scores(
        loader,
        model_object,
        device=device,
    )
    return evaluate_scores(
        scores=scores,
        labels=labels,
        thresholds=model_thresholds(model_object),
    )


def evaluate_spacing_model(h5_path, model_object, test_idx, device):
    loader = make_fixed_spacing_loader(
        h5_path=h5_path,
        test_idx=test_idx,
        model_object=model_object,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = load_lightweight_spacing_cnn_site_model(model_object, device=device)
    scores, labels, _spacing_px = calc_lightweight_spacing_cnn_site_scores_from_loader(
        dataloader=loader,
        model=model,
        device=device,
    )
    return evaluate_scores(
        scores=scores,
        labels=labels,
        thresholds=model_thresholds(model_object),
    )


def collect_records(
    spacing_data_dir=spacing_data_dir,
    data_dir=data_dir,
    model_dir=model_dir,
    device=device,
):
    records = []

    for spacing_h5_path in sorted(spacing_data_dir.glob("qubit_dataset_sim_*.h5")):
        parsed = parse_exposure_from_h5(spacing_h5_path)
        if parsed is None:
            continue

        exposure_s, exposure_ms = parsed
        try:
            h5_path = resolve_h5_path(data_dir, exposure_s)
        except FileNotFoundError as exc:
            print(f"Skipping exposure {exposure_s:g}: {exc}")
            continue

        standard_model = None
        standard_model_path = resolve_model_path(
            model_dir,
            h5_path,
            "lightweight_cnn_site",
        )
        if standard_model_path is not None:
            try:
                standard_model = joblib.load(standard_model_path)
            except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
                print(f"Skipping {standard_model_path.name}: could not load ({exc})")

        test_idx = get_test_indices(
            h5_path=h5_path,
            model_object=standard_model,
            seed=base_seed,
        )

        for method_spec in methods:
            model_path = resolve_model_path(
                model_dir,
                h5_path,
                method_spec["suffix"],
            )
            if model_path is None:
                print(
                    f"Missing {method_spec['label']} model for "
                    f"{exposure_ms:g} ms: expected suffix {method_spec['suffix']}"
                )
                continue

            try:
                model_object = (
                    standard_model
                    if model_path == standard_model_path and standard_model is not None
                    else joblib.load(model_path)
                )
            except (EOFError, OSError, ValueError, pickle.UnpicklingError) as exc:
                print(f"Skipping {model_path.name}: could not load ({exc})")
                continue

            if method_spec["kind"] == "standard":
                metrics = evaluate_standard_lightweight(
                    h5_path=h5_path,
                    model_object=model_object,
                    test_idx=test_idx,
                    device=device,
                )
            elif method_spec["kind"] == "spacing_model":
                metrics = evaluate_spacing_model(
                    h5_path=h5_path,
                    model_object=model_object,
                    test_idx=test_idx,
                    device=device,
                )
            else:
                raise ValueError(f"unsupported method kind {method_spec['kind']}")

            records.append(
                {
                    "method": method_spec["method"],
                    "label": method_spec["label"],
                    "exposure_s": exposure_s,
                    "exposure_ms": exposure_ms,
                    "mean_fidelity": metrics["test_mean_fidelity"],
                    "mean_infidelity": float(1.0 - metrics["test_mean_fidelity"]),
                    "raw_accuracy": metrics["test_raw_accuracy"],
                    "total_errors": metrics["test_total_errors"],
                    "n_test_images": int(len(test_idx)),
                    "n_sites": int(metrics["test_fidelity"].size),
                    "fixed_spacing_px": float(fixed_spacing_px),
                    "model_path": str(model_path),
                    "data_path": str(h5_path),
                }
            )

            print(
                f"{method_spec['label']} | {exposure_ms:g} ms | "
                f"F={metrics['test_mean_fidelity']:.8f} | "
                f"errors={metrics['test_total_errors']}"
            )

    return records


def save_records_csv(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "label",
        "exposure_s",
        "exposure_ms",
        "mean_fidelity",
        "mean_infidelity",
        "raw_accuracy",
        "total_errors",
        "n_test_images",
        "n_sites",
        "fixed_spacing_px",
        "model_path",
        "data_path",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_records(records, output_path):
    if not records:
        raise ValueError("no records to plot")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for method_spec in methods:
        group = sorted(
            [
                record for record in records
                if record["method"] == method_spec["method"]
            ],
            key=lambda record: record["exposure_ms"],
        )
        if not group:
            continue

        ax.plot(
            [record["exposure_ms"] for record in group],
            [max(record["mean_infidelity"], 1e-12) for record in group],
            marker="o",
            linewidth=2,
            label=method_spec["label"],
        )

    ax.set_xlabel("Exposure time (ms)")
    ax.set_ylabel("Infidelity, 1 - F")
    ax.set_title("Infidelity vs Exposure Time")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main(
    spacing_data_dir=spacing_data_dir,
    data_dir=data_dir,
    model_dir=model_dir,
    output_path=output_path,
    csv_output_path=csv_output_path,
    device=device,
):
    records = collect_records(
        spacing_data_dir=spacing_data_dir,
        data_dir=data_dir,
        model_dir=model_dir,
        device=device,
    )
    save_records_csv(records, csv_output_path)
    plot_records(records, output_path)
    print(f"Saved CSV: {csv_output_path}")
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main(
        spacing_data_dir=spacing_data_dir,
        data_dir=data_dir,
        model_dir=model_dir,
        output_path=output_path,
        csv_output_path=csv_output_path,
        device=device,
    )
