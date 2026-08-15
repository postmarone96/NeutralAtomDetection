from pathlib import Path

import h5py
import joblib
import numpy as np
from torch.utils.data import DataLoader

from baselines import (
    calc_gaussian_scores_best,
    calc_square_scores_best,
    compute_average_image,
    fit_sigmas_from_average_image,
    gaussian_with_s_sweep,
    square_with_s_sweep,
)
from cnn_site import (
    base_seed,
    # CROP_SIZE,
    epochs,
    # LEARNING_RATE,
    build_site_crop_cache,
    make_random_split,
    # NUM_WORKERS,
    run_one_split as run_cnn_site_split,
)
from lightweight_cnn_site import (
    # CROP_SIZE as LIGHTWEIGHT_CNN_CROP_SIZE,
    early_stopping_min_delta as lightweight_cnn_early_stopping_min_delta,
    early_stopping_patience as lightweight_cnn_early_stopping_patience,
    epochs as lightweight_cnn_epochs,
    # LEARNING_RATE as LIGHTWEIGHT_CNN_LEARNING_RATE,
    lightweight_cnn_site_architecture,
    lightweight_cnn_site_method,
    run_one_split as run_lightweight_cnn_site_split,
)
from mf_array import run_one_split as run_mf_array_split
from mf_site import classification_fidelity
from mf_site import run_one_split as run_mf_site_split
from utils import H5Dataset, get_device

crop_size = lightweight_cnn_crop_size = 20
num_workers = 20

data_dir = Path("dataset_labeled")
model_dir = Path("models_3x3")
output_dir = Path("output_3x3")
exposure_s = 0.030

training_image_key = "image_2"
label_image_key = "image_1"

mf_s_values = range(2, 15, 2)
mf_alpha = 1e-4
mf_batch_size = 64

learning_rate = 2e-4
cnn_batch_size = 16052
cnn_eval_batch_size = 32768

lightweight_cnn_learning_rate = 2e-4
lightweight_cnn_batch_size = 32104
lightweight_cnn_eval_batch_size = 65536

baseline_batch_size = 32
site_crop_cache_batch_size = 16
use_site_crop_cache = True

mf_array_batch_size = 64
mf_array_train_normal = True
mf_array_train_gaussian = False
mf_array_gaussian_normalize = True
mf_array_neighbor_context = "coordinate_8_nearest"

train_baselines = False
train_mf_site = False
train_mf_array = False
train_cnn_site = False
train_lightweight_cnn_site = True
train_cnn_array = False


def atomic_joblib_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp_path)
    tmp_path.replace(path)


def find_dataset(data_dir):
    h5_paths = sorted(data_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no .h5 datasets found in {data_dir}")
    if len(h5_paths) > 1:
        raise ValueError(f"expected one 30 ms dataset in {data_dir}, found {len(h5_paths)}")
    return h5_paths[0]


def check_dataset_metadata(h5_path):
    with h5py.File(h5_path, "r") as f:
        n_images = f["images"].shape[0]
        n_sites = f["sites"].shape[0]
        input_image_key = str(f.attrs.get("input_image_key", ""))
        label_image_key = str(f.attrs.get("label_image_key", ""))
        threshold_mode = str(f.attrs.get("threshold_mode", ""))

    if input_image_key != training_image_key:
        raise ValueError(
            f"expected training images from {training_image_key}, "
            f"but dataset says {input_image_key}"
        )
    if label_image_key != label_image_key:
        raise ValueError(
            f"expected labels/sites from {label_image_key}, "
            f"but dataset says {label_image_key}"
        )

    return {
        "n_images": int(n_images),
        "n_sites": int(n_sites),
        "input_image_key": input_image_key,
        "label_image_key": label_image_key,
        "threshold_mode": threshold_mode,
    }


def metadata(h5_path, info):
    return {
        "h5_path": str(h5_path),
        "exposure_s": float(exposure_s),
        "exposure_ms": float(1000.0 * exposure_s),
        "training_image_key": info["input_image_key"],
        "label_image_key": info["label_image_key"],
        "label_source": "pseudo_labels_and_sites_from_unattenuated_image_1",
        "threshold_mode": info["threshold_mode"],
        "n_images": info["n_images"],
        "n_sites": info["n_sites"],
    }


def save_baseline_model(
    h5_path,
    info,
    method_name,
    trained,
    train_idx,
    val_idx,
    test_idx,
    test_scores,
    test_preds,
    test_labels,
    extra_fields=None,
):
    test_fids = np.array(
        [
            classification_fidelity(
                test_scores[:, q],
                test_labels[:, q],
                trained["threshold"][q],
            )
            for q in range(test_labels.shape[1])
        ],
        dtype=np.float32,
    )
    errors = test_preds != (test_labels > 0.5)

    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": method_name,
        "seed": base_seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "s_values": list(mf_s_values),
        "test_fidelity": test_fids,
        "test_mean_fidelity": float(np.mean(test_fids)),
        "test_errors_per_site": errors.sum(axis=0).astype(np.int64),
        "test_total_errors": int(errors.sum()),
        "test_raw_accuracy": float(1.0 - errors.mean()),
    }
    if extra_fields:
        save_obj.update(extra_fields)

    output_path = model_dir / f"qubit_dataset_sim_{exposure_s:g}_{method_name}.joblib"
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={save_obj['test_mean_fidelity']:.8f} | "
        f"raw acc={save_obj['test_raw_accuracy']:.8f}"
    )


def train_baselines(h5_path, info, device):
    train_idx, val_idx, test_idx = make_random_split(
        n_samples=info["n_images"],
        seed=base_seed,
    )

    avg_image, sites = compute_average_image(
        h5_path=h5_path,
        indices=train_idx,
        batch_size=baseline_batch_size,
        num_workers=num_workers,
    )
    fitted_sigmas = fit_sigmas_from_average_image(
        avg_image=avg_image,
        sites=sites,
        fit_size=9,
    )

    square_trained = square_with_s_sweep(
        h5_path=h5_path,
        val_idx=val_idx,
        s_values=mf_s_values,
        batch_size=baseline_batch_size,
        num_workers=num_workers,
        device=device,
    )
    gaussian_trained = gaussian_with_s_sweep(
        h5_path=h5_path,
        val_idx=val_idx,
        sigmas=fitted_sigmas,
        s_values=mf_s_values,
        batch_size=baseline_batch_size,
        num_workers=num_workers,
        device=device,
        normalize=False,
    )

    test_loader = DataLoader(
        H5Dataset(h5_path, indices=test_idx),
        batch_size=baseline_batch_size,
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

    save_baseline_model(
        h5_path=h5_path,
        info=info,
        method_name="square_real",
        trained=square_trained,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_scores=square_scores,
        test_preds=square_preds,
        test_labels=square_labels,
    )
    save_baseline_model(
        h5_path=h5_path,
        info=info,
        method_name="gaussian_real",
        trained=gaussian_trained,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_scores=gaussian_scores,
        test_preds=gaussian_preds,
        test_labels=gaussian_labels,
        extra_fields={"fitted_sigmas": fitted_sigmas},
    )


def train_mf_site(h5_path, info, device):
    result = run_mf_site_split(
        h5_path=h5_path,
        split_seed=base_seed,
        s_values=mf_s_values,
        alpha=mf_alpha,
        batch_size=mf_batch_size,
        num_workers=num_workers,
        device=device,
    )

    trained = result["trained_model"]
    output_path = model_dir / f"qubit_dataset_sim_{exposure_s:g}_mf_site_real.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": "mf_site",
        "seed": base_seed,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "s_values": list(mf_s_values),
        "alpha": mf_alpha,
        "batch_size": mf_batch_size,
        "thresholds": result["thresholds"],
        "test_fidelity": result["test_fidelity"],
        "test_mean_fidelity": result["test_mean_fidelity"],
        "test_raw_accuracy": result["test_raw_accuracy"],
        "test_errors_per_site": result["test_errors_per_site"],
        "test_total_errors": result["test_total_errors"],
        "preprocessing": "subtract_train_mean_image_divide_train_max_abs",
    }
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={result['test_mean_fidelity']:.8f} | "
        f"raw acc={result['test_raw_accuracy']:.8f}"
    )


def train_mf_array_variant(h5_path, info, device, gaussian):
    result = run_mf_array_split(
        h5_path=h5_path,
        split_seed=base_seed,
        s_values=mf_s_values,
        alpha=mf_alpha,
        batch_size=mf_array_batch_size,
        num_workers=num_workers,
        device=device,
        gaussian=gaussian,
        gaussian_normalize=mf_array_gaussian_normalize,
        neighbor_context=mf_array_neighbor_context,
    )

    method_name = "mf_array_gaussian_real" if gaussian else "mf_array_real"
    trained = result["trained_model"]
    output_path = model_dir / f"qubit_dataset_sim_{exposure_s:g}_{method_name}.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": method_name,
        "seed": base_seed,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "s_values": list(mf_s_values),
        "alpha": mf_alpha,
        "batch_size": mf_array_batch_size,
        "thresholds": result["thresholds"],
        "test_fidelity": result["test_fidelity"],
        "test_mean_fidelity": result["test_mean_fidelity"],
        "test_raw_accuracy": result["test_raw_accuracy"],
        "test_errors_per_site": result["test_errors_per_site"],
        "test_total_errors": result["test_total_errors"],
        "mf_array_gaussian": bool(gaussian),
        "mf_array_context": "gaussian" if gaussian else "mean",
        "mf_array_sigmas": result["mf_array_sigmas"],
        "mf_array_gaussian_normalize": bool(mf_array_gaussian_normalize),
        "mf_array_neighbor_indices": result["mf_array_neighbor_indices"],
        "mf_array_neighbor_context": result["mf_array_neighbor_context"],
        "preprocessing": "subtract_train_mean_image_divide_train_max_abs",
    }
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={result['test_mean_fidelity']:.8f} | "
        f"raw acc={result['test_raw_accuracy']:.8f}"
    )


def train_mf_array(h5_path, info, device):
    if mf_array_train_normal:
        train_mf_array_variant(h5_path, info, device, gaussian=False)
    if mf_array_train_gaussian:
        train_mf_array_variant(h5_path, info, device, gaussian=True)


def train_cnn_site(h5_path, info, device, crop_cache_path=None):
    result = run_cnn_site_split(
        h5_path=h5_path,
        split_seed=base_seed,
        crop_size=crop_size,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=cnn_batch_size,
        eval_batch_size=cnn_eval_batch_size,
        num_workers=num_workers,
        device=device,
        crop_cache_path=crop_cache_path,
    )

    trained = result["trained_model"]
    output_path = model_dir / f"qubit_dataset_sim_{exposure_s:g}_cnn_site_real.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": "cnn_site",
        "seed": base_seed,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "crop_size": int(crop_size),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "batch_size": int(cnn_batch_size),
        "eval_batch_size": int(cnn_eval_batch_size),
        "crop_cache_path": None if crop_cache_path is None else str(crop_cache_path),
        "thresholds": result["thresholds"],
        "test_fidelity": result["test_fidelity"],
        "test_mean_fidelity": result["test_mean_fidelity"],
        "test_raw_accuracy": result["test_raw_accuracy"],
        "test_errors_per_site": result["test_errors_per_site"],
        "test_total_errors": result["test_total_errors"],
        "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
        "architecture": "cnn_site_3conv_128fc",
    }
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={result['test_mean_fidelity']:.8f} | "
        f"best epoch={result['best_epoch']}"
    )


def train_lightweight_cnn_site(h5_path, info, device, crop_cache_path=None):
    result = run_lightweight_cnn_site_split(
        h5_path=h5_path,
        split_seed=base_seed,
        crop_size=lightweight_cnn_crop_size,
        learning_rate=lightweight_cnn_learning_rate,
        epochs=lightweight_cnn_epochs,
        batch_size=lightweight_cnn_batch_size,
        eval_batch_size=lightweight_cnn_eval_batch_size,
        num_workers=num_workers,
        device=device,
        early_stopping_patience=lightweight_cnn_early_stopping_patience,
        early_stopping_min_delta=lightweight_cnn_early_stopping_min_delta,
        crop_cache_path=crop_cache_path,
    )

    trained = result["trained_model"]
    output_path = model_dir / f"qubit_dataset_sim_{exposure_s:g}_{lightweight_cnn_site_method}_real.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": lightweight_cnn_site_method,
        "seed": base_seed,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "crop_size": int(lightweight_cnn_crop_size),
        "learning_rate": float(lightweight_cnn_learning_rate),
        "epochs": int(lightweight_cnn_epochs),
        "epochs_trained": result["epochs_trained"],
        "batch_size": int(lightweight_cnn_batch_size),
        "eval_batch_size": int(lightweight_cnn_eval_batch_size),
        "crop_cache_path": None if crop_cache_path is None else str(crop_cache_path),
        "early_stopping_patience": (
            None
            if lightweight_cnn_early_stopping_patience is None
            else int(lightweight_cnn_early_stopping_patience)
        ),
        "early_stopping_min_delta": float(lightweight_cnn_early_stopping_min_delta),
        "stopped_epoch": result["stopped_epoch"],
        "stopped_early": result["stopped_early"],
        "thresholds": result["thresholds"],
        "test_fidelity": result["test_fidelity"],
        "test_mean_fidelity": result["test_mean_fidelity"],
        "test_raw_accuracy": result["test_raw_accuracy"],
        "test_errors_per_site": result["test_errors_per_site"],
        "test_total_errors": result["test_total_errors"],
        "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
        "architecture": lightweight_cnn_site_architecture,
    }
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={result['test_mean_fidelity']:.8f} | "
        f"best epoch={result['best_epoch']} | "
        f"trained epochs={result['epochs_trained']}"
    )


def main():
    h5_path = find_dataset(data_dir)
    info = check_dataset_metadata(h5_path)
    device = 'cuda:1' # get_device()

    print(f"Dataset: {h5_path}")
    print(f"Exposure: {1000.0 * exposure_s:g} ms")
    print(f"Training images: {info['input_image_key']}")
    print(f"Labels/sites: {info['label_image_key']}")
    print(f"Images/sites: {info['n_images']} / {info['n_sites']}")
    print(f"Device: {device}")

    crop_cache_path = None
    if use_site_crop_cache and (train_cnn_site or train_lightweight_cnn_site):
        crop_cache_path = build_site_crop_cache(
            h5_path=h5_path,
            cache_path=output_dir / "site_crop_cache" / f"{h5_path.stem}_crop{crop_size}.h5",
            crop_size=crop_size,
            batch_size=site_crop_cache_batch_size,
        )

    if train_baselines:
        train_baselines(h5_path, info, device)

    if train_mf_array:
        train_mf_array(h5_path, info, device)
    if train_cnn_site:
        train_cnn_site(h5_path, info, device, crop_cache_path=crop_cache_path)
    if train_lightweight_cnn_site:
        train_lightweight_cnn_site(h5_path, info, device, crop_cache_path=crop_cache_path)
    if train_cnn_array:
        raise NotImplementedError(
            "CNN-array is disabled for the real 253-site dataset; "
            "the current cnn_array.py crops only the array center."
        )


if __name__ == "__main__":
    main()
