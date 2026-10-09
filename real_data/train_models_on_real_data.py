from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    build_site_crop_cache,
    run_one_split as run_cnn_site_split,
)
from lightweight_cnn_site import run_one_split as run_lightweight_cnn_site_split
from mf_array import run_one_split as run_mf_array_split
from plots.plot_learning_curves import plot_learning_curves
from mf_site import classification_fidelity
from mf_site import run_one_split as run_mf_site_split
from utils import H5Dataset, make_random_split, path_from_root
import constants

lightweight_cnn_channels = constants.tinycnn_channels
lightweight_cnn_batch_size = constants.tinycnn_batch_size
lightweight_cnn_early_stopping_min_delta = constants.tinycnn_early_stopping_min_delta
lightweight_cnn_early_stopping_patience = constants.tinycnn_early_stopping_patience
lightweight_cnn_epochs = constants.tinycnn_epochs
lightweight_cnn_eval_batch_size = constants.tinycnn_eval_batch_size
lightweight_cnn_learning_rate = constants.tinycnn_learning_rate
lightweight_cnn_site_method = constants.tinycnn_method
lightweight_cnn_site_neighbor_method = constants.tinycnn_neighbor_method
lightweight_cnn_site_neighbor_gaussian_method = constants.tinycnn_neighbor_gaussian_method

base_seed = 43
crop_size = 20
num_workers = 20

real_data_dir = Path(__file__).resolve().parent
output_dir = real_data_dir / "output"
model_dir = output_dir / "models"
learning_curve_dir = output_dir / "learning_curves"
real_datasets_to_train = (
    "fast_imaging",
)
real_dataset_jobs = {
    "fast_imaging": {
        "label": "Fast imaging",
        "h5_path": output_dir / "runs_4043_fast_imaging_labeled.h5",
    },
}

training_image_key = constants.experimental_input_image_key
label_image_key = constants.experimental_label_image_key

mf_s_values = range(2, 15, 2)
mf_alpha = 1e-4
mf_batch_size = 64

learning_rate = 2e-4
cnn_batch_size = 16052
cnn_eval_batch_size = 32768

lightweight_cnn_methods = (
    lightweight_cnn_site_method,
    lightweight_cnn_site_neighbor_method,
    lightweight_cnn_site_neighbor_gaussian_method,
)

baseline_batch_size = 32
site_crop_cache_batch_size = 16
use_site_crop_cache = True

mf_array_batch_size = 64
mf_array_train_normal = True
mf_array_train_gaussian = False
mf_array_gaussian_normalize = True
mf_array_neighbor_context = "coordinate_8_nearest"

do_train_baselines = False
do_train_mf_site = False
do_train_mf_array = False
do_train_cnn_site = False
do_train_lightweight_cnn_site = True
resume = False


def check_dataset_metadata(h5_path):
    with h5py.File(h5_path, "r") as f:
        info = {
            "n_images": f["images"].shape[0],
            "n_sites": f["sites"].shape[0],
            "input_image_key": str(f.attrs.get("input_image_key", "")),
            "label_image_key": str(f.attrs.get("label_image_key", "")),
            "threshold_mode": str(f.attrs.get("threshold_mode", "")),
        }

    if info["input_image_key"] != training_image_key:
        raise ValueError(
            f"expected training images from {training_image_key}, "
            f"but dataset says {info['input_image_key']}"
        )
    if info["label_image_key"] != label_image_key:
        raise ValueError(
            f"expected labels/sites from {label_image_key}, "
            f"but dataset says {info['label_image_key']}"
        )

    return info


def metadata(h5_path, info, dataset_key, dataset_label):
    return {
        "h5_path": path_from_root(h5_path),
        "dataset_key": dataset_key,
        "dataset_label": dataset_label,
        "exposure_s": None,
        "exposure_ms": None,
        "training_image_key": info["input_image_key"],
        "label_image_key": info["label_image_key"],
        "label_source": "pseudo_labels_and_sites_from_unattenuated_image_1",
        "threshold_mode": info["threshold_mode"],
        "n_images": info["n_images"],
        "n_sites": info["n_sites"],
    }


def already_trained(dataset_key, method_name):
    output_path = model_dir / f"{dataset_key}_{method_name}.joblib"
    if resume and output_path.exists():
        print(f"Skipping {output_path.name}: already trained")
        return True
    return False


def save_model(result, method_name, h5_path, info, dataset_key, dataset_label, **extra):
    save_obj = {
        **{key: value for key, value in result.items() if key != "trained_model"},
        **result["trained_model"],
        **metadata(h5_path, info, dataset_key, dataset_label),
        "method": method_name,
        **extra,
    }

    output_path = model_dir / f"{dataset_key}_{method_name}.joblib"
    joblib.dump(save_obj, output_path)

    epoch_info = ""
    if "best_epoch" in save_obj:
        epoch_info = (
            f" | best epoch={save_obj['best_epoch']}"
            f" | trained epochs={save_obj['epochs_trained']}"
        )

    print(
        f"Saved {output_path} | "
        f"mean F={save_obj['test_mean_fidelity']:.8f} | "
        f"raw acc={save_obj['test_raw_accuracy']:.8f}"
        f"{epoch_info}"
    )


def save_baseline_model(
    h5_path,
    info,
    dataset_key,
    dataset_label,
    method_name,
    trained,
    train_idx,
    val_idx,
    test_idx,
    test_scores,
    test_preds,
    test_labels,
    **extra,
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
        **metadata(h5_path, info, dataset_key, dataset_label),
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
        **extra,
    }

    output_path = model_dir / f"{dataset_key}_{method_name}.joblib"
    joblib.dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={save_obj['test_mean_fidelity']:.8f} | "
        f"raw acc={save_obj['test_raw_accuracy']:.8f}"
    )


def train_baselines(h5_path, info, dataset_key, dataset_label, device):
    if already_trained(dataset_key, "square") and already_trained(dataset_key, "gaussian"):
        return

    print("Training baselines...")
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
        fit_size=constants.baselines_fit_size,
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
        persistent_workers=True,
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
        dataset_key=dataset_key,
        dataset_label=dataset_label,
        method_name="square",
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
        dataset_key=dataset_key,
        dataset_label=dataset_label,
        method_name="gaussian",
        trained=gaussian_trained,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_scores=gaussian_scores,
        test_preds=gaussian_preds,
        test_labels=gaussian_labels,
        fitted_sigmas=fitted_sigmas,
    )


def train_mf_site(h5_path, info, dataset_key, dataset_label, device):
    if already_trained(dataset_key, "mf_site"):
        return

    print("Training MF-site...")
    result = run_mf_site_split(
        h5_path=h5_path,
        split_seed=base_seed,
        s_values=mf_s_values,
        alpha=mf_alpha,
        batch_size=mf_batch_size,
        num_workers=num_workers,
        device=device,
    )
    save_model(
        result,
        "mf_site",
        h5_path,
        info,
        dataset_key,
        dataset_label,
        s_values=list(mf_s_values),
        batch_size=mf_batch_size,
        preprocessing="subtract_train_mean_image_divide_train_max_abs",
    )


def train_mf_array_variant(h5_path, info, dataset_key, dataset_label, device, gaussian):
    if already_trained(dataset_key, "mf_array_gaussian" if gaussian else "mf_array"):
        return

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
    save_model(
        result,
        "mf_array_gaussian" if gaussian else "mf_array",
        h5_path,
        info,
        dataset_key,
        dataset_label,
        s_values=list(mf_s_values),
        batch_size=mf_array_batch_size,
        preprocessing="subtract_train_mean_image_divide_train_max_abs",
    )


def train_mf_array(h5_path, info, dataset_key, dataset_label, device):
    print("Training MF-array...")
    if mf_array_train_normal:
        train_mf_array_variant(h5_path, info, dataset_key, dataset_label, device, gaussian=False)
    if mf_array_train_gaussian:
        train_mf_array_variant(h5_path, info, dataset_key, dataset_label, device, gaussian=True)


def train_cnn_site(h5_path, info, dataset_key, dataset_label, device, crop_cache_path=None):
    if already_trained(dataset_key, "cnn_site"):
        return

    print("Training CNN-site...")
    result = run_cnn_site_split(
        h5_path=h5_path,
        split_seed=base_seed,
        crop_size=crop_size,
        learning_rate=learning_rate,
        epochs=lightweight_cnn_epochs,
        batch_size=cnn_batch_size,
        eval_batch_size=cnn_eval_batch_size,
        num_workers=num_workers,
        device=device,
        crop_cache_path=crop_cache_path,
    )
    save_model(
        result,
        "cnn_site",
        h5_path,
        info,
        dataset_key,
        dataset_label,
        eval_batch_size=cnn_eval_batch_size,
        preprocessing="subtract_train_global_mean_divide_train_pixel_range",
    )

    learning_curve_path = learning_curve_dir / f"{dataset_key}_cnn_site_learning_curve.png"
    plot_learning_curves(
        history=result["trained_model"]["history"],
        output_path=learning_curve_path,
        title=f"cnn_site learning curve: {dataset_key}",
        best_epoch=result["trained_model"]["best_epoch"],
    )
    print(f"Saved learning curve: {learning_curve_path}")


def train_lightweight_cnn_site_variant(
    h5_path,
    info,
    dataset_key,
    dataset_label,
    device,
    method_name,
    crop_cache_path=None,
):
    print(f"Training {method_name}...")
    result = run_lightweight_cnn_site_split(
        h5_path=h5_path,
        split_seed=base_seed,
        crop_size=crop_size,
        learning_rate=lightweight_cnn_learning_rate,
        epochs=lightweight_cnn_epochs,
        batch_size=lightweight_cnn_batch_size,
        eval_batch_size=lightweight_cnn_eval_batch_size,
        num_workers=num_workers,
        device=device,
        early_stopping_patience=lightweight_cnn_early_stopping_patience,
        early_stopping_min_delta=lightweight_cnn_early_stopping_min_delta,
        method=method_name,
        crop_cache_path=crop_cache_path,
        channels=lightweight_cnn_channels,
    )
    save_model(
        result,
        method_name,
        h5_path,
        info,
        dataset_key,
        dataset_label,
        eval_batch_size=lightweight_cnn_eval_batch_size,
        preprocessing="subtract_train_global_mean_divide_train_pixel_range",
    )

    learning_curve_path = learning_curve_dir / f"{dataset_key}_{method_name}_learning_curve.png"
    plot_learning_curves(
        history=result["trained_model"]["history"],
        output_path=learning_curve_path,
        title=f"{method_name} learning curve: {dataset_key}",
        best_epoch=result["trained_model"]["best_epoch"],
    )
    print(f"Saved learning curve: {learning_curve_path}")


def train_lightweight_cnn_site(h5_path, info, dataset_key, dataset_label, device, crop_cache_path=None):
    for method_name in lightweight_cnn_methods:
        if already_trained(dataset_key, method_name):
            continue

        train_lightweight_cnn_site_variant(
            h5_path=h5_path,
            info=info,
            dataset_key=dataset_key,
            dataset_label=dataset_label,
            device=device,
            method_name=method_name,
            crop_cache_path=crop_cache_path,
        )


def train_real_dataset(dataset_key, device):
    job = real_dataset_jobs[dataset_key]
    dataset_label = job["label"]
    h5_path = job["h5_path"]
    if not h5_path.exists():
        print(f"Skipping {dataset_label}: missing labeled dataset {h5_path}")
        return

    info = check_dataset_metadata(h5_path)

    print()
    print(f"Dataset: {dataset_label}")
    print(f"HDF5: {h5_path}")
    print("Exposure: unknown")
    print(f"Training images: {info['input_image_key']}")
    print(f"Labels/sites: {info['label_image_key']}")
    print(f"Images/sites: {info['n_images']} / {info['n_sites']}")

    crop_cache_path = None
    if use_site_crop_cache and (do_train_cnn_site or do_train_lightweight_cnn_site):
        crop_cache_path = build_site_crop_cache(
            h5_path=h5_path,
            cache_path=output_dir / "site_crop_cache" / f"{dataset_key}_crop{crop_size}.h5",
            crop_size=crop_size,
            batch_size=site_crop_cache_batch_size,
        )

    if do_train_baselines:
        train_baselines(h5_path, info, dataset_key, dataset_label, device)

    if do_train_mf_array:
        train_mf_array(h5_path, info, dataset_key, dataset_label, device)
    if do_train_mf_site:
        train_mf_site(h5_path, info, dataset_key, dataset_label, device)
    if do_train_cnn_site:
        train_cnn_site(h5_path, info, dataset_key, dataset_label, device, crop_cache_path=crop_cache_path)
    if do_train_lightweight_cnn_site:
        train_lightweight_cnn_site(
            h5_path,
            info,
            dataset_key,
            dataset_label,
            device,
            crop_cache_path=crop_cache_path,
        )


def main():
    device = 'cuda:0'

    print("Real-data site model training")
    print(f"Model dir: {model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in real_datasets_to_train:
        train_real_dataset(dataset_key, device)


if __name__ == "__main__":
    main()
