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
    BASE_SEED,
    # CROP_SIZE,
    EPOCHS,
    LEARNING_RATE,
    make_random_split,
    # NUM_WORKERS,
    run_one_split as run_cnn_site_split,
)
from lightweight_cnn_site import (
    # CROP_SIZE as LIGHTWEIGHT_CNN_CROP_SIZE,
    EARLY_STOPPING_MIN_DELTA as LIGHTWEIGHT_CNN_EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE as LIGHTWEIGHT_CNN_EARLY_STOPPING_PATIENCE,
    EPOCHS as LIGHTWEIGHT_CNN_EPOCHS,
    LEARNING_RATE as LIGHTWEIGHT_CNN_LEARNING_RATE,
    LIGHTWEIGHT_CNN_SITE_ARCHITECTURE,
    LIGHTWEIGHT_CNN_SITE_METHOD,
    run_one_split as run_lightweight_cnn_site_split,
)
from mf_array import run_one_split as run_mf_array_split
from mf_site import classification_fidelity
from mf_site import run_one_split as run_mf_site_split
from utils import H5Dataset, get_device

CROP_SIZE = LIGHTWEIGHT_CNN_CROP_SIZE = 20
NUM_WORKERS = 20

DATA_DIR = Path("dataset_labeled")
MODEL_DIR = Path("models_3x3")
EXPOSURE_S = 0.030

TRAINING_IMAGE_KEY = "image_2"
LABEL_IMAGE_KEY = "image_1"

MF_S_VALUES = range(2, 15, 2)
MF_ALPHA = 1e-4
MF_BATCH_SIZE = 64

CNN_BATCH_SIZE = 8096
CNN_EVAL_BATCH_SIZE = 32768

LIGHTWEIGHT_CNN_BATCH_SIZE = 8096
LIGHTWEIGHT_CNN_EVAL_BATCH_SIZE = 32768

BASELINE_BATCH_SIZE = 32

MF_ARRAY_BATCH_SIZE = 64
MF_ARRAY_TRAIN_NORMAL = True
MF_ARRAY_TRAIN_GAUSSIAN = False
MF_ARRAY_GAUSSIAN_NORMALIZE = True
MF_ARRAY_NEIGHBOR_CONTEXT = "coordinate_8_nearest"

TRAIN_BASELINES = False
TRAIN_MF_SITE = False
TRAIN_MF_ARRAY = False
TRAIN_CNN_SITE = False
TRAIN_LIGHTWEIGHT_CNN_SITE = True
TRAIN_CNN_ARRAY = False


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

    if input_image_key != TRAINING_IMAGE_KEY:
        raise ValueError(
            f"expected training images from {TRAINING_IMAGE_KEY}, "
            f"but dataset says {input_image_key}"
        )
    if label_image_key != LABEL_IMAGE_KEY:
        raise ValueError(
            f"expected labels/sites from {LABEL_IMAGE_KEY}, "
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
        "exposure_s": float(EXPOSURE_S),
        "exposure_ms": float(1000.0 * EXPOSURE_S),
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
        "seed": BASE_SEED,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "s_values": list(MF_S_VALUES),
        "test_fidelity": test_fids,
        "test_mean_fidelity": float(np.mean(test_fids)),
        "test_errors_per_site": errors.sum(axis=0).astype(np.int64),
        "test_total_errors": int(errors.sum()),
        "test_raw_accuracy": float(1.0 - errors.mean()),
    }
    if extra_fields:
        save_obj.update(extra_fields)

    output_path = MODEL_DIR / f"qubit_dataset_sim_{EXPOSURE_S:g}_{method_name}.joblib"
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={save_obj['test_mean_fidelity']:.8f} | "
        f"raw acc={save_obj['test_raw_accuracy']:.8f}"
    )


def train_baselines(h5_path, info, device):
    train_idx, val_idx, test_idx = make_random_split(
        n_samples=info["n_images"],
        seed=BASE_SEED,
    )

    avg_image, sites = compute_average_image(
        h5_path=h5_path,
        indices=train_idx,
        batch_size=BASELINE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    fitted_sigmas = fit_sigmas_from_average_image(
        avg_image=avg_image,
        sites=sites,
        fit_size=9,
    )

    square_trained = square_with_s_sweep(
        h5_path=h5_path,
        val_idx=val_idx,
        s_values=MF_S_VALUES,
        batch_size=BASELINE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )
    gaussian_trained = gaussian_with_s_sweep(
        h5_path=h5_path,
        val_idx=val_idx,
        sigmas=fitted_sigmas,
        s_values=MF_S_VALUES,
        batch_size=BASELINE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
        normalize=False,
    )

    test_loader = DataLoader(
        H5Dataset(h5_path, indices=test_idx),
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=NUM_WORKERS > 0,
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
        split_seed=BASE_SEED,
        s_values=MF_S_VALUES,
        alpha=MF_ALPHA,
        batch_size=MF_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )

    trained = result["trained_model"]
    output_path = MODEL_DIR / f"qubit_dataset_sim_{EXPOSURE_S:g}_mf_site_real.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": "mf_site",
        "seed": BASE_SEED,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "s_values": list(MF_S_VALUES),
        "alpha": MF_ALPHA,
        "batch_size": MF_BATCH_SIZE,
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
        split_seed=BASE_SEED,
        s_values=MF_S_VALUES,
        alpha=MF_ALPHA,
        batch_size=MF_ARRAY_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
        gaussian=gaussian,
        gaussian_normalize=MF_ARRAY_GAUSSIAN_NORMALIZE,
        neighbor_context=MF_ARRAY_NEIGHBOR_CONTEXT,
    )

    method_name = "mf_array_gaussian_real" if gaussian else "mf_array_real"
    trained = result["trained_model"]
    output_path = MODEL_DIR / f"qubit_dataset_sim_{EXPOSURE_S:g}_{method_name}.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": method_name,
        "seed": BASE_SEED,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "s_values": list(MF_S_VALUES),
        "alpha": MF_ALPHA,
        "batch_size": MF_ARRAY_BATCH_SIZE,
        "thresholds": result["thresholds"],
        "test_fidelity": result["test_fidelity"],
        "test_mean_fidelity": result["test_mean_fidelity"],
        "test_raw_accuracy": result["test_raw_accuracy"],
        "test_errors_per_site": result["test_errors_per_site"],
        "test_total_errors": result["test_total_errors"],
        "mf_array_gaussian": bool(gaussian),
        "mf_array_context": "gaussian" if gaussian else "mean",
        "mf_array_sigmas": result["mf_array_sigmas"],
        "mf_array_gaussian_normalize": bool(MF_ARRAY_GAUSSIAN_NORMALIZE),
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
    if MF_ARRAY_TRAIN_NORMAL:
        train_mf_array_variant(h5_path, info, device, gaussian=False)
    if MF_ARRAY_TRAIN_GAUSSIAN:
        train_mf_array_variant(h5_path, info, device, gaussian=True)


def train_cnn_site(h5_path, info, device):
    result = run_cnn_site_split(
        h5_path=h5_path,
        split_seed=BASE_SEED,
        crop_size=CROP_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        eval_batch_size=CNN_EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )

    trained = result["trained_model"]
    output_path = MODEL_DIR / f"qubit_dataset_sim_{EXPOSURE_S:g}_cnn_site_real.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": "cnn_site",
        "seed": BASE_SEED,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "crop_size": int(CROP_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "epochs": int(EPOCHS),
        "batch_size": int(CNN_BATCH_SIZE),
        "eval_batch_size": int(CNN_EVAL_BATCH_SIZE),
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


def train_lightweight_cnn_site(h5_path, info, device):
    result = run_lightweight_cnn_site_split(
        h5_path=h5_path,
        split_seed=BASE_SEED,
        crop_size=LIGHTWEIGHT_CNN_CROP_SIZE,
        learning_rate=LIGHTWEIGHT_CNN_LEARNING_RATE,
        epochs=LIGHTWEIGHT_CNN_EPOCHS,
        batch_size=LIGHTWEIGHT_CNN_BATCH_SIZE,
        eval_batch_size=LIGHTWEIGHT_CNN_EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
        early_stopping_patience=LIGHTWEIGHT_CNN_EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=LIGHTWEIGHT_CNN_EARLY_STOPPING_MIN_DELTA,
    )

    trained = result["trained_model"]
    output_path = MODEL_DIR / f"qubit_dataset_sim_{EXPOSURE_S:g}_{LIGHTWEIGHT_CNN_SITE_METHOD}_real.joblib"
    save_obj = {
        **trained,
        **metadata(h5_path, info),
        "method": LIGHTWEIGHT_CNN_SITE_METHOD,
        "seed": BASE_SEED,
        "train_idx": result["train_idx"],
        "val_idx": result["val_idx"],
        "test_idx": result["test_idx"],
        "mean_image": result["mean_image"],
        "scale": result["scale"],
        "crop_size": int(LIGHTWEIGHT_CNN_CROP_SIZE),
        "learning_rate": float(LIGHTWEIGHT_CNN_LEARNING_RATE),
        "epochs": int(LIGHTWEIGHT_CNN_EPOCHS),
        "epochs_trained": result["epochs_trained"],
        "batch_size": int(LIGHTWEIGHT_CNN_BATCH_SIZE),
        "eval_batch_size": int(LIGHTWEIGHT_CNN_EVAL_BATCH_SIZE),
        "early_stopping_patience": (
            None
            if LIGHTWEIGHT_CNN_EARLY_STOPPING_PATIENCE is None
            else int(LIGHTWEIGHT_CNN_EARLY_STOPPING_PATIENCE)
        ),
        "early_stopping_min_delta": float(LIGHTWEIGHT_CNN_EARLY_STOPPING_MIN_DELTA),
        "stopped_epoch": result["stopped_epoch"],
        "stopped_early": result["stopped_early"],
        "thresholds": result["thresholds"],
        "test_fidelity": result["test_fidelity"],
        "test_mean_fidelity": result["test_mean_fidelity"],
        "test_raw_accuracy": result["test_raw_accuracy"],
        "test_errors_per_site": result["test_errors_per_site"],
        "test_total_errors": result["test_total_errors"],
        "preprocessing": "subtract_train_global_mean_divide_train_pixel_range",
        "architecture": LIGHTWEIGHT_CNN_SITE_ARCHITECTURE,
    }
    atomic_joblib_dump(save_obj, output_path)

    print(
        f"Saved {output_path} | "
        f"mean F={result['test_mean_fidelity']:.8f} | "
        f"best epoch={result['best_epoch']} | "
        f"trained epochs={result['epochs_trained']}"
    )


def main():
    h5_path = find_dataset(DATA_DIR)
    info = check_dataset_metadata(h5_path)
    device = 'cuda:1' # get_device()

    print(f"Dataset: {h5_path}")
    print(f"Exposure: {1000.0 * EXPOSURE_S:g} ms")
    print(f"Training images: {info['input_image_key']}")
    print(f"Labels/sites: {info['label_image_key']}")
    print(f"Images/sites: {info['n_images']} / {info['n_sites']}")
    print(f"Device: {device}")

    if TRAIN_BASELINES:
        train_baselines(h5_path, info, device)

    if TRAIN_MF_ARRAY:
        train_mf_array(h5_path, info, device)
    if TRAIN_CNN_SITE:
        train_cnn_site(h5_path, info, device)
    if TRAIN_LIGHTWEIGHT_CNN_SITE:
        train_lightweight_cnn_site(h5_path, info, device)
    if TRAIN_CNN_ARRAY:
        raise NotImplementedError(
            "CNN-array is disabled for the real 253-site dataset; "
            "the current cnn_array.py crops only the array center."
        )


if __name__ == "__main__":
    main()
