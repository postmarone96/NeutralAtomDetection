from pathlib import Path
import shutil

import joblib
import numpy as np
import torch

from cnn_site import make_site_loader
from lightweight_cnn_site import LIGHTWEIGHT_CNN_SITE_ARCHITECTURE, LightweightCnnSiteNet


MODEL_PATH = Path("models_3x3/qubit_dataset_sim_0.043_lightweight_cnn_site.joblib")
HLS_OUTPUT_DIR = Path("hls4ml_projects/lightweight_cnn_site_30ms")
PROJECT_NAME = "lightweight_cnn_site"

BACKEND = "Vitis"
IO_TYPE = "io_parallel"
DEFAULT_PRECISION = "ap_fixed<16,6>"
REUSE_FACTOR = 72
STRATEGY = "Resource"
TARGET_PART = "xc7z020clg400-1"
CLOCK_PERIOD_NS = 10
LAYER_REUSE_FACTORS = {
    "features_0": 72,
    "features_3": 72,
    "features_6": 72,
    "classifier_0": 64,
    "classifier_2": 16,
}

N_VALIDATION_CROPS = 128
VALIDATION_H5_PATH = None
VALIDATION_INDICES = None
RANDOM_VALIDATION_SEED = 42

WRITE_PROJECT = True
RUN_HLS_COMPILE = False
RUN_HLS_PREDICT = False
RUN_SYNTHESIS = False
WRITE_VITIS_CONFIG = True


def import_hls4ml():
    try:
        import hls4ml
    except ImportError as exc:
        raise SystemExit(
            "hls4ml is not installed in this Python environment.\n"
            "In WSL run:\n"
            "  python3 -m venv .venv-hls4ml\n"
            "  source .venv-hls4ml/bin/activate\n"
            "  pip install --upgrade pip\n"
            "  pip install hls4ml torch numpy joblib h5py scikit-learn matplotlib tqdm onnx"
        ) from exc
    return hls4ml


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        candidates = sorted(Path("models_3x3").glob("*_lightweight_cnn_site.joblib"))
        candidate_text = "\n".join(f"  {candidate}" for candidate in candidates[:20])
        if len(candidates) > 20:
            candidate_text += f"\n  ... {len(candidates) - 20} more"
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Copy the trained .joblib into WSL or edit MODEL_PATH.\n"
            f"Available unconditioned lightweight checkpoints:\n{candidate_text or '  none'}"
        )

    checkpoint = joblib.load(path)
    if "state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain a state_dict")

    method = str(checkpoint.get("method", ""))
    architecture = str(checkpoint.get("architecture", ""))
    if "spacing_conditioned" in method or "spacing_conditioned" in architecture:
        raise ValueError(
            "This script only converts the unconditioned LightweightCnnSiteNet. "
            f"Got method={method!r}, architecture={architecture!r}."
        )

    return checkpoint


def load_float_model(checkpoint):
    crop_size = int(checkpoint.get("crop_size", 10))
    model = LightweightCnnSiteNet(crop_size=crop_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def checkpoint_h5_path(checkpoint):
    if VALIDATION_H5_PATH is not None:
        return Path(VALIDATION_H5_PATH)
    h5_path = checkpoint.get("h5_path")
    if h5_path is None:
        return None
    return Path(h5_path)


def checkpoint_indices(checkpoint, n_images):
    if VALIDATION_INDICES is not None:
        return np.asarray(VALIDATION_INDICES, dtype=np.int64)
    for key in ("test_idx", "val_idx", "train_idx"):
        if key in checkpoint:
            indices = np.asarray(checkpoint[key], dtype=np.int64)
            if indices.size:
                return indices
    return np.arange(min(n_images, 16), dtype=np.int64)


def make_real_validation_crops(checkpoint, crop_size):
    h5_path = checkpoint_h5_path(checkpoint)
    if h5_path is None or not h5_path.exists():
        return None, None

    import h5py

    with h5py.File(h5_path, "r") as h5_file:
        n_images = int(h5_file["images"].shape[0])

    indices = checkpoint_indices(checkpoint, n_images)
    loader = make_site_loader(
        h5_path=h5_path,
        indices=indices,
        crop_size=crop_size,
        mean_image=checkpoint.get("mean_image"),
        scale=checkpoint.get("scale"),
        batch_size=N_VALIDATION_CROPS,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    crop_batches = []
    label_batches = []
    for crops, labels in loader:
        crops = crops.reshape(-1, 1, crop_size, crop_size)
        labels = labels.reshape(-1)
        crop_batches.append(crops)
        label_batches.append(labels)
        if sum(batch.shape[0] for batch in crop_batches) >= N_VALIDATION_CROPS:
            break

    if not crop_batches:
        return None, None

    crops = torch.cat(crop_batches, dim=0)[:N_VALIDATION_CROPS]
    labels = torch.cat(label_batches, dim=0)[:N_VALIDATION_CROPS]
    return crops.numpy().astype(np.float32, copy=False), labels.numpy()


def make_random_validation_crops(crop_size):
    rng = np.random.default_rng(RANDOM_VALIDATION_SEED)
    crops = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(N_VALIDATION_CROPS, 1, crop_size, crop_size),
    ).astype(np.float32)
    return crops, None


def make_validation_crops(checkpoint, crop_size):
    crops, labels = make_real_validation_crops(checkpoint, crop_size)
    if crops is not None:
        return crops, labels, "real HDF5 site crops"

    crops, labels = make_random_validation_crops(crop_size)
    return crops, labels, "random fallback crops"


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def torch_predict(model, crops_nchw):
    with torch.no_grad():
        logits = model(torch.from_numpy(crops_nchw).float()).detach().cpu().numpy()
    return logits


def hls_predict_with_layout_fallback(hls_model, crops_nchw):
    attempts = [
        ("NCHW", crops_nchw),
        ("NHWC", np.transpose(crops_nchw, (0, 2, 3, 1))),
    ]

    errors = []
    for layout_name, hls_input in attempts:
        try:
            predictions = hls_model.predict(hls_input)
            if isinstance(predictions, list):
                predictions = predictions[0]
            predictions = np.asarray(predictions)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 2)
            return predictions, layout_name
        except Exception as exc:
            errors.append(f"{layout_name}: {exc}")

    raise RuntimeError(
        "hls_model.predict failed for both tensor layouts:\n"
        + "\n".join(f"  {error}" for error in errors)
    )


def compare_outputs(torch_logits, hls_logits, labels=None):
    n_outputs = min(torch_logits.shape[0], hls_logits.shape[0])
    torch_logits = torch_logits[:n_outputs]
    hls_logits = hls_logits[:n_outputs]

    torch_probabilities = softmax(torch_logits)
    hls_probabilities = softmax(hls_logits)

    torch_classes = torch_probabilities.argmax(axis=1)
    hls_classes = hls_probabilities.argmax(axis=1)

    print("Validation comparison:")
    print(f"  compared crops:        {n_outputs}")
    print(f"  max |logit diff|:      {np.max(np.abs(torch_logits - hls_logits)):.6g}")
    print(f"  mean |logit diff|:     {np.mean(np.abs(torch_logits - hls_logits)):.6g}")
    print(
        "  max |prob diff|:       "
        f"{np.max(np.abs(torch_probabilities - hls_probabilities)):.6g}"
    )
    print(
        "  class disagreement:    "
        f"{np.mean(torch_classes != hls_classes) * 100.0:.3f}%"
    )

    if labels is not None:
        labels = labels[:n_outputs]
        print(
            "  PyTorch accuracy:      "
            f"{np.mean(torch_classes == labels) * 100.0:.3f}%"
        )
        print(
            "  hls4ml accuracy:       "
            f"{np.mean(hls_classes == labels) * 100.0:.3f}%"
        )


def make_hls_config(hls4ml, model, crop_size):
    input_shape = (1, crop_size, crop_size)
    channels_last_conversion = "internal" if IO_TYPE == "io_stream" else "full"

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        granularity="name",
        backend=BACKEND,
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=REUSE_FACTOR,
        channels_last_conversion=channels_last_conversion,
    )

    config.setdefault("Model", {})
    config["Model"]["Strategy"] = STRATEGY
    config["Model"]["ReuseFactor"] = int(REUSE_FACTOR)

    layer_config = config.get("LayerName", {})
    for layer_name, reuse_factor in LAYER_REUSE_FACTORS.items():
        if layer_name in layer_config:
            layer_config[layer_name]["ReuseFactor"] = int(reuse_factor)
            layer_config[layer_name]["Strategy"] = STRATEGY

    return config


def convert_model(hls4ml, model, config, crop_size):
    conversion_kwargs = {
        "input_shape": (1, crop_size, crop_size),
        "hls_config": config,
        "output_dir": str(HLS_OUTPUT_DIR),
        "project_name": PROJECT_NAME,
        "backend": BACKEND,
        "io_type": IO_TYPE,
        "clock_period": CLOCK_PERIOD_NS,
    }
    if TARGET_PART is not None:
        conversion_kwargs["part"] = TARGET_PART

    return hls4ml.converters.convert_from_pytorch_model(model, **conversion_kwargs)


def mirror_runtime_weights():
    source = HLS_OUTPUT_DIR / "firmware" / "weights"
    destination = HLS_OUTPUT_DIR / "weights"
    if not source.exists():
        print(f"Runtime weights not found, skipping mirror: {source}")
        return

    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Mirrored runtime weights: {destination}")


def write_vitis_hls_config():
    if not WRITE_VITIS_CONFIG:
        return

    config_path = HLS_OUTPUT_DIR / "hls_config.cfg"
    part = TARGET_PART or "xc7z020clg400-1"
    config_text = f"""part={part}

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top={PROJECT_NAME}
syn.file=firmware/{PROJECT_NAME}.cpp
syn.cflags=-Ifirmware -Ifirmware/nnet_utils -Ifirmware/weights
tb.file={PROJECT_NAME}_test.cpp
tb.file=tb_data
tb.file=weights
tb.cflags=-Ifirmware -Ifirmware/nnet_utils -Ifirmware/weights
clock={CLOCK_PERIOD_NS}ns
csim.code_analyzer=1
"""
    config_path.write_text(config_text, encoding="utf-8")
    print(f"Wrote Vitis HLS config: {config_path}")


def maybe_run_synthesis(hls_model):
    if not RUN_SYNTHESIS:
        return
    print("Running HLS synthesis. This can take a long time.")
    hls_model.build(csim=True, synth=True, vsynth=False)


def main():
    hls4ml = import_hls4ml()

    checkpoint = load_checkpoint(MODEL_PATH)
    model = load_float_model(checkpoint)
    crop_size = int(checkpoint.get("crop_size", 10))

    print("Lightweight CNN-site hls4ml conversion")
    print(f"  checkpoint:           {MODEL_PATH}")
    print(f"  architecture:         {checkpoint.get('architecture', LIGHTWEIGHT_CNN_SITE_ARCHITECTURE)}")
    print(f"  crop size:            {crop_size}x{crop_size}")
    print(f"  parameters:           {count_parameters(model):,}")
    print(f"  backend:              {BACKEND}")
    print(f"  target part:          {TARGET_PART or 'hls4ml default'}")
    print(f"  clock period:         {CLOCK_PERIOD_NS} ns")
    print(f"  io_type:              {IO_TYPE}")
    print(f"  precision:            {DEFAULT_PRECISION}")
    print(f"  reuse factor:         {REUSE_FACTOR}")
    print(f"  layer reuse factors:  {LAYER_REUSE_FACTORS}")
    print(f"  output dir:           {HLS_OUTPUT_DIR}")

    config = make_hls_config(hls4ml, model, crop_size)
    hls_model = convert_model(hls4ml, model, config, crop_size)

    if WRITE_PROJECT:
        print("Writing hls4ml project.")
        hls_model.write()
        mirror_runtime_weights()
        write_vitis_hls_config()

    if RUN_HLS_COMPILE:
        print("Compiling hls4ml C simulation library.")
        hls_model.compile()

    if RUN_HLS_PREDICT:
        crops, labels, validation_source = make_validation_crops(checkpoint, crop_size)
        print(f"Validation input:       {validation_source}")

        torch_logits = torch_predict(model, crops)
        hls_logits, layout_name = hls_predict_with_layout_fallback(hls_model, crops)
        print(f"hls4ml predict layout:  {layout_name}")
        compare_outputs(torch_logits, hls_logits, labels=labels)

    maybe_run_synthesis(hls_model)

    print("Done.")


if __name__ == "__main__":
    main()
