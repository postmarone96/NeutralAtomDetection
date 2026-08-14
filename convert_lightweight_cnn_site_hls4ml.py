from contextlib import contextmanager
import os
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch

from cnn_site import make_site_loader
from lightweight_cnn_site import LightweightCnnSiteNet


model_path = Path("models_3x3/qubit_dataset_sim_0.043_lightweight_cnn_site.joblib")
hls_output_dir = Path("lightweight_cnn_site_43ms").resolve()
project_name = "lightweight_cnn_site"

backend = "Vitis"
io_type = "io_parallel"
channels_last_conversion = "full"
default_precision = "ap_fixed<32,16>"
reuse_factor = 72
strategy = "Resource"
target_part = "xc7z020clg400-1"
clock_period_ns = 10
layer_reuse_factors = {
    "features_0": 9,
    "features_3": 72,
    "features_6": 72,
    "classifier_0": 64,
    "classifier_2": 16,
}
conv_resource_reuse_limits = {
    "features_0": 3 * 3 * 1,
    "features_3": 3 * 3 * 8,
    "features_6": 3 * 3 * 16,
}

n_validation_crops = 128
n_testbench_crops = 5
validation_h5_path = None
validation_indices = None
random_validation_seed = 42

write_project = True
run_hls_compile = True
run_hls_predict = True
run_synthesis = False
write_vitis_config = True
write_testbench_data_enabled = True
clean_output_dir = True


@contextmanager
def working_directory(path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


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
            "Copy the trained .joblib into WSL or edit model_path.\n"
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
    if validation_h5_path is not None:
        return Path(validation_h5_path)
    h5_path = checkpoint.get("h5_path")
    if h5_path is None:
        return None
    return Path(h5_path)


def checkpoint_indices(checkpoint, n_images):
    if validation_indices is not None:
        return np.asarray(validation_indices, dtype=np.int64)
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
        batch_size=n_validation_crops,
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
        if sum(batch.shape[0] for batch in crop_batches) >= n_validation_crops:
            break

    if not crop_batches:
        return None, None

    crops = torch.cat(crop_batches, dim=0)[:n_validation_crops]
    labels = torch.cat(label_batches, dim=0)[:n_validation_crops]
    return crops.numpy().astype(np.float32, copy=False), labels.numpy()


def make_random_validation_crops(crop_size):
    rng = np.random.default_rng(random_validation_seed)
    crops = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(n_validation_crops, 1, crop_size, crop_size),
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


def normalize_hls_predictions(predictions):
    if isinstance(predictions, list):
        predictions = predictions[0]
    predictions = np.asarray(predictions)
    predictions = np.squeeze(predictions)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 2)
    elif predictions.ndim > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)

    if predictions.shape[-1] != 2:
        raise ValueError(f"expected 2 output logits, got shape {predictions.shape}")

    return predictions


def hls_runtime_weights_dir():
    weights_dir = hls_output_dir / "firmware" / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"hls4ml runtime weights dir not found: {weights_dir}")
    return weights_dir


def hls_predict(hls_model, crops_nchw):
    with working_directory(hls_runtime_weights_dir()):
        predictions = hls_model.predict(crops_nchw)
    return normalize_hls_predictions(predictions)


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
    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(1, crop_size, crop_size),
        granularity="name",
        backend=backend,
        default_precision=default_precision,
        default_reuse_factor=reuse_factor,
        channels_last_conversion=channels_last_conversion,
    )

    config.setdefault("Model", {})
    config["Model"]["Strategy"] = strategy
    config["Model"]["ReuseFactor"] = int(reuse_factor)
    config["Model"].setdefault("Precision", {})
    config["Model"]["Precision"]["default"] = default_precision

    layer_config = config.get("LayerName", {})
    for layer_name, layer_reuse_factor in layer_reuse_factors.items():
        if layer_name in layer_config:
            layer_config[layer_name]["ReuseFactor"] = int(layer_reuse_factor)
            layer_config[layer_name]["Strategy"] = strategy

    return config


def validate_hls_config():
    if strategy != "Resource" or io_type != "io_parallel":
        return

    for layer_name, max_reuse_factor in conv_resource_reuse_limits.items():
        layer_reuse_factor = int(layer_reuse_factors.get(layer_name, reuse_factor))
        if layer_reuse_factor > max_reuse_factor:
            raise ValueError(
                f"{layer_name} reuse_factor={layer_reuse_factor} is invalid for "
                f"io_parallel Resource Conv2D. It must be <= {max_reuse_factor} "
                "(kernel_height * kernel_width * input_channels)."
            )


def convert_model(hls4ml, model, config, crop_size):
    conversion_kwargs = {
        "input_shape": (1, crop_size, crop_size),
        "hls_config": config,
        "output_dir": str(hls_output_dir),
        "project_name": project_name,
        "backend": backend,
        "io_type": io_type,
        "clock_period": clock_period_ns,
    }
    if target_part is not None:
        conversion_kwargs["part"] = target_part

    return hls4ml.converters.convert_from_pytorch_model(model, **conversion_kwargs)


def mirror_runtime_weights():
    source = hls_output_dir / "firmware" / "weights"
    destination = hls_output_dir / "weights"
    if not source.exists():
        print(f"Runtime weights not found, skipping mirror: {source}")
        return

    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Mirrored runtime weights: {destination}")


def write_vitis_hls_config():
    if not write_vitis_config:
        return

    config_path = hls_output_dir / "hls_config.cfg"
    part = target_part or "xc7z020clg400-1"
    config_text = f"""part={part}

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top={project_name}
syn.file=firmware/{project_name}.cpp
syn.cflags=-Ifirmware -Ifirmware/nnet_utils -Ifirmware/weights
tb.file={project_name}_test.cpp
tb.file=tb_data
tb.file=weights
tb.cflags=-Ifirmware -Ifirmware/nnet_utils -Ifirmware/weights
clock={clock_period_ns}ns
csim.code_analyzer=1
"""
    config_path.write_text(config_text, encoding="utf-8")
    print(f"Wrote Vitis HLS config: {config_path}")


def write_matrix(path, values):
    values = np.asarray(values, dtype=np.float32)
    with Path(path).open("w", encoding="utf-8") as file:
        for row in values:
            file.write(" ".join(f"{float(value):.10g}" for value in row))
            file.write("\n")


def write_testbench_data(checkpoint, model, crop_size):
    if not write_testbench_data_enabled:
        return

    crops, _, validation_source = make_validation_crops(checkpoint, crop_size)
    crops = crops[:n_testbench_crops].astype(np.float32, copy=False)
    predictions = torch_predict(model, crops).astype(np.float32, copy=False)

    tb_data_dir = hls_output_dir / "tb_data"
    tb_data_dir.mkdir(parents=True, exist_ok=True)

    input_path = tb_data_dir / "tb_input_features.dat"
    prediction_path = tb_data_dir / "tb_output_predictions.dat"

    write_matrix(input_path, crops.reshape(crops.shape[0], -1))
    write_matrix(prediction_path, predictions.reshape(predictions.shape[0], -1))

    print(f"Wrote testbench inputs: {input_path}")
    print(f"Wrote testbench predictions: {prediction_path}")
    print(f"  testbench crops:      {crops.shape[0]} from {validation_source}")


def maybe_run_synthesis(hls_model):
    if not run_synthesis:
        return
    print("Running HLS synthesis. This can take a long time.")
    hls_model.build(csim=True, synth=True, vsynth=False)


def main():
    hls4ml = import_hls4ml()

    checkpoint = load_checkpoint(model_path)
    model = load_float_model(checkpoint)
    crop_size = int(checkpoint.get("crop_size", 10))

    print("Lightweight CNN-site hls4ml conversion")
    print(f"  checkpoint:           {model_path}")
    print(f"  architecture:         {checkpoint.get('architecture', 'tiny_cnn_site_3conv_pool_8fc')}")
    print(f"  crop size:            {crop_size}x{crop_size}")
    print(f"  parameters:           {count_parameters(model):,}")
    print(f"  backend:              {backend}")
    print(f"  target part:          {target_part or 'hls4ml default'}")
    print(f"  clock period:         {clock_period_ns} ns")
    print(f"  io_type:              {io_type}")
    print(f"  channels conversion:  {channels_last_conversion}")
    print(f"  precision:            {default_precision}")
    print(f"  reuse factor:         {reuse_factor}")
    print(f"  layer reuse factors:  {layer_reuse_factors}")
    print(f"  output dir:           {hls_output_dir}")

    validate_hls_config()

    if clean_output_dir and hls_output_dir.exists():
        print(f"Cleaning existing hls4ml output dir: {hls_output_dir}")
        shutil.rmtree(hls_output_dir)

    config = make_hls_config(hls4ml, model, crop_size)
    hls_model = convert_model(hls4ml, model, config, crop_size)

    if write_project:
        print("Writing hls4ml project.")
        hls_model.write()
        mirror_runtime_weights()
        write_testbench_data(checkpoint, model, crop_size)
        write_vitis_hls_config()

    if run_hls_compile:
        print("Compiling hls4ml C simulation library.")
        hls_model.compile()

    if run_hls_predict:
        crops, labels, validation_source = make_validation_crops(checkpoint, crop_size)
        print(f"Validation input:       {validation_source}")
        torch_logits = torch_predict(model, crops)
        hls_logits = hls_predict(hls_model, crops)
        compare_outputs(torch_logits, hls_logits, labels=labels)

    maybe_run_synthesis(hls_model)

    print("Done.")


if __name__ == "__main__":
    main()
