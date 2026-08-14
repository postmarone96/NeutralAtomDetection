from pathlib import Path
from contextlib import contextmanager
import os
import shutil

import joblib
import numpy as np
import torch

from cnn_site import make_site_loader
from lightweight_cnn_site import LIGHTWEIGHT_CNN_SITE_ARCHITECTURE, LightweightCnnSiteNet


MODEL_PATH = Path("models_3x3/qubit_dataset_sim_0.043_lightweight_cnn_site.joblib")
HLS_OUTPUT_DIR = Path("lightweight_cnn_site_43ms").resolve()
PROJECT_NAME = "lightweight_cnn_site"

BACKEND = "Vitis"
IO_TYPE = "io_parallel"
CHANNELS_LAST_CONVERSION = "full"
DEFAULT_PRECISION = "ap_fixed<32,16>"
# Use a wider precision such as "ap_fixed<32,16>" first when debugging
# PyTorch-vs-hls4ml mismatches. Then reduce precision after outputs match.
REUSE_FACTOR = 72
STRATEGY = "Resource"
TARGET_PART = "xc7z020clg400-1"
CLOCK_PERIOD_NS = 10
LAYER_REUSE_FACTORS = {
    "features_0": 9,
    "features_3": 72,
    "features_6": 72,
    "classifier_0": 64,
    "classifier_2": 16,
}

CONV_RESOURCE_REUSE_LIMITS = {
    "features_0": 3 * 3 * 1,
    "features_3": 3 * 3 * 8,
    "features_6": 3 * 3 * 16,
}

N_VALIDATION_CROPS = 128
N_TESTBENCH_CROPS = 5
VALIDATION_H5_PATH = None
VALIDATION_INDICES = None
RANDOM_VALIDATION_SEED = 42

WRITE_PROJECT = True
RUN_HLS_COMPILE = True
RUN_HLS_PREDICT = True
RUN_HLS_TRACE = True
N_TRACE_CROPS = 4
RUN_SYNTHESIS = False
WRITE_VITIS_CONFIG = True
WRITE_TESTBENCH_DATA = True
CLEAN_OUTPUT_DIR = True


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


def torch_layer_trace(model, crops_nchw):
    x = torch.from_numpy(crops_nchw).float()
    outputs = {}

    with torch.no_grad():
        x = model.features[0](x)
        outputs["features_0"] = x.detach().cpu().numpy()
        x = model.features[1](x)
        outputs["features_1"] = x.detach().cpu().numpy()
        x = model.features[2](x)
        outputs["features_2"] = x.detach().cpu().numpy()
        x = model.features[3](x)
        outputs["features_3"] = x.detach().cpu().numpy()
        x = model.features[4](x)
        outputs["features_4"] = x.detach().cpu().numpy()
        x = model.features[5](x)
        outputs["features_5"] = x.detach().cpu().numpy()
        x = model.features[6](x)
        outputs["features_6"] = x.detach().cpu().numpy()
        x = model.features[7](x)
        outputs["features_7"] = x.detach().cpu().numpy()
        x = model.features[8](x)
        outputs["features_8"] = x.detach().cpu().numpy()
        x = model.classifier[0](x)
        outputs["classifier_0"] = x.detach().cpu().numpy()
        x = model.classifier[1](x)
        outputs["classifier_1"] = x.detach().cpu().numpy()
        x = model.classifier[2](x)
        outputs["classifier_2"] = x.detach().cpu().numpy()

    return outputs


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
    weights_dir = HLS_OUTPUT_DIR / "firmware" / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"hls4ml runtime weights dir not found: {weights_dir}")
    return weights_dir


def hls_predict(hls_model, hls_input):
    with working_directory(hls_runtime_weights_dir()):
        return hls_model.predict(hls_input)


def hls_predict_with_layout_fallback(hls_model, crops_nchw):
    attempts = [
        ("NCHW", crops_nchw),
        ("NHWC", np.transpose(crops_nchw, (0, 2, 3, 1))),
    ]

    errors = []
    for layout_name, hls_input in attempts:
        try:
            predictions = normalize_hls_predictions(hls_predict(hls_model, hls_input))
            return predictions, layout_name
        except Exception as exc:
            errors.append(f"{layout_name}: {exc}")

    raise RuntimeError(
        "hls_model.predict failed for both tensor layouts:\n"
        + "\n".join(f"  {error}" for error in errors)
    )


def hls_predict_best_layout(hls_model, crops_nchw, torch_logits):
    attempts = [
        ("NCHW", crops_nchw),
        ("NHWC", np.transpose(crops_nchw, (0, 2, 3, 1))),
    ]

    results = []
    errors = []
    for layout_name, hls_input in attempts:
        try:
            raw_predictions = hls_predict(hls_model, hls_input)
            predictions = normalize_hls_predictions(raw_predictions)
            print(
                f"  layout check {layout_name}: raw output shape "
                f"{np.asarray(raw_predictions[0] if isinstance(raw_predictions, list) else raw_predictions).shape}, "
                f"normalized {predictions.shape}"
            )

            n_outputs = min(torch_logits.shape[0], predictions.shape[0])
            mean_logit_diff = float(
                np.mean(np.abs(torch_logits[:n_outputs] - predictions[:n_outputs]))
            )
            results.append((mean_logit_diff, layout_name, predictions))
        except Exception as exc:
            errors.append(f"{layout_name}: {exc}")

    if not results:
        raise RuntimeError(
            "hls_model.predict failed for both tensor layouts:\n"
            + "\n".join(f"  {error}" for error in errors)
        )

    for mean_logit_diff, layout_name, _predictions in results:
        print(f"  layout check {layout_name}: mean |logit diff| = {mean_logit_diff:.6g}")

    results.sort(key=lambda item: item[0])
    mean_logit_diff, layout_name, predictions = results[0]
    return predictions, layout_name


def make_hls_input(crops_nchw, layout_name):
    if layout_name == "NCHW":
        return crops_nchw
    if layout_name == "NHWC":
        return np.transpose(crops_nchw, (0, 2, 3, 1))
    raise ValueError(f"unknown HLS input layout: {layout_name}")


def array_range_text(values):
    values = np.asarray(values)
    return (
        f"shape={values.shape}, "
        f"range=[{np.min(values):.6g}, {np.max(values):.6g}], "
        f"mean|x|={np.mean(np.abs(values)):.6g}"
    )


def best_trace_diff(torch_values, hls_values):
    torch_values = np.asarray(torch_values)
    hls_values = np.asarray(hls_values)
    candidates = []

    if hls_values.shape == torch_values.shape:
        candidates.append(("same", hls_values, torch_values))

    if hls_values.ndim == 4 and torch_values.ndim == 4:
        hls_nchw = np.transpose(hls_values, (0, 3, 1, 2))
        if hls_nchw.shape == torch_values.shape:
            candidates.append(("hls NHWC -> NCHW", hls_nchw, torch_values))

        torch_nhwc = np.transpose(torch_values, (0, 2, 3, 1))
        if hls_values.shape == torch_nhwc.shape:
            candidates.append(("torch NCHW -> NHWC", hls_values, torch_nhwc))

    if hls_values.size == torch_values.size:
        candidates.append(("flat", hls_values.reshape(-1), torch_values.reshape(-1)))

    if not candidates:
        return None, None

    best_name, best_hls, best_torch = min(
        candidates,
        key=lambda item: float(np.mean(np.abs(item[1] - item[2]))),
    )
    return best_name, float(np.mean(np.abs(best_hls - best_torch)))


def trace_hls_layers(hls_model, model, crops_nchw, layout_name):
    if not RUN_HLS_TRACE:
        return

    trace_crops = crops_nchw[:N_TRACE_CROPS]
    hls_input = make_hls_input(trace_crops, layout_name)
    torch_trace = torch_layer_trace(model, trace_crops)

    print("Tracing hls4ml layers.")
    try:
        with working_directory(hls_runtime_weights_dir()):
            trace_result = hls_model.trace(hls_input)
    except Exception as exc:
        print(f"  trace failed: {exc}")
        return

    if isinstance(trace_result, tuple) and len(trace_result) == 2:
        _predictions, hls_trace = trace_result
    else:
        hls_trace = trace_result

    if not isinstance(hls_trace, dict):
        print(f"  unexpected trace output type: {type(hls_trace).__name__}")
        return

    print("HLS trace summary:")
    for layer_name, hls_values in hls_trace.items():
        hls_values = np.asarray(hls_values)
        print(f"  {layer_name}: {array_range_text(hls_values)}")
        if layer_name in torch_trace:
            compare_mode, mean_diff = best_trace_diff(torch_trace[layer_name], hls_values)
            if mean_diff is None:
                print(
                    "    torch compare: shape mismatch "
                    f"torch={torch_trace[layer_name].shape}, hls={hls_values.shape}"
                )
            else:
                print(f"    torch compare: mean |diff|={mean_diff:.6g} ({compare_mode})")


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
    print(f"  torch logits shape:    {torch_logits.shape}")
    print(f"  hls logits shape:      {hls_logits.shape}")
    print(
        "  torch logits range:    "
        f"[{np.min(torch_logits):.6g}, {np.max(torch_logits):.6g}]"
    )
    print(
        "  hls logits range:      "
        f"[{np.min(hls_logits):.6g}, {np.max(hls_logits):.6g}]"
    )
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
    print(f"  torch class counts:    {np.bincount(torch_classes, minlength=2).tolist()}")
    print(f"  hls class counts:      {np.bincount(hls_classes, minlength=2).tolist()}")

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

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        granularity="name",
        backend=BACKEND,
        default_precision=DEFAULT_PRECISION,
        default_reuse_factor=REUSE_FACTOR,
        channels_last_conversion=CHANNELS_LAST_CONVERSION,
    )

    config.setdefault("Model", {})
    config["Model"]["Strategy"] = STRATEGY
    config["Model"]["ReuseFactor"] = int(REUSE_FACTOR)
    config["Model"].setdefault("Precision", {})
    config["Model"]["Precision"]["default"] = DEFAULT_PRECISION
    if RUN_HLS_TRACE:
        config["Model"]["Trace"] = True

    layer_config = config.get("LayerName", {})
    for layer_name, reuse_factor in LAYER_REUSE_FACTORS.items():
        if layer_name in layer_config:
            layer_config[layer_name]["ReuseFactor"] = int(reuse_factor)
            layer_config[layer_name]["Strategy"] = STRATEGY

    if RUN_HLS_TRACE:
        for values in layer_config.values():
            values["Trace"] = True

    return config


def validate_hls_config():
    if STRATEGY != "Resource" or IO_TYPE != "io_parallel":
        return

    for layer_name, max_reuse_factor in CONV_RESOURCE_REUSE_LIMITS.items():
        reuse_factor = int(LAYER_REUSE_FACTORS.get(layer_name, REUSE_FACTOR))
        if reuse_factor > max_reuse_factor:
            raise ValueError(
                f"{layer_name} reuse_factor={reuse_factor} is invalid for "
                f"io_parallel Resource Conv2D. It must be <= {max_reuse_factor} "
                "(kernel_height * kernel_width * input_channels)."
            )


def print_hls_config_summary(config):
    layer_config = config.get("LayerName", {})
    print(f"  channels conversion:  {CHANNELS_LAST_CONVERSION}")
    print(f"  hls config layers:    {list(layer_config.keys())}")


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


def write_matrix(path, values):
    values = np.asarray(values, dtype=np.float32)
    with Path(path).open("w", encoding="utf-8") as file:
        for row in values:
            file.write(" ".join(f"{float(value):.10g}" for value in row))
            file.write("\n")


def write_testbench_data(checkpoint, model, crop_size):
    if not WRITE_TESTBENCH_DATA:
        return

    crops, _, validation_source = make_validation_crops(checkpoint, crop_size)
    crops = crops[:N_TESTBENCH_CROPS].astype(np.float32, copy=False)
    predictions = torch_predict(model, crops).astype(np.float32, copy=False)

    tb_data_dir = HLS_OUTPUT_DIR / "tb_data"
    tb_data_dir.mkdir(parents=True, exist_ok=True)

    input_path = tb_data_dir / "tb_input_features.dat"
    prediction_path = tb_data_dir / "tb_output_predictions.dat"

    write_matrix(input_path, crops.reshape(crops.shape[0], -1))
    write_matrix(prediction_path, predictions.reshape(predictions.shape[0], -1))

    print(f"Wrote testbench inputs: {input_path}")
    print(f"Wrote testbench predictions: {prediction_path}")
    print(f"  testbench crops:      {crops.shape[0]} from {validation_source}")


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
    print(f"  channels conversion:  {CHANNELS_LAST_CONVERSION}")
    print(f"  precision:            {DEFAULT_PRECISION}")
    print(f"  reuse factor:         {REUSE_FACTOR}")
    print(f"  layer reuse factors:  {LAYER_REUSE_FACTORS}")
    print(f"  output dir:           {HLS_OUTPUT_DIR}")

    validate_hls_config()

    if CLEAN_OUTPUT_DIR and HLS_OUTPUT_DIR.exists():
        print(f"Cleaning existing hls4ml output dir: {HLS_OUTPUT_DIR}")
        shutil.rmtree(HLS_OUTPUT_DIR)

    config = make_hls_config(hls4ml, model, crop_size)
    print_hls_config_summary(config)
    hls_model = convert_model(hls4ml, model, config, crop_size)

    if WRITE_PROJECT:
        print("Writing hls4ml project.")
        hls_model.write()
        mirror_runtime_weights()
        write_testbench_data(checkpoint, model, crop_size)
        write_vitis_hls_config()
        print(f"hls4ml runtime cwd:     {hls_runtime_weights_dir()}")

    if RUN_HLS_COMPILE:
        print("Compiling hls4ml C simulation library.")
        hls_model.compile()

    if RUN_HLS_PREDICT:
        crops, labels, validation_source = make_validation_crops(checkpoint, crop_size)
        print(f"Validation input:       {validation_source}")
        print(f"  crop tensor:          {array_range_text(crops)}")

        torch_logits = torch_predict(model, crops)
        hls_logits, layout_name = hls_predict_best_layout(hls_model, crops, torch_logits)
        print(f"hls4ml predict layout:  {layout_name}")
        compare_outputs(torch_logits, hls_logits, labels=labels)
        trace_hls_layers(hls_model, model, crops, layout_name)

    maybe_run_synthesis(hls_model)

    print("Done.")


if __name__ == "__main__":
    main()
