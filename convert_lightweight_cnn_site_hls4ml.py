from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import os
from pathlib import Path
import re
import shutil

import joblib
import numpy as np
import torch

from cnn_site import make_site_loader
from lightweight_cnn_site import LightweightCnnSiteNet


data_dir = Path("dataset_3x3")
model_dir = Path("models_3x3")
hls_base_output_dir = Path("hls_models")
hls_base_output_dir.mkdir(parents=True, exist_ok=True)
model_path = None
hls_output_dir = None
current_dataset_path = None
project_name = "lightweight_cnn_site"

backend = "Vitis"
io_type = "io_parallel"
channels_last_conversion = "full"
default_precision = "ap_fixed<8,6>"
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

n_validation_crops = None
validation_batch_size = 2048
n_random_validation_crops = 128
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
selected_exposures_ms = None
stop_after_first_failure = False
quiet_hls4ml_output = True
verbose_steps = False
write_summary_csv = True
summary_csv_name = "lightweight_cnn_site_hls4ml_summary.csv"
good_class_disagreement_percent = 1.0
good_accuracy_gap_pp = 1.0

dataset_re = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5$")


def parse_dataset_exposure(path):
    match = dataset_re.fullmatch(Path(path).name)
    if match is None:
        return None
    return float(match.group(1))


def seconds_token(exposure_s):
    return f"{float(exposure_s):g}"


def exposure_ms_token(exposure_s):
    value = float(exposure_s) * 1000.0
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def model_path_for_exposure(exposure_s):
    return model_dir / f"qubit_dataset_sim_{seconds_token(exposure_s)}_lightweight_cnn_site.joblib"


def output_dir_for_exposure(exposure_s):
    return (hls_base_output_dir / f"lightweight_cnn_site_{exposure_ms_token(exposure_s)}ms").resolve()


def project_name_for_exposure(exposure_s):
    return f"lightweight_cnn_site_{exposure_ms_token(exposure_s)}ms"


def exposure_selected(exposure_s):
    if selected_exposures_ms is None:
        return True
    exposure_ms = exposure_s * 1000.0
    return any(np.isclose(exposure_ms, selected) for selected in selected_exposures_ms)


def collect_dataset_exposures():
    exposures = []
    for path in sorted(data_dir.glob("qubit_dataset_sim_*.h5")):
        exposure_s = parse_dataset_exposure(path)
        if exposure_s is None or not exposure_selected(exposure_s):
            continue
        exposures.append((exposure_s, path))
    return sorted(exposures, key=lambda item: item[0])


def collect_conversion_jobs():
    jobs = []
    missing_models = []

    for exposure_s, dataset_path in collect_dataset_exposures():
        path = model_path_for_exposure(exposure_s)
        if not path.exists():
            missing_models.append((exposure_s, dataset_path, path))
            continue
        jobs.append(
            {
                "exposure_s": exposure_s,
                "dataset_path": dataset_path,
                "model_path": path,
                "hls_output_dir": output_dir_for_exposure(exposure_s),
                "project_name": project_name_for_exposure(exposure_s),
            }
        )

    return jobs, missing_models


@contextmanager
def working_directory(path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def maybe_suppress_output(enabled):
    if not enabled:
        yield
        return

    with Path(os.devnull).open("w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def print_step(message):
    if verbose_steps:
        print(message)


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
        candidates = sorted(model_dir.glob("*_lightweight_cnn_site.joblib"))
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
    if h5_path is not None and Path(h5_path).exists():
        return Path(h5_path)
    if current_dataset_path is not None:
        return Path(current_dataset_path)
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
        batch_size=validation_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )

    crop_batches = []
    label_batches = []
    n_loaded_crops = 0
    for crops, labels in loader:
        crops = crops.reshape(-1, 1, crop_size, crop_size)
        labels = labels.reshape(-1)
        crop_batches.append(crops)
        label_batches.append(labels)
        n_loaded_crops += crops.shape[0]
        if n_validation_crops is not None and n_loaded_crops >= n_validation_crops:
            break

    if not crop_batches:
        return None, None

    crops = torch.cat(crop_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    if n_validation_crops is not None:
        crops = crops[:n_validation_crops]
        labels = labels[:n_validation_crops]
    return crops.numpy().astype(np.float32, copy=False), labels.numpy()


def make_random_validation_crops(crop_size):
    rng = np.random.default_rng(random_validation_seed)
    crops = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(n_random_validation_crops, 1, crop_size, crop_size),
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

    metrics = {
        "compared_crops": int(n_outputs),
        "max_logit_diff": float(np.max(np.abs(torch_logits - hls_logits))),
        "mean_logit_diff": float(np.mean(np.abs(torch_logits - hls_logits))),
        "max_prob_diff": float(np.max(np.abs(torch_probabilities - hls_probabilities))),
        "class_disagreement_percent": float(
            np.mean(torch_classes != hls_classes) * 100.0
        ),
    }

    if labels is not None:
        labels = labels[:n_outputs]
        metrics["torch_accuracy_percent"] = float(np.mean(torch_classes == labels) * 100.0)
        metrics["hls_accuracy_percent"] = float(np.mean(hls_classes == labels) * 100.0)
        metrics["accuracy_gap_pp"] = (
            metrics["hls_accuracy_percent"] - metrics["torch_accuracy_percent"]
        )

    return metrics


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
        print_step(f"Runtime weights not found, skipping mirror: {source}")
        return

    shutil.copytree(source, destination, dirs_exist_ok=True)
    print_step(f"Mirrored runtime weights: {destination}")


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
    print_step(f"Wrote Vitis HLS config: {config_path}")


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

    print_step(f"Wrote testbench inputs: {input_path}")
    print_step(f"Wrote testbench predictions: {prediction_path}")
    print_step(f"  testbench crops:      {crops.shape[0]} from {validation_source}")


def maybe_run_synthesis(hls_model):
    if not run_synthesis:
        return
    print_step("Running HLS synthesis. This can take a long time.")
    with maybe_suppress_output(quiet_hls4ml_output):
        hls_model.build(csim=True, synth=True, vsynth=False)


def result_quality(result):
    if result.get("status") != "ok":
        return "fail"
    if not run_hls_predict:
        return "ok"

    disagreement = abs(float(result.get("class_disagreement_percent", 0.0)))
    accuracy_gap = abs(float(result.get("accuracy_gap_pp", 0.0)))
    if (
        disagreement <= good_class_disagreement_percent
        and accuracy_gap <= good_accuracy_gap_pp
    ):
        return "good"
    return "check"


def print_job_result(result):
    exposure_ms = result["exposure_ms"]
    quality = result_quality(result)

    if result.get("status") != "ok":
        print(
            f"  {exposure_ms:6g} ms  fail   "
            f"{result['error_type']}: {result['error']}"
        )
        return

    if not run_hls_predict:
        print(f"  {exposure_ms:6g} ms  ok     project written")
        return

    print(
        f"  {exposure_ms:6g} ms  {quality:5s}  "
        f"torch={result.get('torch_accuracy_percent', float('nan')):7.3f}%  "
        f"hls={result.get('hls_accuracy_percent', float('nan')):7.3f}%  "
        f"gap={result.get('accuracy_gap_pp', float('nan')):+7.3f} pp  "
        f"disagree={result['class_disagreement_percent']:6.3f}%"
    )


def print_summary(results, missing_models):
    print("\nSummary:")
    print("  ms       status  torch acc  hls acc    gap pp  disagree  mean logit  max prob")
    for result in results:
        exposure_ms = result["exposure_ms"]
        quality = result_quality(result)

        if result.get("status") != "ok":
            print(f"  {exposure_ms:6g}  fail    {result['error_type']}: {result['error']}")
            continue

        if not run_hls_predict:
            print(f"  {exposure_ms:6g}  ok      project written")
            continue

        print(
            f"  {exposure_ms:6g}  {quality:5s}  "
            f"{result.get('torch_accuracy_percent', float('nan')):8.3f}%  "
            f"{result.get('hls_accuracy_percent', float('nan')):7.3f}%  "
            f"{result.get('accuracy_gap_pp', float('nan')):+7.3f}  "
            f"{result['class_disagreement_percent']:8.3f}%  "
            f"{result['mean_logit_diff']:10.4g}  "
            f"{result['max_prob_diff']:8.4g}"
        )

    checks = [
        result for result in results
        if result_quality(result) == "check"
    ]
    failures = [
        result for result in results
        if result_quality(result) == "fail"
    ]

    print("\nResult counts:")
    print(f"  good conversions:     {sum(result_quality(result) == 'good' for result in results)}")
    print(f"  check conversions:    {len(checks)}")
    print(f"  failed conversions:   {len(failures)}")
    print(f"  missing models:       {len(missing_models)}")

    if checks:
        check_text = ", ".join(f"{result['exposure_ms']:g} ms" for result in checks)
        print(f"  needs checking:       {check_text}")

    summary_path = write_summary_csv_file(results, missing_models)
    if summary_path is not None:
        print(f"  summary csv:          {summary_path}")


def write_summary_csv_file(results, missing_models):
    if not write_summary_csv:
        return None

    summary_path = (hls_base_output_dir / summary_csv_name).resolve()
    fieldnames = [
        "exposure_ms",
        "status",
        "quality",
        "torch_accuracy_percent",
        "hls_accuracy_percent",
        "accuracy_gap_pp",
        "class_disagreement_percent",
        "mean_logit_diff",
        "max_logit_diff",
        "max_prob_diff",
        "compared_crops",
        "model_path",
        "dataset_path",
        "output_dir",
        "error",
    ]

    rows = []
    for result in results:
        rows.append(
            {
                "exposure_ms": result["exposure_ms"],
                "status": result.get("status", ""),
                "quality": result_quality(result),
                "torch_accuracy_percent": result.get("torch_accuracy_percent", ""),
                "hls_accuracy_percent": result.get("hls_accuracy_percent", ""),
                "accuracy_gap_pp": result.get("accuracy_gap_pp", ""),
                "class_disagreement_percent": result.get(
                    "class_disagreement_percent", ""
                ),
                "mean_logit_diff": result.get("mean_logit_diff", ""),
                "max_logit_diff": result.get("max_logit_diff", ""),
                "max_prob_diff": result.get("max_prob_diff", ""),
                "compared_crops": result.get("compared_crops", ""),
                "model_path": result.get("model_path", ""),
                "dataset_path": result.get("dataset_path", ""),
                "output_dir": result.get("output_dir", ""),
                "error": result.get("error", ""),
            }
        )

    for exposure_s, dataset_path, path in missing_models:
        rows.append(
            {
                "exposure_ms": exposure_s * 1000.0,
                "status": "missing_model",
                "quality": "missing",
                "torch_accuracy_percent": "",
                "hls_accuracy_percent": "",
                "accuracy_gap_pp": "",
                "class_disagreement_percent": "",
                "mean_logit_diff": "",
                "max_logit_diff": "",
                "max_prob_diff": "",
                "compared_crops": "",
                "model_path": path,
                "dataset_path": dataset_path,
                "output_dir": "",
                "error": "matching lightweight checkpoint not found",
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def run_one_conversion(hls4ml, job):
    global model_path
    global hls_output_dir
    global current_dataset_path
    global project_name

    exposure_s = float(job["exposure_s"])
    exposure_ms = exposure_s * 1000.0
    current_dataset_path = Path(job["dataset_path"])
    model_path = Path(job["model_path"])
    hls_output_dir = Path(job["hls_output_dir"])
    project_name = str(job["project_name"])
    checkpoint = load_checkpoint(model_path)
    model = load_float_model(checkpoint)
    crop_size = int(checkpoint.get("crop_size", 10))
    result = {
        "status": "ok",
        "exposure_ms": exposure_ms,
        "dataset_path": current_dataset_path,
        "model_path": model_path,
        "output_dir": hls_output_dir,
        "project_name": project_name,
        "parameters": count_parameters(model),
    }

    validate_hls_config()

    if clean_output_dir and hls_output_dir.exists():
        print_step(f"Cleaning existing hls4ml output dir: {hls_output_dir}")
        shutil.rmtree(hls_output_dir)

    config = make_hls_config(hls4ml, model, crop_size)
    with maybe_suppress_output(quiet_hls4ml_output):
        hls_model = convert_model(hls4ml, model, config, crop_size)

    if write_project:
        print_step("Writing hls4ml project.")
        with maybe_suppress_output(quiet_hls4ml_output):
            hls_model.write()
        mirror_runtime_weights()
        write_testbench_data(checkpoint, model, crop_size)
        write_vitis_hls_config()

    if run_hls_compile:
        print_step("Compiling hls4ml C simulation library.")
        with maybe_suppress_output(quiet_hls4ml_output):
            hls_model.compile()

    if run_hls_predict:
        crops, labels, validation_source = make_validation_crops(checkpoint, crop_size)
        torch_logits = torch_predict(model, crops)
        with maybe_suppress_output(quiet_hls4ml_output):
            hls_logits = hls_predict(hls_model, crops)
        result["validation_source"] = validation_source
        result.update(compare_outputs(torch_logits, hls_logits, labels=labels))

    maybe_run_synthesis(hls_model)

    return result


def main():
    jobs, missing_models = collect_conversion_jobs()

    print("Lightweight CNN-site batch hls4ml conversion")
    print(f"  dataset dir:          {data_dir}")
    print(f"  model dir:            {model_dir}")
    print(f"  hls output base:      {hls_base_output_dir.resolve()}")
    print(f"  datasets found:       {len(collect_dataset_exposures())}")
    print(f"  conversion jobs:      {len(jobs)}")
    print(f"  missing models:       {len(missing_models)}")

    if missing_models:
        missing_text = ", ".join(
            f"{exposure_s * 1000.0:g} ms"
            for exposure_s, _, _ in missing_models
        )
        print(f"  skipped missing:      {missing_text}")

    if not jobs:
        raise FileNotFoundError(
            "No conversion jobs found. Check data_dir/model_dir and trained lightweight CNN-site models."
        )

    validate_hls_config()
    hls4ml = import_hls4ml()
    results = []

    print("\nRunning conversions:")
    for job_index, job in enumerate(jobs, start=1):
        exposure_ms = float(job["exposure_s"]) * 1000.0
        print(f"  [{job_index:2d}/{len(jobs):2d}] ", end="", flush=True)
        try:
            result = run_one_conversion(hls4ml, job)
        except Exception as exc:
            result = {
                "status": "fail",
                "exposure_ms": exposure_ms,
                "dataset_path": job["dataset_path"],
                "model_path": job["model_path"],
                "output_dir": job["hls_output_dir"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            if stop_after_first_failure:
                raise

        results.append(result)
        print_job_result(result)

    print_summary(results, missing_models)

    failures = [result for result in results if result.get("status") != "ok"]
    if failures:
        raise RuntimeError(f"{len(failures)} hls4ml conversion job(s) failed")

    print("\nAll conversion jobs completed.")


if __name__ == "__main__":
    main()
