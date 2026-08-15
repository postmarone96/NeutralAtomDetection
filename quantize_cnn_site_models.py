import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cnn_site import CnnSiteNet, make_site_loader
from mf_site import classification_fidelity


model_re = re.compile(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)_cnn_site\.joblib$")

data_dir = Path("dataset_3x3")
model_dir = Path("models_3x3")
output_dir = Path("output_3x3")

bits = [16, 12, 10, 8]
batch_size = 256
num_workers = 0
device = "cuda"

max_calib_images = None
min_frac_bits = -8
max_frac_bits = 16
activation_headroom = 1.10
weight_headroom = 1.10

# Set to None for all exposures, or e.g. [0.03] for only 30 ms.
selected_exposures = None

layer_specs = (
    ("conv1", "features.0", "input", "conv1_relu"),
    ("conv2", "features.2", "conv1_relu", "conv2_relu"),
    ("conv3", "features.4", "conv2_relu", "conv3_relu"),
    ("fc1", "classifier.0", "conv3_relu", "fc1_relu"),
    ("fc2", "classifier.2", "fc1_relu", "logits"),
)

activation_signedness = {
    "input": True,
    "conv1_relu": False,
    "conv2_relu": False,
    "conv3_relu": False,
    "fc1_relu": False,
    "logits": True,
}


@dataclass(frozen=True)
class FixedPointFormat:
    total_bits: int
    frac_bits: int
    signed: bool

    @property
    def int_min(self):
        if self.signed:
            return -(2 ** (self.total_bits - 1))
        return 0

    @property
    def int_max(self):
        if self.signed:
            return 2 ** (self.total_bits - 1) - 1
        return 2 ** self.total_bits - 1

    @property
    def scale(self):
        return 2.0 ** (-self.frac_bits)

    @property
    def real_min(self):
        return self.int_min * self.scale

    @property
    def real_max(self):
        return self.int_max * self.scale


def fake_quantize(values, fixed_format):
    scaled = torch.round(values * (2.0 ** fixed_format.frac_bits))
    clamped = torch.clamp(scaled, fixed_format.int_min, fixed_format.int_max)
    return clamped * fixed_format.scale


def choose_fixed_format(
    max_abs_value,
    total_bits,
    signed,
    min_frac_bits,
    max_frac_bits,
    headroom,
):
    max_abs_value = float(max_abs_value) * float(headroom)
    quantized_max = 2 ** (total_bits - 1) - 1 if signed else 2 ** total_bits - 1
    if not np.isfinite(max_abs_value) or max_abs_value <= 0.0:
        frac_bits = 0
    else:
        frac_bits = math.floor(math.log2(quantized_max / max_abs_value))
    frac_bits = int(np.clip(frac_bits, min_frac_bits, max_frac_bits))
    return FixedPointFormat(
        total_bits=int(total_bits),
        frac_bits=frac_bits,
        signed=bool(signed),
    )


def parse_model_path(path):
    match = model_re.fullmatch(path.name)
    if match is None:
        return None
    exposure_s = float(match.group(1))
    return exposure_s, exposure_s * 1000.0


def resolve_h5_path(model_object, data_dir, exposure_s):
    saved_h5_path = model_object.get("h5_path")
    if saved_h5_path is not None:
        saved_h5_path = Path(saved_h5_path)
        if saved_h5_path.exists():
            return saved_h5_path

    expected = data_dir / f"qubit_dataset_sim_{exposure_s:g}.h5"
    if expected.exists():
        return expected

    for path in sorted(data_dir.glob("qubit_dataset_sim_*.h5")):
        match = re.fullmatch(r"qubit_dataset_sim_([0-9]+(?:\.[0-9]+)?)\.h5", path.name)
        if match is not None and np.isclose(float(match.group(1)), exposure_s):
            return path

    raise FileNotFoundError(f"could not find dataset for exposure {exposure_s:g}")


def load_float_model(model_object, device):
    model = CnnSiteNet(crop_size=int(model_object.get("crop_size", 10))).to(device)
    model.load_state_dict(model_object["state_dict"])
    model.eval()
    return model


def flatten_site_batch(crops, labels, device):
    batch_size, site_count = labels.shape
    crops = crops.to(device).float()
    labels = labels.to(device).float()
    crops = crops.reshape(batch_size * site_count, 1, crops.shape[-2], crops.shape[-1])
    return crops, labels, batch_size, site_count


def forward_named_tensors(model, crops):
    tensors = {"input": crops}

    x = F.relu(model.features[0](crops))
    tensors["conv1_relu"] = x

    x = F.relu(model.features[2](x))
    tensors["conv2_relu"] = x

    x = F.relu(model.features[4](x))
    tensors["conv3_relu"] = x

    x = torch.flatten(x, start_dim=1)
    x = F.relu(model.classifier[0](x))
    tensors["fc1_relu"] = x

    tensors["logits"] = model.classifier[2](x)
    return tensors


@torch.no_grad()
def collect_activation_maxima(model, dataloader, device, max_calib_images):
    maxima = {name: 0.0 for name in activation_signedness}
    seen_images = 0

    model.eval()
    for crops, labels in dataloader:
        if max_calib_images is not None:
            remaining = int(max_calib_images) - seen_images
            if remaining <= 0:
                break
            crops = crops[:remaining]
            labels = labels[:remaining]

        if crops.numel() == 0:
            break

        seen_images += int(crops.shape[0])
        crops, _labels, _batch_size, _site_count = flatten_site_batch(
            crops,
            labels,
            device,
        )
        tensors = forward_named_tensors(model, crops)
        for name, tensor in tensors.items():
            maxima[name] = max(maxima[name], float(torch.max(torch.abs(tensor)).item()))

    return maxima


def module_by_path(model, module_path):
    module = model
    for part in module_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def build_quantization_config(
    model,
    activation_maxima,
    activation_bits,
    weight_bits,
    min_frac_bits,
    max_frac_bits,
    activation_headroom,
    weight_headroom,
):
    activation_formats = {
        name: choose_fixed_format(
            max_abs_value=max_abs_value,
            total_bits=activation_bits,
            signed=activation_signedness[name],
            min_frac_bits=min_frac_bits,
            max_frac_bits=max_frac_bits,
            headroom=activation_headroom,
        )
        for name, max_abs_value in activation_maxima.items()
    }

    layer_formats = {}
    for layer_name, module_path, input_activation, _output_activation in layer_specs:
        module = module_by_path(model, module_path)
        weight_format = choose_fixed_format(
            max_abs_value=float(torch.max(torch.abs(module.weight)).item()),
            total_bits=weight_bits,
            signed=True,
            min_frac_bits=min_frac_bits,
            max_frac_bits=max_frac_bits,
            headroom=weight_headroom,
        )
        input_format = activation_formats[input_activation]
        bias_format = FixedPointFormat(
            total_bits=32,
            frac_bits=input_format.frac_bits + weight_format.frac_bits,
            signed=True,
        )
        layer_formats[layer_name] = {
            "weight": weight_format,
            "bias": bias_format,
        }

    return {
        "activation_formats": activation_formats,
        "layer_formats": layer_formats,
    }


class QuantizedCnnSiteNet(nn.Module):
    def __init__(self, float_model, quantization_config):
        super().__init__()
        self.float_model = float_model
        self.quantization_config = quantization_config

    def quantize_activation(self, activation_name, value):
        return fake_quantize(
            value,
            self.quantization_config["activation_formats"][activation_name],
        )

    def quantize_parameter(self, layer_name, parameter_kind, value):
        return fake_quantize(
            value,
            self.quantization_config["layer_formats"][layer_name][parameter_kind],
        )

    def forward(self, x):
        x = self.quantize_activation("input", x)

        x = F.conv2d(
            x,
            self.quantize_parameter("conv1", "weight", self.float_model.features[0].weight),
            self.quantize_parameter("conv1", "bias", self.float_model.features[0].bias),
        )
        x = self.quantize_activation("conv1_relu", F.relu(x))

        x = F.conv2d(
            x,
            self.quantize_parameter("conv2", "weight", self.float_model.features[2].weight),
            self.quantize_parameter("conv2", "bias", self.float_model.features[2].bias),
        )
        x = self.quantize_activation("conv2_relu", F.relu(x))

        x = F.conv2d(
            x,
            self.quantize_parameter("conv3", "weight", self.float_model.features[4].weight),
            self.quantize_parameter("conv3", "bias", self.float_model.features[4].bias),
        )
        x = self.quantize_activation("conv3_relu", F.relu(x))

        x = torch.flatten(x, start_dim=1)
        x = F.linear(
            x,
            self.quantize_parameter("fc1", "weight", self.float_model.classifier[0].weight),
            self.quantize_parameter("fc1", "bias", self.float_model.classifier[0].bias),
        )
        x = self.quantize_activation("fc1_relu", F.relu(x))

        x = F.linear(
            x,
            self.quantize_parameter("fc2", "weight", self.float_model.classifier[2].weight),
            self.quantize_parameter("fc2", "bias", self.float_model.classifier[2].bias),
        )
        return self.quantize_activation("logits", x)


@torch.no_grad()
def scores_and_labels(model, dataloader, device):
    all_scores = []
    all_labels = []

    model.eval()
    for crops, labels in dataloader:
        crops, labels, batch_size, site_count = flatten_site_batch(
            crops,
            labels,
            device,
        )
        logits = model(crops)
        scores = torch.softmax(logits, dim=1)[:, 1].reshape(batch_size, site_count)
        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def summarize_scores(scores, labels, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim == 0:
        thresholds = np.full(labels.shape[1], float(thresholds), dtype=np.float32)

    fidelities = np.zeros(labels.shape[1], dtype=np.float32)
    for site_index in range(labels.shape[1]):
        fidelities[site_index] = classification_fidelity(
            scores[:, site_index],
            labels[:, site_index],
            thresholds[site_index],
        )

    predictions = scores >= thresholds[None, :]
    targets = labels > 0.5
    errors = predictions != targets

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "mean_infidelity": float(1.0 - np.mean(fidelities)),
        "raw_accuracy": float(1.0 - errors.mean()),
        "total_errors": int(errors.sum()),
    }


def evaluate_model_path(
    model_path,
    data_dir,
    bits,
    batch_size,
    num_workers,
    device,
    max_calib_images,
    min_frac_bits,
    max_frac_bits,
    activation_headroom,
    weight_headroom,
):
    exposure_s, exposure_ms = parse_model_path(model_path)
    model_object = joblib.load(model_path)
    h5_path = resolve_h5_path(model_object, data_dir, exposure_s)

    calibration_loader = make_site_loader(
        h5_path=h5_path,
        indices=np.asarray(model_object["val_idx"], dtype=np.int64),
        crop_size=int(model_object.get("crop_size", 10)),
        mean_image=model_object.get("mean_image"),
        scale=model_object.get("scale"),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )
    test_loader = make_site_loader(
        h5_path=h5_path,
        indices=np.asarray(model_object["test_idx"], dtype=np.int64),
        crop_size=int(model_object.get("crop_size", 10)),
        mean_image=model_object.get("mean_image"),
        scale=model_object.get("scale"),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )

    float_model = load_float_model(model_object, device)
    thresholds = model_object.get("thresholds", model_object.get("threshold"))

    float_scores, labels = scores_and_labels(float_model, test_loader, device)
    float_summary = summarize_scores(float_scores, labels, thresholds)

    records = [
        {
            "exposure_s": exposure_s,
            "exposure_ms": exposure_ms,
            "method": "float32",
            "bits": "float32",
            "mean_fidelity": float_summary["mean_fidelity"],
            "mean_infidelity": float_summary["mean_infidelity"],
            "raw_accuracy": float_summary["raw_accuracy"],
            "total_errors": float_summary["total_errors"],
            "delta_mean_fidelity": 0.0,
            "delta_mean_fidelity_percent_points": 0.0,
            "relative_mean_fidelity_percent": 0.0,
            "relative_mean_infidelity_percent": 0.0,
            "relative_total_errors_percent": 0.0,
            "model_path": str(model_path),
        }
    ]

    activation_maxima = collect_activation_maxima(
        model=float_model,
        dataloader=calibration_loader,
        device=device,
        max_calib_images=max_calib_images,
    )

    for bit_width in bits:
        quantization_config = build_quantization_config(
            model=float_model,
            activation_maxima=activation_maxima,
            activation_bits=bit_width,
            weight_bits=bit_width,
            min_frac_bits=min_frac_bits,
            max_frac_bits=max_frac_bits,
            activation_headroom=activation_headroom,
            weight_headroom=weight_headroom,
        )
        quantized_model = QuantizedCnnSiteNet(
            float_model=float_model,
            quantization_config=quantization_config,
        ).to(device)

        quant_scores, quant_labels = scores_and_labels(quantized_model, test_loader, device)
        quant_summary = summarize_scores(quant_scores, quant_labels, thresholds)
        delta_mean_fidelity = (
            quant_summary["mean_fidelity"] - float_summary["mean_fidelity"]
        )
        delta_mean_infidelity = (
            quant_summary["mean_infidelity"] - float_summary["mean_infidelity"]
        )
        delta_total_errors = (
            quant_summary["total_errors"] - float_summary["total_errors"]
        )
        records.append(
            {
                "exposure_s": exposure_s,
                "exposure_ms": exposure_ms,
                "method": f"fixed{bit_width}",
                "bits": int(bit_width),
                "mean_fidelity": quant_summary["mean_fidelity"],
                "mean_infidelity": quant_summary["mean_infidelity"],
                "raw_accuracy": quant_summary["raw_accuracy"],
                "total_errors": quant_summary["total_errors"],
                "delta_mean_fidelity": delta_mean_fidelity,
                "delta_mean_fidelity_percent_points": delta_mean_fidelity * 100.0,
                "relative_mean_fidelity_percent": (
                    0.0
                    if float_summary["mean_fidelity"] == 0.0
                    else delta_mean_fidelity / float_summary["mean_fidelity"] * 100.0
                ),
                "relative_mean_infidelity_percent": (
                    0.0
                    if float_summary["mean_infidelity"] == 0.0
                    else delta_mean_infidelity / float_summary["mean_infidelity"] * 100.0
                ),
                "relative_total_errors_percent": (
                    0.0
                    if float_summary["total_errors"] == 0
                    else delta_total_errors / float_summary["total_errors"] * 100.0
                ),
                "model_path": str(model_path),
            }
        )

    return records


def collect_model_paths(model_dir, selected_exposures):
    paths = []
    for path in sorted(model_dir.glob("qubit_dataset_sim_*_cnn_site.joblib")):
        parsed = parse_model_path(path)
        if parsed is None:
            continue
        exposure_s, _exposure_ms = parsed
        if selected_exposures and not any(np.isclose(exposure_s, selected) for selected in selected_exposures):
            continue
        paths.append(path)
    return paths


def write_csv(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "bits",
        "exposure_s",
        "exposure_ms",
        "mean_fidelity",
        "mean_infidelity",
        "raw_accuracy",
        "total_errors",
        "delta_mean_fidelity",
        "delta_mean_fidelity_percent_points",
        "relative_mean_fidelity_percent",
        "relative_mean_infidelity_percent",
        "relative_total_errors_percent",
        "model_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_results_joblib(
    records,
    output_path,
    bits,
    model_dir,
    data_dir,
    max_calib_images,
    min_frac_bits,
    max_frac_bits,
    activation_headroom,
    weight_headroom,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "records": records,
            "bits": [int(bit_width) for bit_width in bits],
            "model_dir": str(model_dir),
            "data_dir": str(data_dir),
            "max_calib_images": max_calib_images,
            "min_frac_bits": int(min_frac_bits),
            "max_frac_bits": int(max_frac_bits),
            "activation_headroom": float(activation_headroom),
            "weight_headroom": float(weight_headroom),
        },
        output_path,
    )


def method_sort_key(method):
    if method == "float32":
        return 0, 0
    match = re.fullmatch(r"fixed([0-9]+)", method)
    if match is not None:
        return 1, -int(match.group(1))
    return 2, method


def print_table(records):
    records = sorted(
        records,
        key=lambda record: (
            record["exposure_ms"],
            method_sort_key(record["method"]),
        ),
    )
    print(
        f"{'exposure ms':>12} "
        f"{'method':<10} "
        f"{'mean F':>12} "
        f"{'1-F':>12} "
        f"{'accuracy':>12} "
        f"{'errors':>8} "
        f"{'delta F':>12}"
    )
    print("-" * 86)
    for record in records:
        print(
            f"{record['exposure_ms']:>12.4g} "
            f"{record['method']:<10} "
            f"{record['mean_fidelity']:>12.8f} "
            f"{record['mean_infidelity']:>12.8f} "
            f"{record['raw_accuracy']:>12.8f} "
            f"{record['total_errors']:>8d} "
            f"{record['delta_mean_fidelity']:>+12.8f}"
        )


def main(
    model_dir=model_dir,
    data_dir=data_dir,
    output_dir=output_dir,
    bits=bits,
    batch_size=batch_size,
    num_workers=num_workers,
    device=device,
    max_calib_images=max_calib_images,
    min_frac_bits=min_frac_bits,
    max_frac_bits=max_frac_bits,
    activation_headroom=activation_headroom,
    weight_headroom=weight_headroom,
    selected_exposures=selected_exposures,
):
    model_paths = collect_model_paths(model_dir, selected_exposures)
    if not model_paths:
        raise FileNotFoundError(f"no CNN-site models found in {model_dir}")

    records = []
    for model_path in model_paths:
        records.extend(
            evaluate_model_path(
                model_path=model_path,
                data_dir=data_dir,
                bits=bits,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                max_calib_images=max_calib_images,
                min_frac_bits=min_frac_bits,
                max_frac_bits=max_frac_bits,
                activation_headroom=activation_headroom,
                weight_headroom=weight_headroom,
            )
        )

    results_path = output_dir / "cnn_site_quantization_vs_exposure.joblib"
    csv_path = output_dir / "cnn_site_quantization_vs_exposure.csv"
    write_results_joblib(
        records=records,
        output_path=results_path,
        bits=bits,
        model_dir=model_dir,
        data_dir=data_dir,
        max_calib_images=max_calib_images,
        min_frac_bits=min_frac_bits,
        max_frac_bits=max_frac_bits,
        activation_headroom=activation_headroom,
        weight_headroom=weight_headroom,
    )
    write_csv(records, csv_path)
    print_table(records)
    print(f"\nSaved results: {results_path}")
    print(f"Saved CSV:     {csv_path}")


if __name__ == "__main__":
    main()
