"""Rules-based precision profile: every width from measured data or a port width."""

import csv
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

from common import detach, model_dir, model_file, project_root, results_dir, precision_csv
from model import load_lightweight_cnn_site_model, make_site_loader, prepare_site_batch


max_total_bits = {
    "weight": 18,
    "result": 27,
}

saturation_margin_bits = 0
target_method = None
selected_exposures = None
exact_range_tensors = ("weight", "bias", "accum", "integer")


def set_results_dir(path):
    global results_dir, precision_csv
    results_dir = Path(path)
    precision_csv = results_dir / "precisions.csv"

def integer_bits(values, margin_bits=saturation_margin_bits):
    values = np.asarray(values)
    max_abs = max(abs(values.min()), abs(values.max()))
    if max_abs == 0:
        return 1
    return max(1, int(np.floor(np.log2(max_abs))) + 2 + margin_bits)

def fractional(integer_bits, tensor, fraction=None):
    if tensor == "integer":
        return 0
    if fraction is None:
        return max_total_bits[tensor] - integer_bits
    return fraction

def precision(integer_bits, tensor="result", fraction=None):
    fractional_bits = fractional(integer_bits, tensor, fraction)
    return f"fixed<{integer_bits + fractional_bits},{integer_bits},RND,SAT>"

def infere_integer_bits_from(values, tensor):
    values = np.asarray(values).ravel()
    if tensor in exact_range_tensors:
        return integer_bits(values, margin_bits=0)
    return integer_bits([np.percentile(values, 0.1), np.percentile(values, 99.9)])

def product_fraction(module, layer_input):
    """Fw + Fa for one layer: the exact fraction of the products it sums."""
    weight_fraction = max_total_bits["weight"] - infere_integer_bits_from(detach(module.weight), "weight")
    activation_fraction = max_total_bits["result"] - infere_integer_bits_from(layer_input.cpu().numpy(), "result")
    return weight_fraction + activation_fraction


def stats_row(exposure, layer, tensor, values, fraction=None, input_layer=""):
    values = values.ravel()
    integer = infere_integer_bits_from(values, tensor)

    return {
        "exposure_s": exposure,
        "method": target_method,
        "layer": layer,
        "tensor": tensor,
        "input_layer": input_layer,
        "used_precision": precision(integer, tensor, fraction),
    }


def accumulator_bound(module, layer_input):
    weight = module.weight.detach().abs()
    largest_activation = 2.0 ** ( infere_integer_bits_from(detach(layer_input), "result") - 1 )
    if isinstance(module, torch.nn.Conv2d):
        weight_sum = weight.sum(dim=(1, 2, 3))
    else:
        weight_sum = weight.sum(dim=1)
    return weight_sum * largest_activation


def load_profile_batch(job, model):
    indices = job["val_idx"]
    with h5py.File(project_root / Path(job["h5_path"]), "r") as h5_file:
        n_sites = h5_file["sites"].shape[0]
    loader = make_site_loader(
        indices=indices,
        h5_path=project_root / Path(job["h5_path"]),
        crop_size=job["crop_size"],
        mean_image=job["mean_image"],
        pin_memory=False,
        scale=job["scale"],
        batch_size=len(indices) * n_sites,
        return_sites=model.neighbor_feature_count > 0,
    )
    crops, _labels, neighbor_features, _images, _sites = prepare_site_batch(next(iter(loader)), model, "cpu")
    return crops, neighbor_features


def source(layer_input, produced_by, crops):
    """Which profiled tensor feeds this layer."""
    name = produced_by.get(id(layer_input))
    if name is not None:
        return name
    return "__input__" if layer_input is crops else "cat"


def preprocessing(exposure, job, crops):
    scale = float(job["scale"])
    mean_image = float(job["mean_image"])
    raw = np.rint(crops.numpy() * scale + mean_image)
    raw_fraction = 0
    site_sum = raw.reshape(raw.shape[0], -1).sum(axis=1)

    return [
        stats_row(exposure, "__raw__", "integer", raw),
        stats_row(
            exposure, "__site_sum__", "accum", site_sum,
            fraction=raw_fraction, input_layer="__raw__",
        ),
    ]


def profile_model(model_path, exposure):
    job = joblib.load(model_path)
    model = load_lightweight_cnn_site_model(job, device="cpu").eval()
    crops, neighbor_features = load_profile_batch(job, model)
    activations = {}
    layer_inputs = {}
    produced_by = {}
    retained = []
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = detach(out)
            retained.append(out)
            produced_by[id(out)] = "model_" + name.replace(".", "_")
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layer_inputs[name] = (module, inp[0])

        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(crops, neighbor_features)

    for hook in hooks:
        hook.remove()

    rows = []
    rows.extend(preprocessing(exposure, job, crops))
    rows.append(stats_row(exposure, "__input__", "result", crops.numpy()))
    rows.append(
        stats_row(exposure, "__input___neighbor", "result", neighbor_features.numpy())
    )

    for name, values in activations.items():
        rows.append(
            stats_row(exposure, "model_" + name.replace(".", "_"), "result", values)
        )

    hidden = torch.from_numpy(activations["classifier.1"])
    rows.append(
        stats_row(
            exposure, "cat", "result",
            torch.cat([hidden, neighbor_features], dim=1).numpy(),
        )
    )

    accum_fraction = {}

    with torch.no_grad():
        for name, (module, layer_input) in layer_inputs.items():
            fraction = product_fraction(module, layer_input)
            row = stats_row(
                exposure, "model_" + name.replace(".", "_"), "accum",
                accumulator_bound(module, layer_input).cpu().numpy(),
                fraction=fraction, input_layer=source(layer_input, produced_by, crops),
            )
            rows.append(row)
            accum_fraction[row["layer"]] = fraction

    for layer_name, module in model.named_modules():
        if not layer_name:
            continue
        for param_name, param in module.named_parameters(recurse=False):
            table_name = "model_" + layer_name.replace(".", "_")
            rows.append(
                stats_row(
                    exposure, table_name, param_name, detach(param),
                    fraction=accum_fraction[table_name] if param_name == "bias" else None,
                )
            )
    return rows


def run():
    results_dir.mkdir(parents=True, exist_ok=True)
    exposures = sorted(selected_exposures)
    rows = []
    for exposure in exposures:
        model_path = model_file(model_dir, exposure, target_method)
        print(f"  {model_path.name}")
        rows.extend(profile_model(model_path, exposure))

    with open(precision_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {len(rows)} rows over {len(exposures)} exposures -> {precision_csv}")
    return precision_csv
