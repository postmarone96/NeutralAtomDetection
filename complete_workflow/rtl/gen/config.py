"""One build's shape, precisions and weights, at the widths optimize accepted.

The precisions come from widths.make_transform, the function optimize scores
with, so the RTL is built at exactly the widths the search accepted.
"""

import json
import sys
from pathlib import Path

import joblib

workflow_root = Path(__file__).resolve().parents[2]
if str(workflow_root) not in sys.path:
    sys.path.insert(0, str(workflow_root))

from common import model_dir, model_file, results_dir
from fixed_point import layer_module
from model import load_lightweight_cnn_site_model
import run_pipeline
from widths import acceptance_pp, make_transform, resolve_widths

method = "lightweight_cnn_site_neighbor"
exposure = 0.01
mac_units = 100

conv_layers = ("model_features_0", "model_features_3", "model_features_6")
conv_output_layers = ("model_features_2", "model_features_5", "model_features_8")
dense_layers = ("model_classifier_0", "model_classifier_2", "model_classifier_4")
layer_order = conv_layers + dense_layers
layer_input_stages = (
    "__input__",
    "model_features_2",
    "model_features_5",
    "model_features_8",
    "cat",
    "model_classifier_3",
)


def load(exposure=exposure, method=method):
    """Everything the generators need, in one place."""
    path = model_file(model_dir, exposure, method)
    job = joblib.load(path)
    model = load_lightweight_cnn_site_model(job, device="cpu").eval()

    build_results = results_dir / run_pipeline.label_for(method)
    run_pipeline.configure(method, build_results)
    best = json.loads((build_results / f"best_{exposure:g}s.json").read_text())
    resolved = resolve_widths(path, {exposure: best["total_widths"]}, acceptance_pp)
    transform = make_transform(resolved["widths"], resolved["integer_bits"])
    rows = transform(resolved["precision_rows"])

    precision = {}
    for row in rows:
        precision[(row["layer"], row["tensor"])] = row["used_precision"]

    weights = {}
    biases = {}
    for layer in layer_order:
        module = layer_module(model, layer)
        weights[layer] = module.weight.detach().cpu().numpy()
        biases[layer] = module.bias.detach().cpu().numpy()

    crop = int(job["crop_size"])
    c1, c2, c3 = tuple(job["channels"])
    return {
        "exposure": exposure,
        "method": method,
        "model_path": path,
        "crop": crop,
        "channels": (c1, c2, c3),
        "neighbors": int(model.neighbor_feature_count),
        "classes": 2,
        "spatial": (crop, crop // 2, crop // 4),
        "flat": c3 * (crop // 4) ** 2,
        "mac_units": mac_units,
        "precision": precision,
        "weights": weights,
        "biases": biases,
        "scale": float(job["scale"]),
        "mean_image": job["mean_image"],
    }


def macs_per_site(shape):
    """Multiplies one crop costs, layer by layer."""
    crop = shape["crop"]
    c1, c2, c3 = shape["channels"]
    return {
        "model_features_0": 9 * c1 * crop**2,
        "model_features_3": 9 * c1 * c2 * (crop // 2) ** 2,
        "model_features_6": 9 * c2 * c3 * (crop // 4) ** 2,
        "model_classifier_0": shape["flat"] * 8,
        "model_classifier_2": (8 + shape["neighbors"]) * 8,
        "model_classifier_4": 8 * shape["classes"],
    }
