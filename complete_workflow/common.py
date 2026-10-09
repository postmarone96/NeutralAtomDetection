"""Paths and fixed-point helpers for the build stages."""

import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import constants
from utils import exposure_s, model_method

dataset_prefix = constants.dataset_prefix

model_dir = project_root / "models_3x3"
results_dir = Path(__file__).resolve().parent / "results"
metrics_csv = results_dir / "sensitivity.csv"
precision_csv = results_dir / "precisions.csv"

def model_file(model_dir, exposure, method):
    return Path(model_dir) / f"{dataset_prefix}_{exposure:g}_{method}.joblib"


precision_re = re.compile(r"<\s*(-?\d+)\s*,\s*(-?\d+)")


def width_of(precision):
    return int(precision_re.search(precision).group(1))


def integer_bits_of(precision):
    return int(precision_re.search(precision).group(2))


def activation_width(weight_bits):
    return int(round(weight_bits * 27.0 / 18.0))


def detach(this):
    return this.detach().cpu().numpy()


def accum_fraction_at(row, weight_precision, weight_bits, activation_fraction):
    weight_integer = integer_bits_of(weight_precision[row["layer"]])
    return weight_bits - weight_integer + activation_fraction[row["input_layer"]]
