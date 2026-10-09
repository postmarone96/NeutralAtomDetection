"""Fidelity of the fixed-point network against its float32 self."""

import csv
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch

from common import *
import fixed_point
from mf_site import classification_fidelity
from model import load_lightweight_cnn_site_model, make_site_loader, prepare_site_batch

selected_method = None
loaded = {}


def set_results_dir(path):
    global results_dir, precision_csv
    results_dir = Path(path)
    precision_csv = results_dir / "precisions.csv"


def logit_decisions(logits):
    """1.0 where logit1 > logit0, the decision the FPGA makes."""
    return (logits[:, 1] > logits[:, 0]).astype(np.float32)


def mean_fidelity(decisions, labels):
    fidelities = [
        classification_fidelity(decisions[:, site], labels[:, site], 0.5)
        for site in range(labels.shape[1])
    ]
    return float(np.mean(fidelities))


def read_precisions(exposure, method):
    rows = []
    with open(precision_csv, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["exposure_s"]) == exposure and row["method"] == method:
                rows.append(row)
    return rows


def load_test_batch(job, model):
    """The whole test split in one batch: crops, labels, neighbour features."""
    test_idx = np.asarray(job["test_idx"])
    h5_path = project_root / Path(job["h5_path"])
    with h5py.File(h5_path, "r") as handle:
        n_sites = handle["sites"].shape[0]

    loader = make_site_loader(
        indices=test_idx,
        h5_path=h5_path,
        crop_size=job["crop_size"],
        mean_image=job["mean_image"],
        scale=job["scale"],
        batch_size=len(test_idx) * n_sites,
        # Vector generation is a one-shot batch. Keeping it in-process avoids
        # platform-specific shared-memory failures from DataLoader workers.
        num_workers=0,
        pin_memory=False,
        return_sites=True,
    )
    return prepare_site_batch(next(iter(loader)), model, "cpu")


def load(model_path):
    """Model, test split and float decisions: identical for every width tried."""
    key = Path(model_path)
    if key not in loaded:
        job = joblib.load(model_path)
        model = load_lightweight_cnn_site_model(job, device="cpu").eval()
        crops, labels, neighbor_features, image_count, site_count = load_test_batch(job, model)
        with torch.no_grad():
            logits = model(crops, neighbor_features).numpy()
        loaded[key] = {
            "model": model,
            "crops": crops.numpy(),
            "neighbor_features": neighbor_features.numpy(),
            "labels": labels.numpy().reshape(image_count, site_count),
            "float_decisions": logit_decisions(logits).reshape(image_count, site_count),
            "image_count": image_count,
            "site_count": site_count,
        }
    return loaded[key]


def precision_table(model_path, precision_transform=None):
    """{(layer, tensor): precision} as the profile wrote it, after any transform."""
    rows = read_precisions(exposure_s(model_path), model_method(model_path))
    if precision_transform is not None:
        rows = precision_transform(rows)
    return {(row["layer"], row["tensor"]): row["used_precision"] for row in rows}


def run_model(model_path, precision_transform=None):
    """Fixed-point vs float32 for one model.

    precision_transform, when given, rewrites the precision rows in memory
    first, so a width sweep never has to touch precisions.csv.
    """
    data = load(model_path)
    precision = precision_table(model_path, precision_transform)
    logits, _stages = fixed_point.forward(
        data["model"], precision, data["crops"], data["neighbor_features"]
    )

    labels = data["labels"]
    float_decisions = data["float_decisions"]
    fixed_decisions = logit_decisions(logits).reshape(labels.shape)
    float_fidelity = mean_fidelity(float_decisions, labels)
    fixed_fidelity = mean_fidelity(fixed_decisions, labels)
    delta_infidelity = (1.0 - fixed_fidelity) - (1.0 - float_fidelity)
    disagree = fixed_decisions != float_decisions

    return {
        "exposure_s": exposure_s(model_path),
        "method": model_method(model_path),
        "float_fidelity": float_fidelity,
        "fixed_fidelity": fixed_fidelity,
        "delta_infidelity": delta_infidelity,
        "delta_infidelity_pp": 100.0 * delta_infidelity,
        "n_decisions": int(disagree.size),
        "n_disagree": int(disagree.sum()),
        "disagree_rate": float(disagree.mean()),
    }
