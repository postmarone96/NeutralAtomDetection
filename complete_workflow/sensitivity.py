"""Per-layer weight-width sensitivity: one layer swept, the rest left wide."""

import csv
from pathlib import Path

from common import (
    model_dir,
    results_dir,
    metrics_csv,
    accum_fraction_at,
    activation_width,
    integer_bits_of,
    model_file,
    width_of,
)
import quantize
from quantize import run_model


selected_exposures = None
weight_widths = []
resume = None

layer_blocks = {
    "model_features_0": ("__input__", "model_features_0", "model_features_1", "model_features_2"),
    "model_features_3": ("model_features_3", "model_features_4", "model_features_5"),
    "model_features_6": ("model_features_6", "model_features_7", "model_features_8"),
    "model_classifier_0": ("model_classifier_0", "model_classifier_1"),
    "model_classifier_2": ("__input___neighbor", "cat", "model_classifier_2", "model_classifier_3"),
    "model_classifier_4": ("model_classifier_4",),
}

layer_labels = {
    "model_features_0": "Input + Conv1 + ReLu1 + Maxpool1",
    "model_features_3": "Conv2 + ReLu2 + Maxpool2",
    "model_features_6": "Conv3 + ReLu3 + flatten",
    "model_classifier_0": "Fc1 + ReLu4",
    "model_classifier_2": "Neighbors + concat + Fc2 + ReLu5",
    "model_classifier_4": "Fc3 -> Logits",
}


def set_results_dir(path):
    global results_dir, metrics_csv
    results_dir = Path(path)
    metrics_csv = results_dir / "sensitivity.csv"


def make_transform(swept_layer, weight_bits):
    """Only the swept layer moves; every other row keeps its profiled value."""
    block = layer_blocks[swept_layer]
    width = activation_width(weight_bits)

    def transform(rows):
        weight_precision = {
            row["layer"]: row["used_precision"]
            for row in rows
            if row["tensor"] == "weight"
        }

        activation_fraction = {}
        for row in rows:
            if row["tensor"] != "result":
                continue
            integer = integer_bits_of(row["used_precision"])
            if row["layer"] in block:
                activation_fraction[row["layer"]] = width - integer
            else:
                activation_fraction[row["layer"]] = (
                    width_of(row["used_precision"]) - integer
                )

        accum_fraction = {
            row["layer"]: accum_fraction_at(
                row, weight_precision, weight_bits, activation_fraction
            )
            for row in rows
            if row["tensor"] == "accum" and row["layer"] == swept_layer
        }

        out = []
        for row in rows:
            new_row = dict(row)
            layer = row["layer"]
            tensor = row["tensor"]

            if layer == swept_layer and tensor == "weight":
                integer = integer_bits_of(row["used_precision"])
                fractional = weight_bits - integer
                new_row["used_precision"] = (
                    f"fixed<{integer + fractional},{integer},RND,SAT>"
                )
            elif tensor == "result" and layer in block:
                integer = integer_bits_of(row["used_precision"])
                fractional = width - integer
                new_row["used_precision"] = (
                    f"fixed<{integer + fractional},{integer},RND,SAT>"
                )
            elif tensor in ("accum", "bias") and layer == swept_layer:
                integer = integer_bits_of(row["used_precision"])
                fractional = accum_fraction[layer]
                new_row["used_precision"] = (
                    f"fixed<{integer + fractional},{integer},RND,SAT>"
                )

            out.append(new_row)
        return out

    return transform


def block_activation_integer_bits(precision_rows):
    """The widest activation integer part in each block."""
    integers = {
        row["layer"]: integer_bits_of(row["used_precision"])
        for row in precision_rows
        if row["tensor"] == "result"
    }
    return {
        layer: max(integers[member] for member in members if member in integers)
        for layer, members in layer_blocks.items()
    }


def select_records():
    """The one model per selected exposure, in exposure order."""
    return [
        {
            "exposure_s": exposure,
            "path": model_file(model_dir, exposure, quantize.selected_method),
        }
        for exposure in sorted(selected_exposures)
    ]


def load_completed():
    if not (resume and metrics_csv.exists()):
        return [], set()

    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key in ("exposure_s", "float_fidelity", "fixed_fidelity",
                    "delta_infidelity", "delta_infidelity_pp", "disagree_rate"):
            row[key] = float(row[key])
        for key in ("weight_bits", "n_decisions", "n_disagree"):
            row[key] = int(row[key])

    done = {
        (row["swept_layer"], row["weight_bits"], row["exposure_s"]) for row in rows
    }
    return rows, done


def save_csv(metric_rows):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)


def run():
    results_dir.mkdir(parents=True, exist_ok=True)
    records = select_records()
    metric_rows, done = load_completed()
    total = len(layer_blocks) * len(weight_widths) * len(records)

    print(f"  {len(layer_blocks)} layers x {len(weight_widths)} widths "
          f"x {len(records)} exposures = {total} evaluations")
    if done:
        print(f"  resuming: {len(done)} of {total} already measured")

    for swept_layer in layer_blocks:
        print(f"  === {swept_layer}  ({layer_labels.get(swept_layer, '')})")
        for record in records:
            model_path = record["path"]
            exposure = record["exposure_s"]
            precision_rows = quantize.read_precisions(
                exposure, quantize.selected_method
            )
            profiled_integer = {
                row["layer"]: integer_bits_of(row["used_precision"])
                for row in precision_rows
                if row["tensor"] == "weight"
            }
            block_integer = block_activation_integer_bits(precision_rows)
            for weight_bits in weight_widths:
                if (swept_layer, weight_bits, exposure) in done:
                    continue
                integer = profiled_integer[swept_layer]
                if weight_bits <= integer:
                    print(
                        f"    {1000.0 * exposure:>5.0f} ms  w={weight_bits:<3} "
                        f"not measurable: {integer} weight integer bits"
                    )
                    continue
                activation_integer = block_integer[swept_layer]
                if activation_width(weight_bits) <= activation_integer:
                    print(
                        f"    {1000.0 * exposure:>5.0f} ms  w={weight_bits:<3} "
                        f"not measurable: {activation_width(weight_bits)}-bit activation "
                        f"against {activation_integer} integer bits"
                    )
                    continue

                metrics = run_model(
                    model_path,
                    precision_transform=make_transform(swept_layer, weight_bits),
                )
                metrics["swept_layer"] = swept_layer
                metrics["weight_bits"] = weight_bits
                metric_rows.append(metrics)

                print(
                    f"    {1000.0 * exposure:>5.0f} ms  w={weight_bits:<3} "
                    f"dI={metrics['delta_infidelity_pp']:+8.4f} pp  "
                    f"disagree={metrics['n_disagree']:>5}/{metrics['n_decisions']}"
                )
                save_csv(metric_rows)

    save_csv(metric_rows)
    print(f"  {len(metric_rows)} rows -> {metrics_csv}")
    return metrics_csv
