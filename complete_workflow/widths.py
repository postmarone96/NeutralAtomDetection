"""Per-layer widths for one build: seeded from the sensitivity sweep, then manual."""

import csv

from common import *
from quantize import read_precisions
import sensitivity
from sensitivity import layer_blocks

acceptance_pp = 0.05

def weight_integer_bits(precision_rows):
    """Integer bits of each layer's weight tensor, from the profiled ranges."""
    return {
        row["layer"]: integer_bits_of(row["used_precision"])
        for row in precision_rows
        if row["tensor"] == "weight" and row["layer"] in layer_blocks
    }


def profiled_fractional_bits(precision_rows):
    """Fractional bits each layer's weight tensor already carries.
    """
    widths = {}
    for row in precision_rows:
        if row["tensor"] != "weight" or row["layer"] not in layer_blocks:
            continue
        precision = row["used_precision"]
        widths[row["layer"]] = width_of(precision) - integer_bits_of(precision)
    return widths


def read_sensitivity(exposure, method):
    """{layer: {total_weight_bits: delta_infidelity_pp}} for one exposure."""
    table = {}
    with open(sensitivity.metrics_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != method:
                continue
            if abs(float(row["exposure_s"]) - exposure) > 1e-9:
                continue
            layer = row["swept_layer"]
            table.setdefault(layer, {})[int(row["weight_bits"])] = float(
                row["delta_infidelity_pp"]
            )
    return table


def seed_widths(precision_rows, exposure, method, acceptance_pp):
    """Narrowest per-layer fraction that was individually inside the band."""
    integer_bits = weight_integer_bits(precision_rows)
    sensitivity = read_sensitivity(exposure, method)
    ceiling = profiled_fractional_bits(precision_rows)

    widths = {}
    for layer in layer_blocks:
        integer = integer_bits[layer]
        measured = sensitivity[layer]
        passing = sorted(
            total for total, delta in measured.items()
            if abs(delta) <= acceptance_pp
        )
        if passing:
            widths[layer] = passing[0] - integer
        else:
            widths[layer] = ceiling[layer]

    return widths, integer_bits, ceiling


def make_transform(widths, integer_bits):
    block_of = {
        member: layer
        for layer, members in layer_blocks.items()
        for member in members
    }

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
            owner = block_of.get(row["layer"])
            if owner is not None:
                total = int(widths[owner]) + integer_bits[owner]
                activation_fraction[row["layer"]] = activation_width(total) - integer
            else:
                activation_fraction[row["layer"]] = (
                    width_of(row["used_precision"]) - integer
                )

        accum_fraction = {
            row["layer"]: accum_fraction_at(
                row,
                weight_precision,
                int(widths[row["layer"]]) + integer_bits[row["layer"]],
                activation_fraction,
            )
            for row in rows
            if row["tensor"] == "accum" and row["layer"] in widths
        }

        out = []
        for row in rows:
            new_row = dict(row)
            layer = row["layer"]
            tensor = row["tensor"]
            owner = block_of.get(layer)

            if tensor == "weight" and layer in widths:
                integer = integer_bits_of(row["used_precision"])
                fractional = int(widths[layer])
                new_row["used_precision"] = (
                    f"fixed<{integer + fractional},{integer},RND,SAT>"
                )
            elif tensor == "result" and owner is not None:

                total = int(widths[owner]) + integer_bits[owner]
                integer = integer_bits_of(row["used_precision"])
                fractional = activation_width(total) - integer
                new_row["used_precision"] = (
                    f"fixed<{integer + fractional},{integer},RND,SAT>"
                )
            elif tensor in ("accum", "bias") and layer in widths:
                integer = integer_bits_of(row["used_precision"])
                fractional = accum_fraction[layer]
                new_row["used_precision"] = (
                    f"fixed<{integer + fractional},{integer},RND,SAT>"
                )

            out.append(new_row)
        return out

    return transform


def resolve_widths(model_path, total_widths, acceptance_pp):
    exposure = exposure_s(model_path)
    entry = {float(key): value for key, value in total_widths.items()}[exposure]

    missing = set(layer_blocks) - set(entry)
    unknown = set(entry) - set(layer_blocks)
    if missing or unknown:
        raise KeyError(
            f"widths at {1000.0 * exposure:g} ms name {sorted(unknown)} and miss "
            f"{sorted(missing)}; expected exactly {sorted(layer_blocks)}"
        )

    precision_rows = read_precisions(exposure, model_method(model_path))
    integer_bits = weight_integer_bits(precision_rows)
    return {
        "widths": {
            layer: int(total) - integer_bits[layer] for layer, total in entry.items()
        },
        "integer_bits": integer_bits,
        "precision_rows": precision_rows,
    }
