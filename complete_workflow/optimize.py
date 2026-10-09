"""Cheapest per-layer weight widths inside the acceptance band, for one exposure.

The objective is multiplications x bits: under the Resource strategy a layer's
cost scales with how many multiplications it performs, so one bit off conv2 is
worth 28800 bit-multiplies against fc2's 32.
"""

import json

import joblib

from common import *
from model import channels, neighbor_feature_count
from quantize import read_precisions, run_model
from sensitivity import block_activation_integer_bits, layer_blocks, layer_labels
import widths
from widths import (
    make_transform,
    profiled_fractional_bits,
    read_sensitivity,
    seed_widths,
    weight_integer_bits,
)

target_method = None
exposure_s_target = None
best_json = None

max_total_width = 18
descend_step = 1
repair_step = 1
max_evaluations = 120

def multiplication_counts(channels, crop_size):
    c1, c2, c3 = channels
    flat = c3 * (crop_size // 4) ** 2
    return {
        "model_features_0": 9 * c1 * crop_size ** 2,
        "model_features_3": 9 * c1 * c2 * (crop_size // 2) ** 2,
        "model_features_6": 9 * c2 * c3 * (crop_size // 4) ** 2,
        "model_classifier_0": flat * 8,
        "model_classifier_2": (8 + neighbor_feature_count) * 8,
        "model_classifier_4": 8 * 2,
    }


multiplications = multiplication_counts(channels, 10)


def set_exposure(value):
    """Pin this process to one exposure and name its outputs after it."""
    global exposure_s_target, best_json
    exposure_s_target = float(value)
    tag = f"{exposure_s_target:g}s"
    best_json = results_dir / f"best_{tag}.json"


def set_results_dir(path):
    global results_dir
    results_dir = Path(path)


def set_method(method):
    global target_method
    target_method = method


def select_model():
    return model_file(model_dir, exposure_s_target, target_method)


def min_total_widths(precision_rows, integer_bits):
    """The narrowest TOTAL weight width each layer can actually be built at.

    Below it fraction_at has nothing left to resolve with -- either on the
    weight itself, or on the widest activation in the block, which follows the
    weight width through the port ratio.
    """
    block_integers = block_activation_integer_bits(precision_rows)

    floors = {}
    for layer in layer_blocks:
        for width in range(1, max_total_width + 1):
            weight_ok = width - integer_bits[layer] >= 1
            activation_ok = activation_width(width) - block_integers[layer] >= 1
            if weight_ok and activation_ok:
                floors[layer] = width
                break
        else:
            raise ValueError(
                f"{layer}: no weight width up to {max_total_width} leaves a "
                f"fractional bit on an activation with "
                f"{block_integers[layer]} integer bits"
            )
    return floors


def proxy_cost(totals):
    """Bit-multiplies: what the search minimises.

    A multiplier's area scales with the product of its operand widths, not with
    either alone. Below the DSP mapping threshold these are LUT array
    multipliers, where that product is close to exact -- and it is what lets a
    weight bit and an activation bit be traded against each other.
    """
    return sum(
        multiplications[layer] * totals[layer] * activation_width(totals[layer])
        for layer in totals
    )


def bit_value(layer, totals):
    """What taking one bit off this layer's weight is actually worth.

    Measured rather than differentiated: a weight bit drags the activation width
    down with it through the port ratio, so the saving is not multiplications
    alone.
    """
    trial = dict(totals)
    trial[layer] -= 1
    return proxy_cost(totals) - proxy_cost(trial)


class Evaluator:
    """Measures delta infidelity for a set of TOTAL widths, and remembers."""

    def __init__(self, model_path, integer_bits):
        self.model_path = model_path
        self.integer_bits = integer_bits
        self.precision_rows = read_precisions(
            exposure_s(model_path), model_method(model_path),
        )
        self.seen = {}
        self.trace = []

    def key(self, totals):
        return tuple(totals[layer] for layer in layer_blocks)

    def measure(self, totals, phase):
        cached = self.seen.get(self.key(totals))
        if cached is not None:
            return cached

        if len(self.trace) >= max_evaluations:
            raise RuntimeError(
                f"hit max_evaluations = {max_evaluations} before converging; "
                "raise it or widen descend_step"
            )

        fractional = {
            layer: totals[layer] - self.integer_bits[layer] for layer in totals
        }
        transform = make_transform(fractional, self.integer_bits)
        metrics = run_model(
            self.model_path,
            precision_transform=transform,
        )

        delta = float(metrics["delta_infidelity_pp"])
        record = {
            "evaluation": len(self.trace) + 1,
            "phase": phase,
            **{f"W_{layer}": totals[layer] for layer in layer_blocks},
            "proxy_cost": proxy_cost(totals),
            "total_weight_bits": sum(totals.values()),
            "delta_infidelity_pp": delta,
            "n_disagree": int(metrics["n_disagree"]),
            "n_decisions": int(metrics["n_decisions"]),
            "feasible": abs(delta) <= widths.acceptance_pp,
        }
        self.seen[self.key(totals)] = record
        self.trace.append(record)

        print(
            f"  [{record['evaluation']:>3}] {phase:<7} "
            + " ".join(f"{totals[l]:>2}" for l in layer_blocks)
            + f"   dI={delta:+8.4f} pp  cost={record['proxy_cost']:>11,}  "
            + ("inside" if record["feasible"] else "OUTSIDE")
        )
        return record


def starting_widths(precision_rows, integer_bits, min_widths):
    """The per-layer sensitivity minima, as TOTAL widths."""
    fractional, _integer, _ceiling = seed_widths(
        precision_rows, exposure_s_target, target_method, widths.acceptance_pp
    )
    totals = {}
    for layer in layer_blocks:
        total = fractional[layer] + integer_bits[layer]
        if total < min_widths[layer]:
            total = min_widths[layer]
        totals[layer] = total
    return totals


def repair(evaluator, totals, precision_rows, integer_bits):
    """Widen until the combination is inside the band.

    Which layer to widen comes from the sensitivity table: the one whose own
    curve was worst at the width it currently sits on is the one most likely to
    be carrying the combined error.
    """
    sensitivity = read_sensitivity(exposure_s_target, target_method)

    def implication(layer):
        measured = sensitivity[layer]
        nearest = min(measured, key=lambda bits: abs(bits - totals[layer]))
        return abs(measured[nearest])

    record = evaluator.measure(totals, "repair")
    while not record["feasible"]:
        candidates = [
            layer for layer in layer_blocks if totals[layer] < max_total_width
        ]
        if not candidates:
            raise RuntimeError(
                "every layer is at max_total_width and the band is still missed; "
                "the profile itself does not pass, which is a profiling problem"
            )
        layer = max(candidates, key=implication)
        totals[layer] = min(max_total_width, totals[layer] + repair_step)
        print(f"        widening {layer_labels.get(layer, layer)} -> {totals[layer]}")
        record = evaluator.measure(totals, "repair")
    return totals, record


def descend(evaluator, totals, precision_rows, integer_bits):
    """Take bits off the most expensive layer that still passes.

    Layers are tried in descending bit value, so the first accepted reduction in
    each round is the most valuable one available. When no single layer can come
    down without leaving the band, this is a local minimum for one-bit moves.

    """
    best = evaluator.seen[evaluator.key(totals)]
    floors = min_total_widths(precision_rows, integer_bits)
    improved = True
    while improved:
        improved = False
        order = sorted(
            layer_blocks, key=lambda layer: bit_value(layer, totals),
            reverse=True,
        )
        for layer in order:
            if totals[layer] - descend_step < floors[layer]:
                continue
            trial = dict(totals)
            trial[layer] -= descend_step
            record = evaluator.measure(trial, "descend")
            if record["feasible"]:
                totals, best = trial, record
                improved = True
                print(
                    f"        accepted -{descend_step} on "
                    f"{layer_labels.get(layer, layer)} -> {totals[layer]}"
                )
                break
    return totals, best


def run(exposure):
    results_dir.mkdir(parents=True, exist_ok=True)
    set_exposure(exposure)

    model_path = select_model()
    job = joblib.load(model_path)
    global multiplications
    multiplications = multiplication_counts(
        tuple(job["channels"]) if "channels" in job else channels, job["crop_size"]
    )
    precision_rows = read_precisions(
        exposure_s_target, target_method
    )
    integer_bits = weight_integer_bits(precision_rows)
    fractional = profiled_fractional_bits(precision_rows)
    profiled = {
        layer: fractional[layer] + integer_bits[layer] for layer in layer_blocks
    }

    print(f"  {model_path.name}")
    print(f"  band +/-{widths.acceptance_pp} pp   upper bound {max_total_width} bits")

    evaluator = Evaluator(model_path, integer_bits)
    totals = starting_widths(
        precision_rows, integer_bits,
        min_total_widths(precision_rows, integer_bits),
    )
    totals, _ = repair(evaluator, totals, precision_rows, integer_bits)
    totals, best = descend(evaluator, totals, precision_rows, integer_bits)

    payload = {"total_widths": {layer: totals[layer] for layer in layer_blocks}}
    with best_json.open("w") as handle:
        json.dump(payload, handle, indent=2)

    saving = 100.0 * (1.0 - best["proxy_cost"] / proxy_cost(profiled))
    print(f"  widths " + " ".join(f"{totals[l]}" for l in layer_blocks)
          + f"   dI={best['delta_infidelity_pp']:+.4f} pp"
          + f"   cost {best['proxy_cost']:,} ({saving:.1f}% under baseline)")
    print(f"  -> {best_json}")
    return payload
