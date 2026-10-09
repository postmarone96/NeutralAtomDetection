"""Writes src/nn_pkg.vhd: the dimensions and fixed-point formats the RTL uses, as VHDL-2008 constants."""

from pathlib import Path

import numpy as np

import config
from fixed_point import fractional_bits, precision_parts
from gen_weights import flatten, flatten_bias

src_dir = Path(__file__).resolve().parents[1] / "src"

def check_accumulators_cannot_overflow(shape):
    for layer, stage in zip(config.layer_order, config.layer_input_stages):
        input_precision = shape["precision"][(stage, "result")]
        weight_precision = shape["precision"][(layer, "weight")]
        accum_precision = shape["precision"][(layer, "accum")]
        accum_w, _ = precision_parts(accum_precision)
        input_w, _ = precision_parts(input_precision)
        shift = (
            fractional_bits(accum_precision)
            - fractional_bits(input_precision)
            - fractional_bits(weight_precision)
        )
        weights = flatten(shape, layer).reshape(shape["weights"][layer].shape[0], -1)
        bias = flatten_bias(shape, layer)
        extremes = np.array([-(1 << (input_w - 1)), (1 << (input_w - 1)) - 1])
        products = weights[:, :, None] * extremes * (1 << shift)
        high = products.max(axis=2).sum(axis=1) + bias
        low = products.min(axis=2).sum(axis=1) + bias
        if high.max() >= 1 << (accum_w - 1) or low.min() < -(1 << (accum_w - 1)):
            raise ValueError(f"{layer}: worst-case sum does not fit ACC_W = {accum_w}")


def vhd_int_array(name, values):
    body = ", ".join(str(int(value)) for value in values)
    return f"    constant {name} : integer_vector(0 to N_LAYERS-1) := ({body});"


def render(shape):
    c1, c2, c3 = shape["channels"]
    crop = shape["crop"]
    mac_units = shape["mac_units"]
    weight = [precision_parts(shape["precision"][(l, "weight")]) for l in config.layer_order]
    result = [precision_parts(shape["precision"][(l, "result")]) for l in config.layer_order]
    accum = [precision_parts(shape["precision"][(l, "accum")]) for l in config.layer_order]

    in_w, in_i = precision_parts(shape["precision"][("__input__", "result")])
    nb_w, nb_i = precision_parts(shape["precision"][("__input___neighbor", "result")])
    cat_w, cat_i = precision_parts(shape["precision"][("cat", "result")])
    raw_w, _ = precision_parts(shape["precision"][("__raw__", "integer")])

    lines = [
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "use ieee.numeric_std.all;",
        "",
        "package nn_pkg is",
        "",
        f"    constant CROP       : integer := {shape['crop']};",
        f"    constant MAX_CHANNEL : integer := {max(c1, c2, c3)};",
        f"    constant NEIGHBORS  : integer := {shape['neighbors']};",
        f"    constant CLASSES    : integer := {shape['classes']};",
        f"    constant MAX_MAC_UNITS : integer := {shape['mac_units']};",
        "",
        f"    constant RAW_W     : integer := {raw_w};",
        f"    constant IN_W      : integer := {in_w};",
        f"    constant IN_I      : integer := {in_i};",
        f"    constant NB_W      : integer := {nb_w};",
        f"    constant NB_I      : integer := {nb_i};",
        f"    constant CAT_W     : integer := {cat_w};",
        f"    constant CAT_I     : integer := {cat_i};",
        "",
        f"    constant N_LAYERS  : integer := {len(config.layer_order)};",
        f"    constant N_CONV    : integer := {len(config.conv_layers)};",
        "",
        vhd_int_array("ACT_W", [w for w, _ in result]),
        vhd_int_array("ACT_I", [i for _, i in result]),
        vhd_int_array("ACC_W", [w for w, _ in accum]),
        vhd_int_array("ACC_I", [i for _, i in accum]),
        "",
        f"    constant WEIGHT_W_MAX   : integer := {max(w for w, _ in weight)};",
        f"    constant ACC_W_MAX      : integer := {max(w for w, _ in accum)};",
        f"    constant ACT_W_MAX      : integer := {max(w for w, _ in result)};",
        "",
        "end package nn_pkg;",
        "",
    ]
    return "\n".join(lines)


def write(shape=None):
    shape = config.load() if shape is None else shape
    check_accumulators_cannot_overflow(shape)
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / "nn_pkg.vhd"
    path.write_text(render(shape))
    return path


if __name__ == "__main__":
    print(f"wrote {write()}")
