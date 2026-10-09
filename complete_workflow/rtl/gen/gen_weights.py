"""Writes packed weight ROM data, scalar debug data, biases, and metadata.

Weights keep their own 5-7 bit format and are sign-extended to 8-bit values.
Sixteen values fit in each 128-bit ROM word, with the first value in bits
7 downto 0. ``weights_scalar.mem`` keeps the original one-byte-per-line form
for debugging and generator verification.

Every ROM word contains all weights needed for one MAC cycle. A convolution
word is selected by output block, input channel, and kernel tap. A dense word
is selected by input feature.

Biases use the same block-oriented idea. One 512-bit word contains sixteen
signed 32-bit bias slots: one word per convolution output block and one word
per dense layer. Unused slots are zero.
"""

from pathlib import Path

import numpy as np

import config
from fixed_point import fractional_bits, quantise, rescale, twos_complement

src_dir = Path(__file__).resolve().parents[1] / "src"

weight_bits = 8
weight_slots_per_word = 16
weight_rom_bits = weight_bits * weight_slots_per_word
bias_bits = 32
bias_slots_per_word = 16
bias_rom_bits = bias_bits * bias_slots_per_word


def flatten(shape, layer):
    """Weights in ROM order, as raw fixed-point codes."""
    precision = shape["precision"][(layer, "weight")]
    return quantise(shape["weights"][layer].reshape(-1), precision)


def flatten_bias(shape, layer):
    bias_precision = shape["precision"][(layer, "bias")]
    accum_precision = shape["precision"][(layer, "accum")]
    codes = quantise(shape["biases"][layer].reshape(-1), bias_precision)
    return rescale(
        codes,
        fractional_bits(bias_precision),
        fractional_bits(accum_precision),
    )


def layout(shape):
    """Base address and count per layer, for weights and biases."""
    table = {}
    weight_base = 0
    bias_base = 0
    for layer in config.layer_order:
        weights = flatten(shape, layer)
        biases = flatten_bias(shape, layer)
        tensor = shape["weights"][layer]
        output_count = int(tensor.shape[0])
        input_count = int(tensor.shape[1])
        if layer in config.conv_layers:
            conv_index = config.conv_layers.index(layer)
            positions_per_channel = int(shape["spatial"][conv_index] ** 2)
            channels_per_block = int(shape["mac_units"] // positions_per_channel)
            output_block_count = (
                output_count + channels_per_block - 1
            ) // channels_per_block
            kernel_tap_count = int(tensor.shape[2] * tensor.shape[3])
            word_count = output_block_count * input_count * kernel_tap_count
            bias_word_count = output_block_count
        else:
            word_count = input_count
            bias_word_count = 1
        table[layer] = {
            "weight_base": weight_base,
            "weight_count": int(weights.size),
            "weight_word_count": word_count,
            "layer_input_count": input_count,
            "bias_base": bias_base,
            "bias_count": int(biases.size),
            "bias_word_count": bias_word_count,
        }
        weight_base += weights.size
        bias_base += biases.size
    return table


def write_mem(path, codes, bits):
    digits = (bits + 3) // 4
    packed = twos_complement(codes, bits)
    path.write_text("".join(f"{int(value):0{digits}x}\n" for value in packed))


def weight_word_groups(shape, layer):
    """Return the values belonging in each ROM word for one layer."""
    precision = shape["precision"][(layer, "weight")]
    codes = quantise(shape["weights"][layer], precision)
    if layer in config.conv_layers:
        conv_index = config.conv_layers.index(layer)
        positions_per_channel = int(shape["spatial"][conv_index] ** 2)
        channels_per_block = int(shape["mac_units"] // positions_per_channel)
        output_count = int(codes.shape[0])
        groups = []
        for first_output in range(0, output_count, channels_per_block):
            last_output = min(first_output + channels_per_block, output_count)
            for input_channel in range(codes.shape[1]):
                for kernel_y in range(codes.shape[2]):
                    for kernel_x in range(codes.shape[3]):
                        groups.append(
                            codes[
                                first_output:last_output,
                                input_channel,
                                kernel_y,
                                kernel_x,
                            ]
                        )
        return groups
    return codes.transpose(1, 0)


def pack_weight_words(word_groups):
    """Pack little-lane groups into 128-bit ROM words."""
    words = []
    mask = (1 << weight_bits) - 1
    for codes in word_groups:
        if len(codes) > weight_slots_per_word:
            raise ValueError(
                f"a weight word needs {len(codes)} slots; "
                f"only {weight_slots_per_word} are available"
            )
        unsigned = twos_complement(codes, weight_bits)
        word = 0
        for lane, value in enumerate(unsigned):
            word |= (int(value) & mask) << (lane * weight_bits)
        words.append(word)
    return words


def bias_word_groups(shape, layer):
    """Return the biases belonging in each output block word."""
    codes = flatten_bias(shape, layer)
    if layer in config.conv_layers:
        conv_index = config.conv_layers.index(layer)
        positions_per_channel = int(shape["spatial"][conv_index] ** 2)
        channels_per_block = int(shape["mac_units"] // positions_per_channel)
        return [
            codes[start:start + channels_per_block]
            for start in range(0, len(codes), channels_per_block)
        ]
    return [codes]


def pack_bias_words(word_groups):
    """Pack little-lane bias groups into 512-bit ROM words."""
    words = []
    mask = (1 << bias_bits) - 1
    for codes in word_groups:
        if len(codes) > bias_slots_per_word:
            raise ValueError(
                f"a bias word needs {len(codes)} slots; "
                f"only {bias_slots_per_word} are available"
            )
        unsigned = twos_complement(codes, bias_bits)
        word = 0
        for lane, value in enumerate(unsigned):
            word |= (int(value) & mask) << (lane * bias_bits)
        words.append(word)
    return words


def write_packed_weights(path, words):
    digits = (weight_rom_bits + 3) // 4
    path.write_text("".join(f"{word:0{digits}x}\n" for word in words))


def write_packed_biases(path, words):
    digits = (bias_rom_bits + 3) // 4
    path.write_text("".join(f"{word:0{digits}x}\n" for word in words))


def _array_constant(table, name):
    body = ", ".join(str(table[layer][name]) for layer in config.layer_order)
    return f"    constant {name.upper()} : integer_vector(0 to {len(table)}-1) := ({body});"


def render_weights_package(table):
    totals_w_words = sum(entry["weight_word_count"] for entry in table.values())
    max_layer_inputs = max(entry["layer_input_count"] for entry in table.values())
    lines = [
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "use ieee.numeric_std.all;",
        "",
        "package weights_pkg is",
        "",
        f"    constant WEIGHT_BITS      : integer := {weight_bits};",
        f"    constant WEIGHT_SLOTS_PER_WORD : integer := {weight_slots_per_word};",
        f"    constant WEIGHT_ROM_BITS  : integer := {weight_rom_bits};",
        f"    constant WEIGHT_ROM_DEPTH : integer := {totals_w_words};",
        f"    constant MAX_LAYER_INPUT_COUNT : integer := {max_layer_inputs};",
        "",
        _array_constant(table, "layer_input_count"),
        "",
        "end package weights_pkg;",
        "",
    ]
    return "\n".join(lines)


def render_bias_package(table):
    totals_b_words = sum(entry["bias_word_count"] for entry in table.values())
    lines = [
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "use ieee.numeric_std.all;",
        "",
        "package bias_pkg is",
        "",
        f"    constant BIAS_BITS        : integer := {bias_bits};",
        f"    constant BIAS_ROM_BITS    : integer := {bias_rom_bits};",
        f"    constant BIAS_ROM_DEPTH   : integer := {totals_b_words};",
        "",
        "end package bias_pkg;",
        "",
    ]
    return "\n".join(lines)


def write(shape=None):
    shape = config.load() if shape is None else shape
    src_dir.mkdir(parents=True, exist_ok=True)
    table = layout(shape)

    layer_weights = [flatten(shape, layer) for layer in config.layer_order]
    weights = np.concatenate(layer_weights)
    biases = np.concatenate([flatten_bias(shape, l) for l in config.layer_order])
    word_groups = [
        group
        for layer in config.layer_order
        for group in weight_word_groups(shape, layer)
    ]
    packed_weights = pack_weight_words(word_groups)
    bias_groups = [
        group
        for layer in config.layer_order
        for group in bias_word_groups(shape, layer)
    ]
    packed_biases = pack_bias_words(bias_groups)

    write_packed_weights(src_dir / "weights.mem", packed_weights)
    write_mem(src_dir / "weights_scalar.mem", weights, weight_bits)
    write_packed_biases(src_dir / "biases.mem", packed_biases)
    write_mem(src_dir / "biases_scalar.mem", biases, bias_bits)
    (src_dir / "weights_pkg.vhd").write_text(render_weights_package(table))
    (src_dir / "bias_pkg.vhd").write_text(render_bias_package(table))
    return table, weights, biases


if __name__ == "__main__":
    table, weights, biases = write()
    for layer, entry in table.items():
        print(f"{layer:20} weights {entry['weight_count']:5} @ {entry['weight_base']:5}"
              f"   biases {entry['bias_count']:3} @ {entry['bias_base']:3}")
    print(f"\n{weights.size} weights, {biases.size} biases")
    packed_bits = sum(entry["weight_word_count"] for entry in table.values()) * weight_rom_bits
    packed_bias_bits = sum(entry["bias_word_count"] for entry in table.values()) * bias_rom_bits
    print(f"ROM {packed_bits / 1024:.1f} kbit "
          f"+ {packed_bias_bits / 1024:.1f} kbit")
