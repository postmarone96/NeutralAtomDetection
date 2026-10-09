"""Writes src/activation_map_pkg.vhd: kernel taps and output blocks per convolution."""

from pathlib import Path

import config


src_dir = Path(__file__).resolve().parents[1] / "src"
kernel_size = 3
kernel_taps = kernel_size * kernel_size


def vhd_array(name, values):
    body = ", ".join(str(int(value)) for value in values)
    return f"    constant {name} : integer_vector(0 to N_CONV-1) := ({body});"


def output_block_counts(shape):
    mac_units = int(shape["mac_units"])
    block_counts = []

    for layer, side in zip(config.conv_layers, shape["spatial"]):
        out_channels = shape["weights"][layer].shape[0]
        output_count = int(side) ** 2
        output_value_count = int(out_channels) * output_count
        output_block_count = (output_value_count + mac_units - 1) // mac_units

        if output_block_count > 1 and mac_units % output_count != 0:
            raise ValueError(
                f"{layer}: {mac_units} MAC lanes do not end on a spatial "
                f"block boundary of {output_count} values"
            )
        if output_block_count > 1 and output_value_count % mac_units != 0:
            raise ValueError(f"{layer}: the final output block is partial")

        block_counts.append(output_block_count)

    return block_counts


def render(shape):
    block_counts = output_block_counts(shape)
    lines = [
        "library ieee;",
        "use ieee.std_logic_1164.all;",
        "use ieee.numeric_std.all;",
        "",
        "use work.nn_pkg.all;",
        "",
        "package activation_map_pkg is",
        "",
        f"    constant KERNEL_TAP_COUNT : integer := {kernel_taps};",
        vhd_array("CONV_OUTPUT_BLOCK_COUNT", block_counts),
        f"    constant MAX_OUTPUT_BLOCK_COUNT : integer := {max(block_counts)};",
        "",
        "end package activation_map_pkg;",
        "",
    ]
    return "\n".join(lines)


def write(shape=None):
    shape = config.load() if shape is None else shape
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / "activation_map_pkg.vhd"
    path.write_text(render(shape))
    return path


if __name__ == "__main__":
    print(f"wrote {write()}")
