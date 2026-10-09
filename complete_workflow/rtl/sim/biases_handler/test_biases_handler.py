from pathlib import Path

import cocotb
from cocotb.triggers import Timer


MAX_MAC_UNITS = 100
ACC_WIDTH = 17
POSITIONS_PER_CHANNEL = (100, 25, 4)
OUTPUT_COUNTS = (4, 8, 16, 8, 8, 2)
OUTPUT_BLOCK_COUNTS = (4, 2, 1)
BIAS_BASES = (0, 4, 12, 28, 36, 44)
BIAS_WORD_BASES = (0, 4, 6, 7, 8, 9)


def to_signed(value, width):
    value &= (1 << width) - 1
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value


def unpack_lanes(packed):
    value = int(packed)
    mask = (1 << ACC_WIDTH) - 1
    return [
        to_signed((value >> (lane * ACC_WIDTH)) & mask, ACC_WIDTH)
        for lane in range(MAX_MAC_UNITS)
    ]


@cocotb.test()
async def routes_every_bias_and_zeros_unused_lanes(dut):
    scalar_path = Path(__file__).resolve().parents[2] / "src" / "biases_scalar.mem"
    scalar_biases = [
        to_signed(int(line, 16), 32)
        for line in scalar_path.read_text().splitlines()
        if line.strip()
    ]

    # Convolution layers: one row per output block and broadcast per channel.
    dut.conv_enable.value = 1
    for layer in range(3):
        positions = POSITIONS_PER_CHANNEL[layer]
        total_outputs = OUTPUT_COUNTS[layer] * positions

        for output_block in range(OUTPUT_BLOCK_COUNTS[layer]):
            dut.layer_index.value = layer
            dut.output_block_index.value = output_block
            await Timer(1, unit="ns")

            assert int(dut.bias_row.value) == BIAS_WORD_BASES[layer] + output_block

            expected = []
            for lane in range(MAX_MAC_UNITS):
                work_index = output_block * MAX_MAC_UNITS + lane
                if work_index >= total_outputs:
                    expected.append(0)
                else:
                    output_channel = work_index // positions
                    expected.append(
                        to_signed(
                            scalar_biases[BIAS_BASES[layer] + output_channel],
                            ACC_WIDTH,
                        )
                    )

            assert unpack_lanes(dut.bias_out_flat.value) == expected

    # Dense layers: block index is ignored and one bias is routed per lane.
    dut.conv_enable.value = 0
    dut.output_block_index.value = 3
    for layer in range(3, 6):
        dut.layer_index.value = layer
        await Timer(1, unit="ns")

        assert int(dut.bias_row.value) == BIAS_WORD_BASES[layer]

        expected = [0] * MAX_MAC_UNITS
        for lane in range(OUTPUT_COUNTS[layer]):
            expected[lane] = to_signed(
                scalar_biases[BIAS_BASES[layer] + lane],
                ACC_WIDTH,
            )

        assert unpack_lanes(dut.bias_out_flat.value) == expected
