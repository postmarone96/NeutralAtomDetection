from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


MAX_MAC_UNITS = 100
WEIGHT_WIDTH = 6
KERNEL_TAPS = 9
POSITIONS = (100, 25, 4)
INPUT_COUNTS = (1, 4, 8, 64, 16, 8)
OUTPUT_COUNTS = (4, 8, 16, 8, 8, 2)
WEIGHT_BASES = (0, 36, 324, 1476, 1988, 2116)


def signed_value(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def unpack_lanes(packed):
    value = int(packed)
    mask = (1 << WEIGHT_WIDTH) - 1
    return [
        signed_value((value >> (lane * WEIGHT_WIDTH)) & mask, WEIGHT_WIDTH)
        for lane in range(MAX_MAC_UNITS)
    ]


async def read_weights(dut, layer, input_index, output_block=0, tap=0):
    await FallingEdge(dut.clk)
    dut.layer_index.value = layer
    dut.input_index.value = input_index
    dut.output_block_index.value = output_block
    dut.tap_index.value = tap
    dut.read_enable.value = 1

    await RisingEdge(dut.clk)
    dut.read_enable.value = 0
    await ReadOnly()

    assert int(dut.weights_valid.value) == 1
    return unpack_lanes(dut.mac_weights_flat.value)


@cocotb.test()
async def checks_every_weight_row_and_mac_lane(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))

    scalar_path = Path(__file__).resolve().parents[2] / "src" / "weights_scalar.mem"
    scalar_weights = [
        signed_value(int(line, 16), 8)
        for line in scalar_path.read_text().splitlines()
    ]

    dut.rst.value = 1
    dut.read_enable.value = 0
    dut.layer_index.value = 0
    dut.output_block_index.value = 0
    dut.input_index.value = 0
    dut.tap_index.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0

    # Every convolution row: block, input channel, and tap.
    for layer in range(3):
        positions = POSITIONS[layer]
        channels_per_block = MAX_MAC_UNITS // positions
        block_count = (
            OUTPUT_COUNTS[layer] + channels_per_block - 1
        ) // channels_per_block

        for output_block in range(block_count):
            for input_channel in range(INPUT_COUNTS[layer]):
                for tap in range(KERNEL_TAPS):
                    actual = await read_weights(
                        dut,
                        layer=layer,
                        input_index=input_channel,
                        output_block=output_block,
                        tap=tap,
                    )

                    expected = []
                    for lane in range(MAX_MAC_UNITS):
                        work_index = output_block * MAX_MAC_UNITS + lane
                        if work_index >= OUTPUT_COUNTS[layer] * positions:
                            expected.append(0)
                            continue

                        output_channel = work_index // positions
                        scalar_index = (
                            WEIGHT_BASES[layer]
                            + output_channel * INPUT_COUNTS[layer] * KERNEL_TAPS
                            + input_channel * KERNEL_TAPS
                            + tap
                        )
                        expected.append(scalar_weights[scalar_index])

                    assert actual == expected, (
                        f"conv layer={layer}, block={output_block}, "
                        f"input={input_channel}, tap={tap}"
                    )

    # Every dense row: one row per input feature.
    for layer in range(3, 6):
        for input_feature in range(INPUT_COUNTS[layer]):
            actual = await read_weights(
                dut,
                layer=layer,
                input_index=input_feature,
            )

            expected = [0] * MAX_MAC_UNITS
            for output_neuron in range(OUTPUT_COUNTS[layer]):
                scalar_index = (
                    WEIGHT_BASES[layer]
                    + output_neuron * INPUT_COUNTS[layer]
                    + input_feature
                )
                expected[output_neuron] = scalar_weights[scalar_index]

            assert actual == expected, (
                f"dense layer={layer}, input={input_feature}"
            )

    # Valid must drop one clock after read_enable drops.
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.weights_valid.value) == 0
    assert unpack_lanes(dut.mac_weights_flat.value) == [0] * MAX_MAC_UNITS
