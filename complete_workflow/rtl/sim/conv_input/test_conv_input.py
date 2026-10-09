import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge, Timer


ACTIVATION_WIDTH = 9
MAC_LANES = 100
KERNEL_SIDE = 3
CONV_SIDES = (10, 5, 2)
CONV_INPUT_CHANNELS = (1, 4, 8)
CONV_OUTPUT_CHANNELS = (4, 8, 16)


def pack_signed(values, width):
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (value & mask) << (lane * width)
    return packed


def unpack_signed(packed, lane_count, width):
    values = []
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    for lane in range(lane_count):
        value = (packed >> (lane * width)) & mask
        if value & sign_bit:
            value -= 1 << width
        values.append(value)
    return values


def expected_lanes(source_values, conv_index, input_channel, tap_index):
    side = CONV_SIDES[conv_index]
    positions_per_channel = side * side
    output_count = CONV_OUTPUT_CHANNELS[conv_index] * positions_per_channel
    tap_row = tap_index // KERNEL_SIDE - 1
    tap_col = tap_index % KERNEL_SIDE - 1
    expected = []

    for lane in range(MAC_LANES):
        if lane >= output_count:
            expected.append(0)
            continue

        output_position = lane % positions_per_channel
        output_row, output_col = divmod(output_position, side)
        input_row = output_row + tap_row
        input_col = output_col + tap_col

        if not (0 <= input_row < side and 0 <= input_col < side):
            expected.append(0)
            continue

        spatial_index = input_row * side + input_col
        shelf_element = input_channel * positions_per_channel + spatial_index
        expected.append(source_values[shelf_element])

    return expected


def assert_lanes(actual, expected, conv_index, input_channel, tap_index):
    for lane, (actual_value, expected_value) in enumerate(zip(actual, expected)):
        assert actual_value == expected_value, (
            f"conv={conv_index}, input_channel={input_channel}, tap={tap_index}, "
            f"lane={lane}: expected {expected_value}, got {actual_value}"
        )


@cocotb.test
async def test_conv_input_all_layers_channels_and_taps(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    source_values = [lane - 50 for lane in range(MAC_LANES)]
    dut.read_data_flat.value = pack_signed(source_values, ACTIVATION_WIDTH)
    dut.read_valid.value = 0
    dut.input_channel.value = 0
    dut.conv_index.value = 0
    dut.tap_index.value = 0

    dut.rst.value = 1
    await Timer(1, unit="ns")
    await ReadOnly()
    assert dut.conv_input_valid.value == 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0

    last_output = [0] * MAC_LANES
    for conv_index, input_channel_count in enumerate(CONV_INPUT_CHANNELS):
        for input_channel in range(input_channel_count):
            for tap_index in range(KERNEL_SIDE * KERNEL_SIDE):
                await FallingEdge(dut.clk)
                dut.conv_index.value = conv_index
                dut.input_channel.value = input_channel
                dut.tap_index.value = tap_index
                dut.read_valid.value = 1

                await RisingEdge(dut.clk)
                await ReadOnly()
                assert dut.conv_input_valid.value == 1, (
                    f"valid missing for conv={conv_index}, "
                    f"input_channel={input_channel}, tap={tap_index}"
                )

                last_output = unpack_signed(
                    dut.conv_input_flat.value.to_unsigned(),
                    MAC_LANES,
                    ACTIVATION_WIDTH,
                )
                expected = expected_lanes(
                    source_values,
                    conv_index,
                    input_channel,
                    tap_index,
                )
                assert_lanes(
                    last_output,
                    expected,
                    conv_index,
                    input_channel,
                    tap_index,
                )

    await FallingEdge(dut.clk)
    dut.read_valid.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.conv_input_valid.value == 0
    held_output = unpack_signed(
        dut.conv_input_flat.value.to_unsigned(),
        MAC_LANES,
        ACTIVATION_WIDTH,
    )
    assert held_output == last_output

    await FallingEdge(dut.clk)
    dut.rst.value = 1
    await Timer(1, unit="ns")
    await ReadOnly()
    assert dut.conv_input_valid.value == 0
