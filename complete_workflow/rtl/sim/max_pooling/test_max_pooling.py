from itertools import islice, product

import cocotb
from cocotb.triggers import Timer


MAX_MAC_UNITS = 100
ACT_WIDTH = 9
MIN_VALUE = -(1 << (ACT_WIDTH - 1))
MAX_VALUE = (1 << (ACT_WIDTH - 1)) - 1


def pack_signed(values):
    packed = 0
    mask = (1 << ACT_WIDTH) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * ACT_WIDTH)
    return packed


def unpack_signed(packed):
    value = int(packed)
    mask = (1 << ACT_WIDTH) - 1
    sign_bit = 1 << (ACT_WIDTH - 1)
    lanes = []
    for lane in range(MAX_MAC_UNITS):
        lane_value = (value >> (lane * ACT_WIDTH)) & mask
        if lane_value & sign_bit:
            lane_value -= 1 << ACT_WIDTH
        lanes.append(lane_value)
    return lanes


def pool_channel(values, side):
    output_side = side // 2
    return [
        max(
            values[(2 * row) * side + 2 * column],
            values[(2 * row) * side + 2 * column + 1],
            values[(2 * row + 1) * side + 2 * column],
            values[(2 * row + 1) * side + 2 * column + 1],
        )
        for row in range(output_side)
        for column in range(output_side)
    ]


def ties_to_even_shift(value, shift):
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient


@cocotb.test()
async def pools_conv0_conv1_and_bypasses_conv2(dut):
    # Conv0: one 10x10 channel -> 25 packed results.
    conv0 = [((lane * 17) % 91) - 45 for lane in range(MAX_MAC_UNITS)]
    dut.conv_index.value = 0
    dut.activation_in_flat.value = pack_signed(conv0)
    await Timer(1, unit="ns")
    expected = pool_channel(conv0, 10) + [0] * 75
    assert unpack_signed(dut.activation_out_flat.value) == expected

    # Conv1: four 5x5 channels -> four packed 2x2 outputs.
    conv1 = []
    expected = []
    for channel in range(4):
        values = [channel * 60 + ((lane * 13) % 47) - 20 for lane in range(25)]
        conv1.extend(values)
        expected.extend(ties_to_even_shift(value, 1) for value in pool_channel(values, 5))
    dut.conv_index.value = 1
    dut.activation_in_flat.value = pack_signed(conv1)
    await Timer(1, unit="ns")
    expected.extend([0] * (MAX_MAC_UNITS - len(expected)))
    assert unpack_signed(dut.activation_out_flat.value) == expected

    # Conv2 does not pool.
    conv2 = [((lane * 9) % 101) - 50 for lane in range(MAX_MAC_UNITS)]
    dut.conv_index.value = 2
    dut.activation_in_flat.value = pack_signed(conv2)
    await Timer(1, unit="ns")
    assert unpack_signed(dut.activation_out_flat.value) == conv2


def pooling_windows(conv_index):
    if conv_index == 0:
        return [
            (row * 20 + col * 2, row * 20 + col * 2 + 1,
             row * 20 + col * 2 + 10, row * 20 + col * 2 + 11)
            for row in range(5) for col in range(5)
        ]
    return [
        (channel * 25 + row * 10 + col * 2,
         channel * 25 + row * 10 + col * 2 + 1,
         channel * 25 + row * 10 + col * 2 + 5,
         channel * 25 + row * 10 + col * 2 + 6)
        for channel in range(4) for row in range(2) for col in range(2)
    ]


async def check_window_cases(dut, conv_index, cases):
    windows = pooling_windows(conv_index)
    dut.conv_index.value = conv_index
    cases = iter(cases)
    checked = 0
    while batch := list(islice(cases, len(windows))):
        inputs = [MAX_VALUE] * MAX_MAC_UNITS
        expected = [0] * MAX_MAC_UNITS
        for output_lane, window in enumerate(windows):
            values = batch[output_lane] if output_lane < len(batch) else (MIN_VALUE,) * 4
            for input_lane, value in zip(window, values):
                inputs[input_lane] = value
            maximum = max(values)
            expected[output_lane] = (
                maximum if conv_index == 0 else ties_to_even_shift(maximum, 1)
            )

        dut.activation_in_flat.value = pack_signed(inputs)
        await Timer(1, unit="ns")
        actual = unpack_signed(dut.activation_out_flat.value)
        assert actual == expected, (
            f"conv {conv_index}, cases {batch}: expected {expected}, got {actual}"
        )
        checked += len(batch)
    return checked


@cocotb.test()
async def covers_all_winners_ties_signed_limits_and_rounding(dut):
    values = (MIN_VALUE, MIN_VALUE + 1, MIN_VALUE + 2, -3, -2, -1, 0, 1, 2, 3,
              MAX_VALUE // 2 - 1, MAX_VALUE // 2, MAX_VALUE // 2 + 1, MAX_VALUE - 1, MAX_VALUE)
    for conv_index in (0, 1):
        checked = await check_window_cases(dut, conv_index, product(values, repeat=4))
        assert checked == len(values)**4

        cases = (
            tuple(value if lane == winner else MIN_VALUE for lane in range(4))
            for value in range(MIN_VALUE, MAX_VALUE + 1) for winner in range(4)
        )
        assert await check_window_cases(dut, conv_index, cases) == (MAX_VALUE - MIN_VALUE + 1) * 4
        dut._log.info("conv %d: %d boundary windows and all %d signed maxima passed",
                      conv_index, checked, MAX_VALUE - MIN_VALUE + 1)

    dut.conv_index.value = 2
    for base in range(MIN_VALUE, MAX_VALUE + 1, MAX_MAC_UNITS):
        values = list(range(base, min(base + MAX_MAC_UNITS, MAX_VALUE + 1)))
        values.extend([0] * (MAX_MAC_UNITS - len(values)))
        dut.activation_in_flat.value = pack_signed(values)
        await Timer(1, unit="ns")
        assert unpack_signed(dut.activation_out_flat.value) == values
