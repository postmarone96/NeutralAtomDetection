import cocotb
from cocotb.triggers import Timer


MAX_MAC_UNITS = 100
ACC_W_MAX = 17
ACT_W_MAX = 9
ACC_W = (14, 16, 17, 15, 16, 15)
ACC_I = (3, 5, 7, 8, 7, 7)
ACT_W = (8, 9, 6, 9, 9, 8)
ACT_I = (1, 1, 3, 5, 5, 5)


def pack_lanes(values, width):
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * width)
    return packed


def unpack_signed_lanes(packed, lane_count, width):
    packed = int(packed)
    mask = (1 << width) - 1
    result = []
    for lane in range(lane_count):
        value = (packed >> (lane * width)) & mask
        if value & (1 << (width - 1)):
            value -= 1 << width
        result.append(value)
    return result


def round_ties_even(raw, shift):
    quotient = raw >> shift
    remainder = raw - (quotient << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient


def reference(layer, raw):
    acc_low = -(1 << (ACC_W[layer] - 1))
    acc_high = (1 << (ACC_W[layer] - 1)) - 1
    raw = min(max(raw, acc_low), acc_high)
    shift = (ACC_W[layer] - ACC_I[layer]) - (ACT_W[layer] - ACT_I[layer])
    value = round_ties_even(raw, shift)
    high = (1 << (ACT_W[layer] - 1)) - 1
    low = -(1 << (ACT_W[layer] - 1)) if layer == 5 else 0
    return min(max(value, low), high)


@cocotb.test()
async def requantizes_all_layers_and_lanes(dut):
    raw_cases = [
        -(1 << (ACC_W_MAX - 1)), -20000, -97, -96, -95, -65, -64, -63, -33, -32,
        -31, -1, 0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 20000,
        (1 << (ACC_W_MAX - 1)) - 1,
    ]

    dut.input_valid.value = 0
    dut.accum_data_flat.value = 0
    dut.layer_index.value = 0
    await Timer(1, unit="ns")
    dut.input_valid.value = 1
    for layer in range(6):
        lane_values = [raw_cases[lane % len(raw_cases)] for lane in range(MAX_MAC_UNITS)]
        dut.layer_index.value = layer
        dut.accum_data_flat.value = pack_lanes(lane_values, ACC_W_MAX)
        await Timer(1, unit="ns")

        assert int(dut.output_valid.value) == 1
        actual = unpack_signed_lanes(
            dut.activation_data_flat.value,
            MAX_MAC_UNITS,
            ACT_W_MAX,
        )
        expected = [reference(layer, value) for value in lane_values]
        assert actual == expected, f"layer {layer}: expected {expected}, got {actual}"

    dut.input_valid.value = 0
    await Timer(1, unit="ns")
    assert int(dut.output_valid.value) == 0
    assert int(dut.activation_data_flat.value) == 0


@cocotb.test()
async def matches_every_accumulator_value(dut):
    dut.input_valid.value = 1
    low = -(1 << (ACC_W_MAX - 1))
    high = 1 << (ACC_W_MAX - 1)
    checked = 0

    for layer in range(len(ACT_W)):
        dut.layer_index.value = layer
        for start in range(low, high, MAX_MAC_UNITS):
            values = list(range(start, min(start + MAX_MAC_UNITS, high)))
            checked += len(values)
            values.extend([0] * (MAX_MAC_UNITS - len(values)))
            dut.accum_data_flat.value = pack_lanes(values, ACC_W_MAX)
            await Timer(1, unit="ns")

            expected = [reference(layer, value) for value in values]
            assert int(dut.output_valid.value) == 1
            actual_packed = int(dut.activation_data_flat.value)
            if actual_packed != pack_lanes(expected, ACT_W_MAX):
                actual = unpack_signed_lanes(actual_packed, MAX_MAC_UNITS, ACT_W_MAX)
                for lane, (want, got) in enumerate(zip(expected, actual)):
                    assert got == want, (
                        f"layer {layer}, lane {lane}, accumulator {values[lane]}: "
                        f"expected {want}, got {got}"
                    )

        dut._log.info("layer %d: all %d accumulator values passed", layer, high - low)
        dut.input_valid.value = 0
        dut.accum_data_flat.value = pack_lanes([low] * MAX_MAC_UNITS, ACC_W_MAX)
        await Timer(1, unit="ns")
        assert int(dut.output_valid.value) == 0
        assert int(dut.activation_data_flat.value) == 0
        dut.input_valid.value = 1

    assert checked == len(ACT_W) * (1 << ACC_W_MAX)
    dut._log.info("checked %d layer/accumulator combinations", checked)
