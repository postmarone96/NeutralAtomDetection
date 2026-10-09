import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


MAX_MAC_UNITS = 100
ACT_WIDTH = 9
NB_WIDTH = 9
LAYER_INPUT_COUNTS = (1, 4, 8, 64, 16, 8)
LAYER_OUTPUT_COUNTS = (4, 8, 16, 8, 8, 2)


def to_signed(value, width):
    value &= (1 << width) - 1
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value


def pack_signed(values, width):
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * width)
    return packed


def unpack_signed(packed, lane_count, width):
    value = int(packed)
    mask = (1 << width) - 1
    return [
        to_signed((value >> (lane * width)) & mask, width)
        for lane in range(lane_count)
    ]


def round_shift_ties_to_even(raw, shift):
    quotient = raw >> shift
    remainder = raw - (quotient << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient


@cocotb.test()
async def selects_broadcasts_and_concatenates_neighbors(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))

    ram_values = [to_signed(index * 11 - 300, ACT_WIDTH) for index in range(MAX_MAC_UNITS)]
    neighbor_values = [-72, -56, -40, -24, 24, 40, 56, 72]

    dut.rst.value = 1
    dut.read_valid.value = 0
    dut.layer_index.value = 3
    dut.input_index.value = 0
    dut.read_data_flat.value = pack_signed(ram_values, ACT_WIDTH)
    dut.neighbor_data_flat.value = pack_signed(neighbor_values, NB_WIDTH)
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0

    for layer in range(3, 6):
        for input_index in range(LAYER_INPUT_COUNTS[layer]):
            await FallingEdge(dut.clk)
            dut.layer_index.value = layer
            dut.input_index.value = input_index
            dut.read_valid.value = 1

            await RisingEdge(dut.clk)
            await ReadOnly()
            assert int(dut.dense_input_valid.value) == 1

            if layer == 4 and input_index >= LAYER_OUTPUT_COUNTS[layer - 1]:
                neighbor_index = input_index - LAYER_OUTPUT_COUNTS[layer - 1]
                selected = round_shift_ties_to_even(neighbor_values[neighbor_index], 4)
            else:
                selected = ram_values[input_index]

            expected = [selected] * LAYER_OUTPUT_COUNTS[layer]
            expected.extend([0] * (MAX_MAC_UNITS - len(expected)))
            assert unpack_signed(
                dut.dense_input_data_flat.value,
                MAX_MAC_UNITS,
                ACT_WIDTH,
            ) == expected

    await FallingEdge(dut.clk)
    dut.read_valid.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.dense_input_valid.value) == 0
