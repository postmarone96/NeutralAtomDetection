import cocotb
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, ReadOnly
from cocotb.clock import Clock
import numpy as np

weights = np.array([
    [3, -1, 2, -4],
    [2, 3, -2, 1],
    [4, 2, 5, -3],
], dtype=np.int64)

activations = np.array([
    [2, 4, -3, -2],
    [5, -2, 6, 7],
    [-1, 6, 0, -5],
], dtype=np.int64)

expected = np.array([
    [ 6,  2, -4,  4],
    [10,  4, -8, -1],
    [-4,  8,  8, 23],
], dtype=np.int64)


def pack_signed(values, width):
    packed = 0
    mask = (1 << width) - 1

    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * width)

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

    return np.array(values, dtype=np.int64)

@cocotb.test
async def test_mac_unit(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)
    dut.clear.value = 1
    await ClockCycles(dut.clk, 1)
    dut.clear.value = 0
    await ClockCycles(dut.clk, 1)

    dut.valid.value = 1
    for tap in range(weights.shape[1] - 1):
        dut.weight_flat.value = pack_signed(weights[:, tap], width=8)
        dut.activation_flat.value = pack_signed(activations[:, tap], width=10)
        await RisingEdge(dut.clk)
        await ReadOnly()
        actual = unpack_signed(
            dut.accum_flat.value.to_unsigned(),
            lane_count=3,
            width=18,
        )
        np.testing.assert_array_equal(actual, expected[:, tap])
        await FallingEdge(dut.clk)

    dut.valid.value = 0
    await ClockCycles(dut.clk, 4)

    dut.valid.value = 1
    index = weights.shape[1] - 1
    dut.weight_flat.value = pack_signed(weights[:, index], width=8)
    dut.activation_flat.value = pack_signed(activations[:, index], width=10)
    await RisingEdge(dut.clk)
    await ReadOnly()
    actual = unpack_signed(
        dut.accum_flat.value.to_unsigned(),
        lane_count=3,
        width=18,
    )
    np.testing.assert_array_equal(actual, expected[:, index])

    await FallingEdge(dut.clk)
    dut.valid.value = 0
    await ClockCycles(dut.clk, 1)

    # test clearing between two dot products
    await ClockCycles(dut.clk, 1)
    dut.clear.value = 1
    await ClockCycles(dut.clk, 1)
    dut.clear.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    actual = unpack_signed(
        dut.accum_flat.value.to_unsigned(),
        lane_count=3,
        width=18,
    )
    np.testing.assert_array_equal(actual, np.array([0,0,0]))

    # second dot product
    await FallingEdge(dut.clk)
    dut.valid.value = 1
    for tap in range(weights.shape[1]):
        dut.weight_flat.value = pack_signed(weights[:, tap], width=8)
        dut.activation_flat.value = pack_signed(activations[:, tap], width=10)
        await RisingEdge(dut.clk)
        await ReadOnly()
        actual = unpack_signed(
            dut.accum_flat.value.to_unsigned(),
            lane_count=3,
            width=18,
        )
        np.testing.assert_array_equal(actual, expected[:, tap])
        await FallingEdge(dut.clk)






