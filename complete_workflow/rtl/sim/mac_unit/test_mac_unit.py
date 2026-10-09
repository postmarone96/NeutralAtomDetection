import cocotb
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, ReadOnly
from cocotb.clock import Clock
import numpy as np

MAC_LATENCY = 3

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


def column(tap):
    return weights[:, tap], activations[:, tap]


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))

    dut.rst.value = 1
    dut.load_bias.value = 0
    dut.valid.value = 0
    dut.bias_flat.value = 0
    dut.activation_flat.value = 0
    dut.weight_flat.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0


async def step(dut, product=None, load=0, bias=None):
    await FallingEdge(dut.clk)
    dut.valid.value = int(product is not None)
    dut.load_bias.value = load
    if bias is not None:
        dut.bias_flat.value = pack_signed(bias, width=18)
    if product is not None:
        weight, activation = product
        dut.weight_flat.value = pack_signed(weight, width=8)
        dut.activation_flat.value = pack_signed(activation, width=10)
    await RisingEdge(dut.clk)
    await ReadOnly()
    return unpack_signed(dut.accum_flat.value.to_unsigned(), 3, 18)


@cocotb.test()
async def test_mac_accumulation_and_stall(dut):
    await reset(dut)

    # Three products back to back; each sum appears MAC_LATENCY clocks after its product.
    sums = [await step(dut, column(tap)) for tap in range(3)]
    sums += [await step(dut) for _ in range(MAC_LATENCY - 1)]
    for early in sums[:MAC_LATENCY - 1]:
        np.testing.assert_array_equal(early, np.zeros(3, dtype=np.int64))
    for tap in range(3):
        np.testing.assert_array_equal(sums[tap + MAC_LATENCY - 1], expected[:, tap])

    # valid=0 must hold the previous accumulator value.
    for _ in range(4):
        np.testing.assert_array_equal(await step(dut), expected[:, 2])
    assert int(dut.busy.value) == 0

    # Final product completes the first dot product; busy lasts until it is summed.
    await step(dut, column(3))
    for _ in range(MAC_LATENCY - 1):
        assert int(dut.busy.value) == 1
        final = await step(dut)
    assert int(dut.busy.value) == 0
    np.testing.assert_array_equal(final, expected[:, 3])

    # The real controller starts a new dot product by loading a zero bias with
    # its first product; no test-only clear input is required.
    zero_bias = np.zeros(3, dtype=np.int64)
    sums = [await step(dut, column(0), load=1, bias=zero_bias)]
    sums += [await step(dut, column(tap)) for tap in range(1, 4)]
    sums += [await step(dut) for _ in range(MAC_LATENCY - 1)]
    for tap in range(4):
        np.testing.assert_array_equal(sums[tap + MAC_LATENCY - 1], expected[:, tap])


@cocotb.test()
async def test_bias_starts_a_new_sum(dut):
    first_bias = np.array([5, -7, 11], dtype=np.int64)
    second_bias = np.array([-3, 9, 2], dtype=np.int64)
    products = weights * activations
    await reset(dut)

    # A load product is added to the bias instead of the running sum.
    sums = [
        await step(dut, column(0), load=1, bias=first_bias),
        await step(dut, column(1)),
        await step(dut, column(2), load=1, bias=second_bias),
    ]
    sums += [await step(dut) for _ in range(MAC_LATENCY - 1)]
    np.testing.assert_array_equal(sums[2], first_bias + products[:, 0])
    np.testing.assert_array_equal(sums[3], first_bias + products[:, 0] + products[:, 1])
    np.testing.assert_array_equal(sums[4], second_bias + products[:, 2])

    # load_bias without valid changes nothing.
    await step(dut, load=1, bias=first_bias)
    for _ in range(MAC_LATENCY):
        np.testing.assert_array_equal(await step(dut), second_bias + products[:, 2])

    # The production reset also drops products still in flight.
    await step(dut, column(3))
    assert int(dut.busy.value) == 1
    await FallingEdge(dut.clk)
    dut.valid.value = 0
    dut.load_bias.value = 0
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    np.testing.assert_array_equal(
        unpack_signed(dut.accum_flat.value.to_unsigned(), 3, 18),
        np.zeros(3, dtype=np.int64),
    )
    assert int(dut.busy.value) == 0
