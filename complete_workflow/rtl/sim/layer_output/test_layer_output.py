import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


MAX_MAC_UNITS = 100
ACT_WIDTH = 9
VALUES_PER_BLOCK = (25, 16, 64, 8, 8, 2)
BLOCK_COUNTS = (4, 2, 1, 1, 1, 1)


def signed_value(value):
    value &= (1 << ACT_WIDTH) - 1
    if value & (1 << (ACT_WIDTH - 1)):
        value -= 1 << ACT_WIDTH
    return value


def pack_lanes(values):
    packed = 0
    mask = (1 << ACT_WIDTH) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * ACT_WIDTH)
    return packed


def unpack_lanes(packed):
    value = int(packed)
    mask = (1 << ACT_WIDTH) - 1
    return [
        signed_value((value >> (lane * ACT_WIDTH)) & mask)
        for lane in range(MAX_MAC_UNITS)
    ]


def block_values(layer, block):
    count = VALUES_PER_BLOCK[layer]
    values = [signed_value(layer * 90 + block * 27 + lane - 200) for lane in range(count)]
    return values + [333] * (MAX_MAC_UNITS - count)


async def reset(dut):
    dut.rst.value = 1
    dut.input_valid.value = 0
    dut.layer_index.value = 0
    dut.output_block_index.value = 0
    dut.input_data_flat.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0


@cocotb.test()
async def assembles_every_conv_and_dense_layer(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))
    await reset(dut)

    for layer in range(6):
        expected = []
        for block in range(BLOCK_COUNTS[layer]):
            values = block_values(layer, block)
            expected.extend(values[:VALUES_PER_BLOCK[layer]])

            await FallingEdge(dut.clk)
            dut.layer_index.value = layer
            dut.output_block_index.value = block
            dut.input_data_flat.value = pack_lanes(values)
            dut.input_valid.value = 1

            await RisingEdge(dut.clk)
            await ReadOnly()
            should_be_valid = block == BLOCK_COUNTS[layer] - 1
            assert int(dut.output_valid.value) == should_be_valid

            if should_be_valid:
                complete_expected = expected + [0] * (MAX_MAC_UNITS - len(expected))
                assert unpack_lanes(dut.output_data_flat.value) == complete_expected

        # No input means no repeated output-valid pulse.
        await FallingEdge(dut.clk)
        dut.input_valid.value = 0
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert int(dut.output_valid.value) == 0
