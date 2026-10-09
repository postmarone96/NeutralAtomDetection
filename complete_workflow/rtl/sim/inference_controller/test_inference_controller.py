import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge


INPUT_COUNTS = (1, 4, 8, 64, 16, 8)
BLOCK_COUNTS = (4, 2, 1, 1, 1, 1)
TAP_COUNTS = (9, 9, 9, 1, 1, 1)


def expected_operations():
    return [
        (layer, block, input_index, tap)
        for layer in range(6)
        for block in range(BLOCK_COUNTS[layer])
        for input_index in range(INPUT_COUNTS[layer])
        for tap in range(TAP_COUNTS[layer])
    ]


@cocotb.test()
async def runs_every_layer_in_order_and_measures_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))

    dut.rst.value = 1
    dut.start.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    dut.start.value = 1

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.busy.value) == 1
    bias_loads = []
    if int(dut.mac_load_bias.value):
        bias_loads.append((
            int(dut.layer_index.value),
            int(dut.output_block_index.value),
            int(dut.input_index.value),
            int(dut.tap_index.value),
        ))
    await FallingEdge(dut.clk)
    dut.start.value = 0

    operations = []
    emitted_blocks = []
    write_layers = []
    elapsed_cycles = 0

    while True:
        await RisingEdge(dut.clk)
        elapsed_cycles += 1
        await ReadOnly()

        layer = int(dut.layer_index.value)
        block = int(dut.output_block_index.value)
        input_index = int(dut.input_index.value)
        tap = int(dut.tap_index.value)

        if int(dut.mac_valid.value):
            operations.append((layer, block, input_index, tap))
        if int(dut.mac_load_bias.value):
            bias_loads.append((layer, block, input_index, tap))
        if int(dut.layer_output_input_valid.value):
            emitted_blocks.append((layer, block))
        if int(dut.activation_write_enable.value):
            write_layers.append(layer)

        if int(dut.done.value):
            assert int(dut.result_valid.value) == 1
            assert int(dut.busy.value) == 0
            break

        assert elapsed_cycles < 1000, "controller did not finish"

    assert operations == expected_operations()
    assert bias_loads == [
        (layer, block, 0, 0)
        for layer in range(6)
        for block in range(BLOCK_COUNTS[layer])
    ]
    assert emitted_blocks == [
        (layer, block)
        for layer in range(6)
        for block in range(BLOCK_COUNTS[layer])
    ]
    assert write_layers == [0, 1, 2, 3, 4]

    # From the clock after start is accepted through the registered done pulse.
    assert elapsed_cycles == 840, f"expected 840 cycles, got {elapsed_cycles}"

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.done.value) == 0
    assert int(dut.result_valid.value) == 0
