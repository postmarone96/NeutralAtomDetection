import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, ReadOnly
from cocotb.clock import Clock

from pathlib import Path

mem_path = Path(__file__).resolve().parents[2] / "src" / "weights.mem"

words = []
with mem_path.open("r") as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        value = int(line, 16)
        words.append(value)
        # print(f"line {idx}: {line} -> {line}")

@cocotb.test()
async def test_rom(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))

    dut.read_enable.value = 1
    for i, expected in enumerate(words):
        dut.addr.value = i
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert dut.read_valid.value == 1
        assert expected == dut.data_out.value.to_unsigned()
        await FallingEdge(dut.clk)

    dut.read_enable.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.read_valid.value == 0
