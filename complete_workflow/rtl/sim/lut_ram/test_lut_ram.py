import cocotb
from cocotb.triggers import ReadOnly, RisingEdge, with_timeout


@cocotb.test
async def test_vhdl_lut_ram_testbench(dut):
    """Let the VHDL testbench drive and verify all nested-array signals."""
    await with_timeout(RisingEdge(dut.tb_done), 2, "us")
    await ReadOnly()
    assert dut.tb_passed.value == 1
