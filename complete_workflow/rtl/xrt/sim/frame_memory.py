"""AXI frame reader model with burst, boundary and arbitrary response-gap checks."""
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge


async def serve_frame_reads(dut, frame, base, response_gap=0):
    beat = 0
    while beat < 25:
        await ClockCycles(dut.ap_clk, beat % 3)
        while True:
            await FallingEdge(dut.ap_clk)
            if int(dut.m_axi_gmem_arvalid.value):
                break
        address = int(dut.m_axi_gmem_araddr.value)
        count = int(dut.m_axi_gmem_arlen.value) + 1
        size = int(dut.m_axi_gmem_arsize.value)
        assert address == base + beat * 64
        assert 1 <= count <= 16
        assert (address % 4096) + count * (1 << size) <= 4096
        assert size == (5 if beat == 24 else 6)
        assert count == 1 if beat == 24 else beat + count <= 24
        assert int(dut.m_axi_gmem_arburst.value) == 1
        dut.m_axi_gmem_arready.value = 1
        await RisingEdge(dut.ap_clk)
        await FallingEdge(dut.ap_clk)
        dut.m_axi_gmem_arready.value = 0
        for offset in range(count):
            await ClockCycles(dut.ap_clk, response_gap + beat % 5)
            await FallingEdge(dut.ap_clk)
            word = 0
            for slot in range(32):
                pixel = beat * 32 + slot
                value = frame[pixel] if pixel < 784 else 1023
                word |= int(value) << (16 * slot)
            dut.m_axi_gmem_rdata.value = word
            dut.m_axi_gmem_rresp.value = 0
            dut.m_axi_gmem_rlast.value = int(offset == count - 1)
            dut.m_axi_gmem_rvalid.value = 1
            while True:
                ready = int(dut.m_axi_gmem_rready.value)
                await RisingEdge(dut.ap_clk)
                await FallingEdge(dut.ap_clk)
                if ready:
                    break
            dut.m_axi_gmem_rvalid.value = 0
            beat += 1
    return beat
