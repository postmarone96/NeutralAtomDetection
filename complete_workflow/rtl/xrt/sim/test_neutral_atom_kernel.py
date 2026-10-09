import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge
from frame_memory import serve_frame_reads


FRAME_BASE = 0x0000_0001_0000_1FC0
RESULTS_BASE = 0x0000_0002_0000_2000
FRAME_PIXELS = 784
FRAME_BEATS = 25


async def axi_lite_write(dut, address, value, address_first=True):
    """Issue split AW and W channels to exercise independent acceptance."""

    async def send_address():
        while True:
            await FallingEdge(dut.ap_clk)
            if int(dut.s_axi_control_awready.value):
                break
        dut.s_axi_control_awaddr.value = address
        dut.s_axi_control_awvalid.value = 1
        await RisingEdge(dut.ap_clk)
        await FallingEdge(dut.ap_clk)
        dut.s_axi_control_awvalid.value = 0

    async def send_data():
        while True:
            await FallingEdge(dut.ap_clk)
            if int(dut.s_axi_control_wready.value):
                break
        dut.s_axi_control_wdata.value = value
        dut.s_axi_control_wstrb.value = 0xF
        dut.s_axi_control_wvalid.value = 1
        await RisingEdge(dut.ap_clk)
        await FallingEdge(dut.ap_clk)
        dut.s_axi_control_wvalid.value = 0

    if address_first:
        await send_address()
        await ClockCycles(dut.ap_clk, 1)
        await send_data()
    else:
        await send_data()
        await ClockCycles(dut.ap_clk, 1)
        await send_address()

    dut.s_axi_control_bready.value = 0
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.s_axi_control_bvalid.value):
            assert int(dut.s_axi_control_bresp.value) == 0
            break
    dut.s_axi_control_bready.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.s_axi_control_bready.value = 0


async def axi_lite_read(dut, address):
    # Hold RREADY low until the slave presents data so RVALID is exercised
    # under backpressure and cannot be missed on the address-handshake edge.
    dut.s_axi_control_rready.value = 0
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.s_axi_control_arready.value):
            break
    dut.s_axi_control_araddr.value = address
    dut.s_axi_control_arvalid.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.s_axi_control_arvalid.value = 0
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.s_axi_control_rvalid.value):
            assert int(dut.s_axi_control_rresp.value) == 0
            value = int(dut.s_axi_control_rdata.value)
            break
    dut.s_axi_control_rready.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.s_axi_control_rready.value = 0
    return value


def frame_beat(beat):
    packed = 0
    for lane in range(32):
        pixel_index = beat * 32 + lane
        value = pixel_index & 0x3FF if pixel_index < FRAME_PIXELS else 0
        packed |= value << (lane * 16)
    return packed


def expected_result_word():
    packed = 0
    for site in range(9):
        logit0 = (site - 4) & 0x3FF
        logit1 = (20 + site) & 0x3FF
        record = logit0 | (logit1 << 10) | ((site & 1) << 20)
        packed |= record << (site * 32)
    for word, value in enumerate((812, 858, 8550, 0x43594331), start=9):
        packed |= value << (word * 32)
    return packed


async def service_global_memory(dut, captured):
    await serve_frame_reads(dut, [i & 1023 for i in range(784)], FRAME_BASE, response_gap=40)

    # Accept W before AW to verify that the two channels are tracked separately.
    await ClockCycles(dut.ap_clk, 3)
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.m_axi_gmem_wvalid.value):
            captured["wdata"] = int(dut.m_axi_gmem_wdata.value)
            captured["wstrb"] = int(dut.m_axi_gmem_wstrb.value)
            assert int(dut.m_axi_gmem_wlast.value) == 1
            break
    dut.m_axi_gmem_wready.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.m_axi_gmem_wready.value = 0

    await ClockCycles(dut.ap_clk, 2)
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.m_axi_gmem_awvalid.value):
            captured["awaddr"] = int(dut.m_axi_gmem_awaddr.value)
            assert int(dut.m_axi_gmem_awlen.value) == 0
            assert int(dut.m_axi_gmem_awsize.value) == 6
            break
    dut.m_axi_gmem_awready.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.m_axi_gmem_awready.value = 0

    await ClockCycles(dut.ap_clk, 2)
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.m_axi_gmem_bready.value):
            break
    dut.m_axi_gmem_bresp.value = 0
    dut.m_axi_gmem_bvalid.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.m_axi_gmem_bvalid.value = 0


@cocotb.test()
async def loads_raw_frame_and_writes_packed_results(dut):
    cocotb.start_soon(Clock(dut.ap_clk, 4, unit="ns").start(start_high=False))

    dut.ap_rst_n.value = 0
    dut.s_axi_control_awaddr.value = 0
    dut.s_axi_control_awvalid.value = 0
    dut.s_axi_control_wdata.value = 0
    dut.s_axi_control_wstrb.value = 0
    dut.s_axi_control_wvalid.value = 0
    dut.s_axi_control_bready.value = 0
    dut.s_axi_control_araddr.value = 0
    dut.s_axi_control_arvalid.value = 0
    dut.s_axi_control_rready.value = 0
    dut.m_axi_gmem_awready.value = 0
    dut.m_axi_gmem_wready.value = 0
    dut.m_axi_gmem_bresp.value = 0
    dut.m_axi_gmem_bvalid.value = 0
    dut.m_axi_gmem_arready.value = 0
    dut.m_axi_gmem_rdata.value = 0
    dut.m_axi_gmem_rresp.value = 0
    dut.m_axi_gmem_rlast.value = 0
    dut.m_axi_gmem_rvalid.value = 0
    await ClockCycles(dut.ap_clk, 5)
    await FallingEdge(dut.ap_clk)
    dut.ap_rst_n.value = 1

    await axi_lite_write(dut, 0x10, FRAME_BASE & 0xFFFF_FFFF, address_first=True)
    await axi_lite_write(dut, 0x14, FRAME_BASE >> 32, address_first=False)
    await axi_lite_write(dut, 0x1C, RESULTS_BASE & 0xFFFF_FFFF, address_first=True)
    await axi_lite_write(dut, 0x20, RESULTS_BASE >> 32, address_first=False)

    assert await axi_lite_read(dut, 0x10) == (FRAME_BASE & 0xFFFF_FFFF)
    assert await axi_lite_read(dut, 0x14) == (FRAME_BASE >> 32)
    assert await axi_lite_read(dut, 0x1C) == (RESULTS_BASE & 0xFFFF_FFFF)
    assert await axi_lite_read(dut, 0x20) == (RESULTS_BASE >> 32)

    captured = {}
    memory_task = cocotb.start_soon(service_global_memory(dut, captured))
    await axi_lite_write(dut, 0x00, 1, address_first=True)
    await memory_task

    for _ in range(30):
        control = await axi_lite_read(dut, 0x00)
        if control & 0x2:
            break
    else:
        raise AssertionError("ap_done did not assert")

    assert control & 0x4, "kernel did not return to idle"
    assert captured["awaddr"] == RESULTS_BASE
    assert captured["wstrb"] == (1 << 52) - 1
    assert captured["wdata"] == expected_result_word()
    assert await axi_lite_read(dut, 0x28) == 0
    assert await axi_lite_read(dut, 0x2C) == 9
