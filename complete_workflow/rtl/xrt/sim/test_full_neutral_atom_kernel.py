from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge, Timer, with_timeout
from frame_memory import serve_frame_reads


FRAME_BASE = 0x0000_0001_0000_1000
RESULTS_BASE = 0x0000_0002_0000_2000
FRAME_PIXELS = 28 * 28
FRAME_BEATS = 25
SITE_COUNT = 9


def read_hex_values(path, signed_width=None):
    values = []
    sign_bit = 1 << (signed_width - 1) if signed_width else 0
    full_range = 1 << signed_width if signed_width else 0
    for token in path.read_text().split():
        value = int(token, 16)
        if signed_width and value & sign_bit:
            value -= full_range
        values.append(value)
    return values


async def axi_lite_write(dut, address, value):
    while True:
        await FallingEdge(dut.ap_clk)
        if int(dut.s_axi_control_awready.value) and int(
            dut.s_axi_control_wready.value
        ):
            break
    dut.s_axi_control_awaddr.value = address
    dut.s_axi_control_awvalid.value = 1
    dut.s_axi_control_wdata.value = value
    dut.s_axi_control_wstrb.value = 0xF
    dut.s_axi_control_wvalid.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.s_axi_control_awvalid.value = 0
    dut.s_axi_control_wvalid.value = 0

    while not int(dut.s_axi_control_bvalid.value):
        await FallingEdge(dut.ap_clk)
    assert int(dut.s_axi_control_bresp.value) == 0
    dut.s_axi_control_bready.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.s_axi_control_bready.value = 0


async def service_memory(dut, frame, captured, response_gap):
    await serve_frame_reads(dut, frame, FRAME_BASE, response_gap=response_gap)

    address_seen = False
    data_seen = False
    while not (address_seen and data_seen):
        await FallingEdge(dut.ap_clk)
        if int(dut.m_axi_gmem_awvalid.value) and not address_seen:
            assert int(dut.m_axi_gmem_awaddr.value) == RESULTS_BASE
            dut.m_axi_gmem_awready.value = 1
            address_seen = True
        if int(dut.m_axi_gmem_wvalid.value) and not data_seen:
            captured["word"] = int(dut.m_axi_gmem_wdata.value)
            captured["strobe"] = int(dut.m_axi_gmem_wstrb.value)
            assert int(dut.m_axi_gmem_wlast.value) == 1
            dut.m_axi_gmem_wready.value = 1
            data_seen = True
        await RisingEdge(dut.ap_clk)
        await FallingEdge(dut.ap_clk)
        dut.m_axi_gmem_awready.value = 0
        dut.m_axi_gmem_wready.value = 0

    dut._log.info("captured result write")

    while not int(dut.m_axi_gmem_bready.value):
        await FallingEdge(dut.ap_clk)
    dut.m_axi_gmem_bresp.value = 0
    dut.m_axi_gmem_bvalid.value = 1
    await RisingEdge(dut.ap_clk)
    await FallingEdge(dut.ap_clk)
    dut.m_axi_gmem_bvalid.value = 0


def unpack_signed_10(value):
    value &= 0x3FF
    return value - 0x400 if value & 0x200 else value


async def measure_hardware_events(dut, measured):
    first_pixel_ns = ready_ns = cnn_start_ns = None
    previous_pixel_ns = None
    pixels = 0
    while True:
        await FallingEdge(dut.ap_clk)
        await Timer(0.5, unit="ns")
        first_pixel = (int(dut.frame_write_enable.value)
                       and int(dut.frame_write_addr.value) == 0)
        pixel_valid = int(dut.frame_write_enable.value)
        cnn_start = int(dut.detector.inference_start.value)
        await RisingEdge(dut.ap_clk)
        await ReadOnly()
        now = int(cocotb.utils.get_sim_time(unit="ns"))
        if first_pixel:
            first_pixel_ns = now
        if pixel_valid:
            if previous_pixel_ns is not None:
                assert now - previous_pixel_ns == 4, 'Gap in preprocessing input'
            previous_pixel_ns = now
            pixels += 1
        if cnn_start:
            cnn_start_ns = now
        if first_pixel_ns is not None and ready_ns is None and int(
                dut.detector.preprocess_site_valid.value):
            ready_ns = now
        if int(dut.detector_result_valid.value):
            assert (now - cnn_start_ns) // 4 == 858
            if int(dut.detector_result_site.value) == SITE_COUNT - 1:
                assert pixels == 784
                measured["cycles"] = ((ready_ns - first_pixel_ns) // 4, 858,
                                      (now - first_pixel_ns) // 4)
                return


@cocotb.test()
async def full_kernel_matches_eight_frames_without_pixel_gaps(dut):
    dut._log.info("starting clock")
    cocotb.start_soon(Clock(dut.ap_clk, 4, unit="ns").start(start_high=False))

    vectors = Path(__file__).resolve().parents[2] / "tb" / "vectors"
    frames = read_hex_values(vectors / "raw_frame.txt")
    all_logits = read_hex_values(vectors / "model_classifier_4.txt", signed_width=8)
    all_decisions = [int(value) for value in (vectors / "decisions.txt").read_text().split()]

    dut._log.info("driving reset")
    dut.ap_rst_n.value = 0
    dut._log.info("initializing AXI inputs")
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
    dut._log.info("waiting for reset clocks")
    await ClockCycles(dut.ap_clk, 5)
    await FallingEdge(dut.ap_clk)
    dut.ap_rst_n.value = 1
    dut._log.info("reset released")

    await axi_lite_write(dut, 0x10, FRAME_BASE & 0xFFFF_FFFF)
    await axi_lite_write(dut, 0x14, FRAME_BASE >> 32)
    await axi_lite_write(dut, 0x1C, RESULTS_BASE & 0xFFFF_FFFF)
    await axi_lite_write(dut, 0x20, RESULTS_BASE >> 32)
    dut._log.info("addresses programmed")

    for image in range(8):
        frame = frames[image * FRAME_PIXELS:(image + 1) * FRAME_PIXELS]
        logits = all_logits[image * 18:(image + 1) * 18]
        decisions = all_decisions[image * 9:(image + 1) * 9]
        captured = {}
        measured = {}
        monitor_task = cocotb.start_soon(measure_hardware_events(dut, measured))
        memory_task = cocotb.start_soon(service_memory(dut, frame, captured, response_gap=(image % 3) * 40))
        await axi_lite_write(dut, 0x00, 1)
        dut._log.info("kernel started")
        start_ns = int(cocotb.utils.get_sim_time(unit="ns"))
        await with_timeout(memory_task, 100, "us")
        elapsed_clocks = (
            int(cocotb.utils.get_sim_time(unit="ns")) - start_ns
        ) // 4

        await monitor_task
        assert captured["strobe"] == (1 << 52) - 1
        counters = tuple((captured["word"] >> (word * 32)) & 0xFFFF_FFFF
                         for word in range(9, 12))
        assert counters == measured["cycles"]
        assert counters == (802, 858, 8559)
        assert captured["word"] >> 416 == 0
        assert (captured["word"] >> 384) & 0xFFFF_FFFF == 0x43594331
        dut._log.info("XRT cycle payload: preprocessing=%d, CNN=%d, pipeline=%d", *counters)
        for site in range(SITE_COUNT):
            record = (captured["word"] >> (site * 32)) & 0xFFFF_FFFF
            actual = (
                unpack_signed_10(record),
                unpack_signed_10(record >> 10),
                (record >> 20) & 1,
            )
            expected = (logits[2 * site], logits[2 * site + 1], decisions[site])
            assert actual == expected, f"site {site}: expected {expected}, got {actual}"

        dut._log.info("frame %d: full XRT kernel completed in %d clocks", image, elapsed_clocks)
        assert elapsed_clocks < 12_000
