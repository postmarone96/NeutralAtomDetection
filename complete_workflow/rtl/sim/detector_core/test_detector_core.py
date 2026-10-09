from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge, Timer


IMAGE_COUNT = 8
FRAME_PIXELS = 28 * 28
SITE_COUNT = 9


def read_codes(path, width, signed=False):
    sign_bit = 1 << (width - 1)
    full_range = 1 << width
    values = []
    for line in path.read_text().splitlines():
        value = int(line, 16)
        if signed and value & sign_bit:
            value -= full_range
        values.append(value)
    return values


def unpack_signed(packed, count, width):
    packed = int(packed)
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    values = []
    for lane in range(count):
        value = (packed >> (lane * width)) & mask
        if value & sign_bit:
            value -= 1 << width
        values.append(value)
    return values


async def check_cycle_counters(dut, measurements):
    first_pixel_ns = cnn_start_ns = None
    ready_ns = last_result_ns = None
    frozen = None
    while True:
        await FallingEdge(dut.clk)
        await Timer(1, unit="ns")
        reset = int(dut.rst.value)
        first_pixel = (int(dut.frame_write_enable.value)
                       and int(dut.frame_write_addr.value) == 0)
        cnn_start = int(dut.dut.inference_start.value)
        await RisingEdge(dut.clk)
        await ReadOnly()
        now = int(cocotb.utils.get_sim_time(unit="ns"))
        if reset:
            assert all(int(signal.value) == 0 for signal in
                       (dut.preprocess_cycles, dut.cnn_cycles, dut.pipeline_cycles))
            first_pixel_ns = cnn_start_ns = None
            ready_ns = last_result_ns = frozen = None
            continue
        if first_pixel:
            first_pixel_ns = now
            ready_ns = last_result_ns = frozen = None
            assert int(dut.pipeline_cycles.value) == 0
        if cnn_start:
            cnn_start_ns = now
            assert int(dut.cnn_cycles.value) == 0
        if first_pixel_ns is None:
            continue
        if ready_ns is None and int(dut.dut.preprocess_site_valid.value):
            ready_ns = now
        if ready_ns is not None:
            assert int(dut.preprocess_cycles.value) == (ready_ns - first_pixel_ns) // 10
        if int(dut.result_valid.value):
            assert cnn_start_ns is not None
            assert int(dut.cnn_cycles.value) == (now - cnn_start_ns) // 10 == 858
            if int(dut.result_site_index.value) == SITE_COUNT - 1:
                last_result_ns = now
        if last_result_ns is not None:
            assert int(dut.pipeline_cycles.value) == (last_result_ns - first_pixel_ns) // 10
        if int(dut.done.value):
            frozen = tuple(int(signal.value) for signal in
                           (dut.preprocess_cycles, dut.cnn_cycles, dut.pipeline_cycles))
            measurements.append(frozen)
        if frozen is not None:
            assert frozen == tuple(int(signal.value) for signal in
                                   (dut.preprocess_cycles, dut.cnn_cycles, dut.pipeline_cycles))


@cocotb.test()
async def raw_frames_match_all_golden_results(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))
    activation_width = len(dut.result_logits_flat) // 2

    vectors = Path(__file__).resolve().parents[2] / "tb" / "vectors"
    frames = read_codes(vectors / "raw_frame.txt", 10)
    logits = read_codes(vectors / "model_classifier_4.txt", 8, signed=True)
    decisions = [int(value) for value in (vectors / "decisions.txt").read_text().splitlines()]

    assert len(frames) == IMAGE_COUNT * FRAME_PIXELS
    assert len(logits) == IMAGE_COUNT * SITE_COUNT * 2
    assert len(decisions) == IMAGE_COUNT * SITE_COUNT

    dut.rst.value = 1
    dut.frame_write_enable.value = 0
    dut.frame_write_addr.value = 0
    dut.frame_write_data.value = 0
    dut.start.value = 0
    counter_measurements = []
    counter_task = cocotb.start_soon(check_cycle_counters(dut, counter_measurements))
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst.value = 0

    # Abort after accepting a crop, before loading it into the CNN.
    for address, pixel in enumerate(frames[:FRAME_PIXELS]):
        dut.frame_write_enable.value = 1
        dut.frame_write_addr.value = address
        dut.frame_write_data.value = pixel
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
    dut.frame_write_enable.value = 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(40):
        if int(dut.dut.preprocess_site_ready.value):
            break
        await FallingEdge(dut.clk)
    else:
        raise AssertionError("preprocessing did not present the first crop")
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.dut.inference_load_input.value) == 1
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    await ClockCycles(dut.clk, 2)
    await ReadOnly()
    assert int(dut.busy.value) == 0
    assert int(dut.result_valid.value) == 0
    assert int(dut.dut.inference_load_input.value) == 0
    await FallingEdge(dut.clk)
    dut.rst.value = 0

    measured_latencies = []
    for image in range(IMAGE_COUNT):
        image_pixels = frames[image * FRAME_PIXELS:(image + 1) * FRAME_PIXELS]
        for address, pixel in enumerate(image_pixels):
            if image in (1, 4) and address in (32, 200, 700):
                dut.frame_write_enable.value = 0
                await ClockCycles(dut.clk, 3)
                await FallingEdge(dut.clk)
            dut.frame_write_enable.value = 1
            dut.frame_write_addr.value = address
            dut.frame_write_data.value = pixel
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)

        dut.frame_write_enable.value = 0
        dut.start.value = 1
        await RisingEdge(dut.clk)
        start_time_ns = cocotb.utils.get_sim_time(unit="ns")
        await FallingEdge(dut.clk)
        dut.start.value = 0

        seen = 0
        clocks = 0
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            clocks += 1

            if int(dut.result_valid.value):
                site = int(dut.result_site_index.value)
                assert site == seen, f"image {image}: expected site {seen}, got {site}"
                actual_logits = unpack_signed(dut.result_logits_flat.value, 2, activation_width)
                base = (image * SITE_COUNT + site) * 2
                assert actual_logits == logits[base:base + 2], (
                    f"image {image}, site {site}: expected {logits[base:base + 2]}, "
                    f"got {actual_logits}"
                )
                assert int(dut.result_decision.value) == decisions[image * SITE_COUNT + site]
                seen += 1

            if int(dut.done.value):
                assert seen == SITE_COUNT
                measured_latencies.append(clocks)
                break

            assert clocks < 20_000, f"image {image}: detector did not finish"

        elapsed_ns = int(cocotb.utils.get_sim_time(unit="ns") - start_time_ns)
        dut._log.info(
            "image %d completed in %d clocks (%d ns at the 100 MHz test clock)",
            image,
            clocks,
            elapsed_ns,
        )
        await FallingEdge(dut.clk)
        await ClockCycles(dut.clk, 3)
        await FallingEdge(dut.clk)

    assert len(set(measured_latencies)) == 1, measured_latencies
    assert len(counter_measurements) == IMAGE_COUNT
    baseline = counter_measurements[0]
    assert baseline == (802, 858, 8559), baseline
    for image, counts in enumerate(counter_measurements):
        gaps = 9 if image in (1, 4) else 0
        assert counts == (baseline[0] + gaps, baseline[1], baseline[2] + gaps)
    dut._log.info("hardware counters without input gaps: preprocessing=%d, CNN=%d, pipeline=%d",
                  *baseline)

    dut.frame_write_enable.value = 1
    dut.frame_write_addr.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.frame_write_enable.value = 0
    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    await ClockCycles(dut.clk, 2)
    await ReadOnly()
    assert all(int(signal.value) == 0 for signal in
               (dut.preprocess_cycles, dut.cnn_cycles, dut.pipeline_cycles))
    counter_task.cancel()
