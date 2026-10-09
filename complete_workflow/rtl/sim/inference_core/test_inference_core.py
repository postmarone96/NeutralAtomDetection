from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


SITE_COUNT = 72
MAX_MAC_UNITS = 100
ACT_W_MAX = 9
NEIGHBORS = 8
NB_W = 9
EXPECTED_LATENCY_CYCLES = 858


def read_signed_codes(path, width):
    sign_bit = 1 << (width - 1)
    full_range = 1 << width
    values = []
    for line in path.read_text().splitlines():
        value = int(line, 16)
        if value & sign_bit:
            value -= full_range
        values.append(value)
    return values


def pack_lanes(values, width):
    packed = 0
    mask = (1 << width) - 1
    for lane, value in enumerate(values):
        packed |= (int(value) & mask) << (lane * width)
    return packed


def unpack_lanes(packed, lane_count, width):
    packed = int(packed)
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    values = []
    for lane in range(lane_count):
        value = (packed >> (lane * width)) & mask
        if value & sign_bit:
            value -= 1 << width
        values.append(value)
    return values


@cocotb.test()
async def matches_all_72_golden_sites(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start(start_high=False))

    vectors = Path(__file__).resolve().parents[2] / "tb" / "vectors"
    crops = read_signed_codes(vectors / "input_crop.txt", 8)
    neighbors = read_signed_codes(vectors / "input_neighbor.txt", 9)
    logits = read_signed_codes(vectors / "model_classifier_4.txt", 8)
    decisions = [int(value) for value in (vectors / "decisions.txt").read_text().splitlines()]

    assert len(crops) == SITE_COUNT * MAX_MAC_UNITS
    assert len(neighbors) == SITE_COUNT * NEIGHBORS
    assert len(logits) == SITE_COUNT * 2

    dut.rst.value = 1
    dut.load_input.value = 0
    dut.start.value = 0
    dut.input_data_flat.value = 0
    dut.neighbor_data_flat.value = 0
    await ClockCycles(dut.clk, 2)
    await FallingEdge(dut.clk)
    dut.rst.value = 0

    for site in range(SITE_COUNT):
        crop = crops[site * MAX_MAC_UNITS:(site + 1) * MAX_MAC_UNITS]
        neighbor = neighbors[site * NEIGHBORS:(site + 1) * NEIGHBORS]

        dut.input_data_flat.value = pack_lanes(crop, ACT_W_MAX)
        dut.neighbor_data_flat.value = pack_lanes(neighbor, NB_W)
        dut.load_input.value = 1
        await RisingEdge(dut.clk)

        await FallingEdge(dut.clk)
        dut.load_input.value = 0
        dut.start.value = 1
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert int(dut.busy.value) == 1

        await FallingEdge(dut.clk)
        dut.start.value = 0
        for latency_cycles in range(1, EXPECTED_LATENCY_CYCLES + 1):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if int(dut.result_valid.value) == 1:
                break
        else:
            raise AssertionError(
                f"site {site}: no result within {EXPECTED_LATENCY_CYCLES} cycles"
            )

        assert latency_cycles == EXPECTED_LATENCY_CYCLES, (
            f"site {site}: expected {EXPECTED_LATENCY_CYCLES} cycles, "
            f"got {latency_cycles}"
        )
        dut._log.info("site %d latency: %d cycles", site, latency_cycles)
        actual = unpack_lanes(dut.result_data_flat.value, MAX_MAC_UNITS, ACT_W_MAX)
        expected_logits = logits[site * 2:(site + 1) * 2]
        assert actual[:2] == expected_logits, (
            f"site {site}: expected logits {expected_logits}, got {actual[:2]}"
        )
        assert int(dut.decision.value) == decisions[site], (
            f"site {site}: expected decision {decisions[site]}, got {int(dut.decision.value)}"
        )
        await FallingEdge(dut.clk)
