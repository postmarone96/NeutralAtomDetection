import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run_detector import (
    COUNTER_FORMAT, CycleCounts, FRAME_PIXELS, latency_us, load_frame,
    run_kernel, signed_10, unpack_cycles, unpack_results,
)


def pack_result(logit_0: int, logit_1: int, decision: int) -> int:
    return (logit_0 & 0x3FF) | ((logit_1 & 0x3FF) << 10) | (decision << 20)


class HostUtilityTests(unittest.TestCase):
    def test_binary_frame_selection(self):
        frames = np.arange(2 * FRAME_PIXELS, dtype=np.uint16) % 1024
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.bin"
            frames.astype("<u2").tofile(path)
            selected = load_frame(path, frame_index=1)
        np.testing.assert_array_equal(selected, frames[FRAME_PIXELS:])

    def test_hex_text_frame_selection(self):
        frames = np.arange(2 * FRAME_PIXELS, dtype=np.uint16) % 1024
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.txt"
            path.write_text("\n".join(f"{value:03x}" for value in frames))
            selected = load_frame(path, frame_index=1)
        np.testing.assert_array_equal(selected, frames[FRAME_PIXELS:])

    def test_signed_10_limits(self):
        self.assertEqual(signed_10(0x1FF), 511)
        self.assertEqual(signed_10(0x200), -512)
        self.assertEqual(signed_10(0x3FF), -1)

    def test_unpack_results(self):
        words = np.zeros(16, dtype=np.uint32)
        expected = []
        for site in range(9):
            result = (site - 4, 4 - site, site & 1)
            expected.append(result)
            words[site] = pack_result(*result)
        self.assertEqual(unpack_results(words), expected)

    def test_hardware_cycles_and_latency(self):
        words = np.zeros(16, dtype=np.uint32)
        words[9:13] = [802, 858, 8550, COUNTER_FORMAT]
        self.assertEqual(unpack_cycles(words), CycleCounts(802, 858, 8550))
        self.assertAlmostEqual(latency_us(858, 300), 2.86)
        self.assertAlmostEqual(latency_us(8550, 300), 28.5)
        for frequency in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                latency_us(858, frequency)

    def test_old_xclbin_and_short_buffer_rejected(self):
        for words in (np.zeros(16, dtype=np.uint32), np.zeros(12, dtype=np.uint32)):
            with self.assertRaisesRegex(ValueError, "rebuild"):
                unpack_cycles(words)

    def test_xrt_transfers_preserve_clear_input_launch_wait_readback_order(self):
        events = []
        frame = np.arange(FRAME_PIXELS, dtype=np.uint16)

        class Buffer:
            normal = 0

            def __init__(self, device, size, flags, group):
                self.group = group
                self.data = bytearray(size)

            def map(self):
                return self.data

            def sync(self, direction, size, offset):
                events.append(("sync", self.group, direction))
                if self.group == 0:
                    np.testing.assert_array_equal(
                        np.frombuffer(self.data, dtype="<u2"), frame)
                if direction == "from":
                    words = np.frombuffer(self.data, dtype="<u4")
                    words[9:13] = [802, 858, 8559, COUNTER_FORMAT]

        class Kernel:
            def group_id(self, index):
                return index

            def __call__(self, frame_bo, result_bo):
                events.append("launch")
                return SimpleNamespace(wait=lambda: events.append("wait"))

        pyxrt = SimpleNamespace(
            device=lambda index: object(), xclbin=lambda path: object(),
            hw_context=lambda device, xclbin: object(),
            kernel=lambda context, name: Kernel(), bo=Buffer,
            xclBOSyncDirection=SimpleNamespace(
                XCL_BO_SYNC_BO_TO_DEVICE="to",
                XCL_BO_SYNC_BO_FROM_DEVICE="from"))
        with patch.dict("sys.modules", {"pyxrt": pyxrt}):
            results, cycles = run_kernel(Path("test.xclbin"), frame, 0)

        self.assertEqual(events, [
            ("sync", 1, "to"), ("sync", 0, "to"),
            "launch", "wait", ("sync", 1, "from")])
        self.assertEqual(cycles, CycleCounts(802, 858, 8559))
        self.assertEqual(results, [(0, 0, 0)] * 9)


if __name__ == "__main__":
    unittest.main()
