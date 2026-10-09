import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_detector import run_profiled
from xrt_profile import summarize_native_trace


# Native XRT 2024.2 VTF: id,start_id,timestamp_ms,bucket,type,string_id.
TRACE = """HEADER
VTF File Version,1.0
Resolution,ms

STRUCTURE
Dynamic_Row,1,Native XRT API Calls,API Events
Dynamic_Row,2,Reads,Read Transfers
Dynamic_Row,3,Writes,Write Transfers

MAPPING
1,WRITE
2,READ
3,xrt::bo::sync

EVENTS
1,0,1.000000,3,API_CALL,1
2,1,1.100000,3,API_CALL,1
3,0,2.000000,3,API_CALL,1
7,0,2.000001,1,API_CALL,3
4,3,2.200000,3,API_CALL,1
8,7,2.200001,1,API_CALL,3
5,0,2.800000,2,API_CALL,2
6,5,2.900000,2,API_CALL,2

DEPENDENCIES
"""


class NativeTraceTests(unittest.TestCase):
    def summarize(self, trace):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native_trace.csv"
            path.write_text(trace)
            timing = summarize_native_trace(path)
            saved = json.loads(path.with_name("transfer_timing.json").read_text())
            self.assertEqual(saved["end_to_end_us"], timing.end_to_end_us)
            return timing

    def test_uses_upload_start_and_download_end_excluding_result_clear(self):
        timing = self.summarize(TRACE)
        self.assertEqual(timing.input_start_ms, 2.0)
        self.assertEqual(timing.input_end_ms, 2.2)
        self.assertEqual(timing.output_start_ms, 2.8)
        self.assertEqual(timing.output_end_ms, 2.9)
        self.assertEqual(timing.end_to_end_us, 900.0)

    def test_wrong_timestamp_units_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamps in ms"):
            self.summarize(TRACE.replace("Resolution,ms", "Resolution,ns"))

    def test_incomplete_transfer_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.summarize(TRACE.replace("6,5,2.900000,2,API_CALL,2\n", ""))

    def test_unmatched_end_rejected(self):
        with self.assertRaisesRegex(ValueError, "no matching start"):
            self.summarize(TRACE.replace("6,5,2.900000", "6,99,2.900000"))

    def test_extra_transfers_rejected_instead_of_guessing_buffer_identity(self):
        with self.assertRaisesRegex(ValueError, "expected result clear"):
            self.summarize(TRACE.replace("DEPENDENCIES", (
                "9,0,3.000000,3,API_CALL,1\n"
                "10,9,3.100000,3,API_CALL,1\nDEPENDENCIES")))

    def test_profile_worker_has_own_directory_and_explicit_xrt_config(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                profile_dir=Path(directory), xclbin=Path("test.xclbin"),
                frame=Path("frame.bin"), device=0, freq_mhz=400.0, frame_index=2)

            def worker(command, cwd, env, check):
                self.assertTrue(check)
                self.assertIn("--xrt-worker", command)
                self.assertEqual(command[-3:], ["--frame-index", "2", "--xrt-worker"])
                self.assertTrue(Path(command[2]).is_absolute())
                self.assertTrue(Path(command[3]).is_absolute())
                self.assertEqual(Path(env["XRT_INI_PATH"]).read_text(),
                                 "[Debug]\nnative_xrt_trace=true\n")
                (cwd / "native_trace.csv").write_text(TRACE)

            output = io.StringIO()
            with patch("run_detector.subprocess.run", side_effect=worker), \
                    contextlib.redirect_stdout(output):
                run_profiled(args)
            self.assertIn("900.000 us", output.getvalue())
            self.assertEqual(len(list(Path(directory).glob("single_*"))), 1)

    def test_missing_xrt_trace_is_reported_without_python_timer_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                profile_dir=Path(directory), xclbin=Path("test.xclbin"),
                frame=Path("frame.bin"), device=0, freq_mhz=400.0, frame_index=0)
            with patch("run_detector.subprocess.run"), \
                    self.assertRaisesRegex(RuntimeError, "XRT profiling"):
                run_profiled(args)


if __name__ == "__main__":
    unittest.main()
