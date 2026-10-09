#!/usr/bin/env python3
"""Validate eight frames on the 3x3 kernel and save XRT measurements."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xclbin", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = Path(__file__).resolve().parent
    vectors = host.parents[1] / "tb" / "vectors"
    frame_path = vectors / "raw_frame.txt"
    reference_path = vectors / "model_classifier_4.txt"
    decision_path = vectors / "decisions.txt"
    logits = [int(token, 16) for token in reference_path.read_text().split()]
    logits = [value - 256 if value & 128 else value for value in logits]
    decisions = [int(token) for token in decision_path.read_text().split()]
    frames = frame_path.read_text().split()
    if (len(frames), len(logits), len(decisions)) != (8 * 784, 8 * 18, 8 * 9):
        raise ValueError("expected the eight-frame 3x3 golden-vector set")
    args.output.mkdir(parents=True, exist_ok=True)
    measurements = []
    for image in range(8):
        command = [sys.executable, str(host / "run_detector.py"),
                   str(args.xclbin.resolve()), str(frame_path),
                   "--frame-index", str(image), "--device", "0", "--freq-mhz", "400",
                   "--profile-dir", str((args.output / "profiles").resolve())]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        (args.output / f"frame_{image}.log").write_text(result.stdout)
        print(f"Frame {image}:\n{result.stdout}", flush=True)
        result.check_returncode()
        records = re.findall(r"site (\d+): logits=\((-?\d+), (-?\d+)\) decision=([01])",
                             result.stdout)
        if [int(record[0]) for record in records] != list(range(9)):
            raise ValueError(f"frame {image}: missing or reordered site results")
        actual = [tuple(map(int, record[1:])) for record in records]
        expected = [(logits[image * 18 + site * 2], logits[image * 18 + site * 2 + 1],
                     decisions[image * 9 + site]) for site in range(9)]
        if actual != expected:
            raise ValueError(f"frame {image}: predictions differ from golden vectors")
        counts = []
        for label in ("Preprocessing", "CNN", "Pipeline"):
            match = re.search(rf"^{label}[^\n]*: (\d+) cycles =", result.stdout, re.MULTILINE)
            if match is None:
                raise ValueError(f"frame {image}: missing {label} counter")
            counts.append(int(match[1]))
        if counts != [802, 858, 8559]:
            raise ValueError(f"frame {image}: unexpected hardware counters {counts}")
        profile = re.search(r"^XRT profiling reports: (.+)$", result.stdout, re.MULTILINE)
        if profile is None:
            raise ValueError(f"frame {image}: missing XRT profiling output")
        timing = json.loads((Path(profile[1]) / "transfer_timing.json").read_text())
        measurements.append({"frame_index": image, "predictions": actual,
                             "preprocess_cycles": counts[0], "cnn_cycles": counts[1],
                             "pipeline_cycles": counts[2], "xrt_timing": timing})
    platform = subprocess.run(
        ["xrt-smi", "examine", "--device", "0000:01:00.1", "--report", "platform"],
        capture_output=True, text=True, check=True, timeout=30).stdout
    (args.output / "platform.txt").write_text(platform)
    if not re.search(r"DATA_CLK[^\n]*:\s*400 MHz", platform):
        raise ValueError("device DATA_CLK was not reported as 400 MHz")
    payload = {"measured_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
               "design": "3x3, one CNN, one pixel per clock",
               "frequency_mhz": 400, "device": 0,
               "xclbin": str(args.xclbin.resolve()), "xclbin_sha256": sha256(args.xclbin),
               "input_sha256": sha256(frame_path), "reference_sha256": sha256(reference_path),
               "golden_frames_passed": 8, "golden_predictions_passed": 72,
               "measurements": measurements}
    (args.output / "measurements.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("PASS: 72 golden predictions, hardware counters, XRT timing, and 400 MHz device clock.")


if __name__ == "__main__":
    main()
