"""Reject a single-CNN XCLBIN whose DATA clock metadata differs from 400 MHz."""
import re
import sys
from pathlib import Path


def check_clock(text):
    frequencies = re.findall(
        r"Type:[ \t]*DATA[ \t]*\r?\n[ \t]*Frequency:[ \t]*(\d+(?:\.\d+)?)[ \t]*MHz",
        text,
    )
    if len(frequencies) != 1 or float(frequencies[0]) != 400.0:
        raise ValueError(f"Expected one DATA clock at 400 MHz; found {frequencies}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: check_xclbin_clock_400.py <xclbin_info.txt>")
    try:
        check_clock(Path(sys.argv[1]).read_text())
    except (OSError, ValueError) as error:
        sys.exit(str(error))
    print("Verified XCLBIN runtime DATA clock: 400 MHz")
