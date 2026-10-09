import shutil
from pathlib import Path

from cocotb_tools.runner import get_runner


def test_bias_rom_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"
    build_dir = here / "sim_build"
    rom_lines = [
        line.strip()
        for line in (rtl / "biases.mem").read_text().splitlines()
        if line.strip()
    ]
    rom_depth = len(rom_lines)
    word_length = len(rom_lines[0]) * 4

    build_dir.mkdir(exist_ok=True)
    shutil.copy2(rtl / "biases.mem", build_dir / "biases.mem")

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "bias_pkg.vhd",
            rtl / "bias_rom.vhd",
        ],
        hdl_toplevel="bias_rom",
        parameters={
            "ROM_DEPTH": rom_depth,
            "WORD_LENGTH": word_length,
            "FILE_NAME": "biases.mem",
        },
        build_args=["--std=08"],
        always=True,
        build_dir=build_dir,
    )
    runner.test(
        hdl_toplevel="bias_rom",
        test_module="test_bias_rom",
    )


if __name__ == "__main__":
    test_bias_rom_runner()
