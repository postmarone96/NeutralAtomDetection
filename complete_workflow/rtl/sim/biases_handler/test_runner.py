import shutil
from pathlib import Path

from cocotb_tools.runner import get_runner


def test_biases_handler_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"
    build_dir = here / "sim_build"
    build_dir.mkdir(exist_ok=True)
    shutil.copy2(rtl / "biases.mem", build_dir / "biases.mem")

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "activation_map_pkg.vhd",
            rtl / "bias_pkg.vhd",
            rtl / "bias_rom.vhd",
            rtl / "biases_handler.vhd",
            here / "biases_handler_wrapper.vhd",
        ],
        hdl_toplevel="biases_handler_wrapper",
        build_dir=build_dir,
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="biases_handler_wrapper",
        test_module="test_biases_handler",
    )


if __name__ == "__main__":
    test_biases_handler_runner()
