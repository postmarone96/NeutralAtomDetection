from pathlib import Path
import shutil

from cocotb_tools.runner import get_runner


def test_weights_handler():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"
    build_dir = here / "sim_build"
    build_dir.mkdir(exist_ok=True)
    shutil.copy2(rtl / "weights.mem", build_dir / "weights.mem")

    sources = [
        rtl / "helpers_pkg.vhd",
        rtl / "nn_pkg.vhd",
        rtl / "activation_map_pkg.vhd",
        rtl / "weights_pkg.vhd",
        rtl / "weight_rom.vhd",
        rtl / "weights_handler.vhd",
        here / "weights_handler_wrapper.vhd",
    ]

    runner = get_runner("ghdl")
    runner.build(
        sources=sources,
        hdl_toplevel="weights_handler_wrapper",
        build_dir=build_dir,
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="weights_handler_wrapper",
        test_module="test_weights_handler_new",
        test_args=["--std=08"],
    )


if __name__ == "__main__":
    test_weights_handler()
