from pathlib import Path

from cocotb_tools.runner import get_runner


def test_conv_input_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "activation_map_pkg.vhd",
            rtl / "conv_input.vhd",
            here / "conv_input_wrapper.vhd",
        ],
        hdl_toplevel="conv_input_wrapper",
        build_args=["--std=08"],
        always=True,
        build_dir=here / "sim_build",
    )

    runner.test(
        hdl_toplevel="conv_input_wrapper",
        test_module="test_conv_input",
        waves=True,
    )


if __name__ == "__main__":
    test_conv_input_runner()
