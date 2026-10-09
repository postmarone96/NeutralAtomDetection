from pathlib import Path

from cocotb_tools.runner import get_runner


def test_layer_output_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "activation_map_pkg.vhd",
            rtl / "weights_pkg.vhd",
            rtl / "layer_output.vhd",
            here / "layer_output_wrapper.vhd",
        ],
        hdl_toplevel="layer_output_wrapper",
        build_dir=here / "sim_build",
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="layer_output_wrapper",
        test_module="test_layer_output",
    )


if __name__ == "__main__":
    test_layer_output_runner()
