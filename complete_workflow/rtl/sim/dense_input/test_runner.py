from pathlib import Path

from cocotb_tools.runner import get_runner


def test_dense_input_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "weights_pkg.vhd",
            rtl / "dense_input.vhd",
            here / "dense_input_wrapper.vhd",
        ],
        hdl_toplevel="dense_input_wrapper",
        build_dir=here / "sim_build",
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="dense_input_wrapper",
        test_module="test_dense_input",
    )


if __name__ == "__main__":
    test_dense_input_runner()
