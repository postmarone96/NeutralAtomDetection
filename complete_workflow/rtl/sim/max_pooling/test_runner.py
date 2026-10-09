from pathlib import Path

from cocotb_tools.runner import get_runner


def test_max_pooling_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "activation_map_pkg.vhd",
            rtl / "max_pooling.vhd",
            here / "max_pooling_wrapper.vhd",
        ],
        hdl_toplevel="max_pooling_wrapper",
        build_dir=here / "sim_build",
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="max_pooling_wrapper",
        test_module="test_max_pooling",
        plusargs=["--ieee-asserts=disable-at-0"],
    )


if __name__ == "__main__":
    test_max_pooling_runner()
