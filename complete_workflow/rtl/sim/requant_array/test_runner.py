from pathlib import Path

from cocotb_tools.runner import get_runner


def test_requant_array_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "requant_array.vhd",
            here / "requant_array_wrapper.vhd",
        ],
        hdl_toplevel="requant_array_wrapper",
        build_dir=here / "sim_build",
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="requant_array_wrapper",
        test_module="test_requant_array",
    )


if __name__ == "__main__":
    test_requant_array_runner()
