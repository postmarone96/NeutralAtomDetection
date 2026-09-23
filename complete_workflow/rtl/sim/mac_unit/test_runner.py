from pathlib import Path

from cocotb_tools.runner import get_runner


def test_mac_unit_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")

    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "mac_array.vhd",
            here / "mac_array_wrapper.vhd",
        ],
        hdl_toplevel="mac_array_wrapper",
        parameters={
            "ACT_W_MAX": 10,
            "WEIGHT_W_MAX": 8,
            "ACC_W_MAX": 18,
            "MAC_UNITS": 3,
            "MAX_MAC_UNITS": 3,
        },
        build_args=["--std=08"],
        always=True,
        build_dir=here / "sim_build",
    )

    runner.test(
        hdl_toplevel="mac_array_wrapper",
        test_module="test_mac_unit",
        waves=True,
    )
