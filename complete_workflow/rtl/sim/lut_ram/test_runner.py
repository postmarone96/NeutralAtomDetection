from pathlib import Path

from cocotb_tools.runner import get_runner


def test_lut_ram_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "act_ram.vhd",
            here / "tb_lut_ram.vhd",
        ],
        hdl_toplevel="tb_lut_ram",
        build_args=["--std=08"],
        always=True,
        build_dir=here / "sim_build",
    )

    runner.test(
        hdl_toplevel="tb_lut_ram",
        test_module="test_lut_ram",
        waves=True,
    )
