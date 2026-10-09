from pathlib import Path

from cocotb_tools.runner import get_runner


def test_neutral_atom_kernel_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1]
    sources = rtl / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            sources / "helpers_pkg.vhd",
            sources / "nn_pkg.vhd",
            here / "detector_core_stub.vhd",
            here.parent / "src" / "neutral_atom_kernel.vhd",
        ],
        hdl_toplevel="neutral_atom_kernel",
        build_dir=here / "sim_build",
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="neutral_atom_kernel",
        test_module="test_neutral_atom_kernel",
    )


if __name__ == "__main__":
    test_neutral_atom_kernel_runner()
