from pathlib import Path
from cocotb_tools.runner import get_runner

def test_requant_runner():
    here = Path(__file__).resolve().parent
    here_parent = Path(here.parent).resolve().parent
    rtl = here_parent.parent / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[rtl / "requant.vhd",],
        hdl_toplevel="requant",
        parameters={
            "IN_W": 16,
            "IN_I": 6,
            "OUT_W": 8,
            "OUT_I": 4,
        },
        build_args=["--std=08"],
        always=True,
    )

    runner.test(
        hdl_toplevel="requant",
        test_module="test_requant"
    )

if __name__ == "__main__":
    test_requant_runner()
