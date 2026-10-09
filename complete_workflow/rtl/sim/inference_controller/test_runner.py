from pathlib import Path

from cocotb_tools.runner import get_runner


def test_inference_controller_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "nn_pkg.vhd",
            rtl / "activation_map_pkg.vhd",
            rtl / "weights_pkg.vhd",
            rtl / "inference_controller.vhd",
            here / "inference_controller_wrapper.vhd",
        ],
        hdl_toplevel="inference_controller_wrapper",
        build_dir=here / "sim_build",
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="inference_controller_wrapper",
        test_module="test_inference_controller",
    )


if __name__ == "__main__":
    test_inference_controller_runner()
