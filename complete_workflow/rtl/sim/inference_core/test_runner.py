import shutil
from pathlib import Path

from cocotb_tools.runner import get_runner


def test_inference_core_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1] / "src"
    build_dir = here / "sim_build"
    build_dir.mkdir(exist_ok=True)
    shutil.copy2(rtl / "weights.mem", build_dir / "weights.mem")
    shutil.copy2(rtl / "biases.mem", build_dir / "biases.mem")

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            rtl / "helpers_pkg.vhd",
            rtl / "nn_pkg.vhd",
            rtl / "activation_map_pkg.vhd",
            rtl / "weights_pkg.vhd",
            rtl / "bias_pkg.vhd",
            rtl / "act_ram.vhd",
            rtl / "conv_input.vhd",
            rtl / "dense_input.vhd",
            rtl / "weight_rom.vhd",
            rtl / "weights_handler.vhd",
            rtl / "bias_rom.vhd",
            rtl / "biases_handler.vhd",
            rtl / "mac_array.vhd",
            rtl / "requant_array.vhd",
            rtl / "max_pooling.vhd",
            rtl / "layer_output.vhd",
            rtl / "inference_controller.vhd",
            rtl / "inference_core.vhd",
            here / "inference_core_wrapper.vhd",
        ],
        hdl_toplevel="inference_core_wrapper",
        build_dir=build_dir,
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="inference_core_wrapper",
        test_module="test_inference_core",
    )


if __name__ == "__main__":
    test_inference_core_runner()
