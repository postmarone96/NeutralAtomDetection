import shutil
from pathlib import Path

from cocotb_tools.runner import get_runner


def test_full_neutral_atom_kernel_runner():
    here = Path(__file__).resolve().parent
    rtl = here.parents[1]
    sources = rtl / "src"
    build_dir = here / "full_sim_build"
    build_dir.mkdir(exist_ok=True)
    shutil.copy2(sources / "weights.mem", build_dir / "weights.mem")
    shutil.copy2(sources / "biases.mem", build_dir / "biases.mem")

    runner = get_runner("ghdl")
    runner.build(
        sources=[
            sources / "helpers_pkg.vhd",
            sources / "nn_pkg.vhd",
            sources / "preprocess_pkg.vhd",
            sources / "activation_map_pkg.vhd",
            sources / "weights_pkg.vhd",
            sources / "bias_pkg.vhd",
            sources / "preprocess.vhd",
            sources / "act_ram.vhd",
            sources / "conv_input.vhd",
            sources / "dense_input.vhd",
            sources / "weight_rom.vhd",
            sources / "weights_handler.vhd",
            sources / "bias_rom.vhd",
            sources / "biases_handler.vhd",
            sources / "mac_array.vhd",
            sources / "requant_array.vhd",
            sources / "max_pooling.vhd",
            sources / "layer_output.vhd",
            sources / "inference_controller.vhd",
            sources / "inference_core.vhd",
            sources / "detector_core.vhd",
            here.parent / "src" / "neutral_atom_kernel.vhd",
        ],
        hdl_toplevel="neutral_atom_kernel",
        build_dir=build_dir,
        build_args=["--std=08"],
        always=True,
    )
    runner.test(
        hdl_toplevel="neutral_atom_kernel",
        test_module="test_full_neutral_atom_kernel",
        plusargs=["--ieee-asserts=disable-at-0"],
    )


if __name__ == "__main__":
    test_full_neutral_atom_kernel_runner()
