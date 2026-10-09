#!/usr/bin/env bash
# Build the 3x3 reader, then validate it on the Alveo at 400 MHz.
set -eo pipefail
if [[ $# -gt 1 ]]; then
    echo "usage: bash impl/run_alveo_400.sh [output-directory]" >&2
    exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl_root="$(cd "$script_dir/.." && pwd)"
output_dir="${1:-$script_dir/alveo_400_single_cnn}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
platform=/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
cd "$rtl_root"
set +e
bash "$script_dir/build_alveo_400.sh" "$platform" "$output_dir" \
    2>&1 | tee "$output_dir/console.log"
build_status=${PIPESTATUS[0]}
printf '%s\n' "$build_status" > "$output_dir/build_exit_code.txt"
set -e
if [[ "$build_status" -ne 0 ]]; then
    echo "Build failed; hardware validation has not been run."
    exit "$build_status"
fi
source /opt/xilinx/xrt/setup.sh
set +e
python3 -u "$rtl_root/xrt/host/validate_hardware.py" \
    "$output_dir/neutral_atom_detector.xclbin" --output "$output_dir/hardware" \
    2>&1 | tee "$output_dir/hardware_run.log"
hardware_status=${PIPESTATUS[0]}
printf '%s\n' "$hardware_status" > "$output_dir/hardware_exit_code.txt"
exit "$hardware_status"
