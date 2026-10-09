#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v vivado >/dev/null 2>&1; then
    if [[ -f /opt/xilinx/vivado-2025.2/2025.2/Vitis/settings64.sh ]]; then
        source /opt/xilinx/vivado-2025.2/2025.2/Vitis/settings64.sh
    elif [[ -f /opt/xilinx/vivado-2024.2/Vitis/2024.2/settings64.sh ]]; then
        source /opt/xilinx/vivado-2024.2/Vitis/2024.2/settings64.sh
    else
        echo "Vivado settings were not found." >&2
        exit 1
    fi
fi

set -u
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [build-directory]" >&2
    exit 2
fi
build_dir="${1:-$script_dir/../build}"
mkdir -p "$build_dir"
build_dir="$(cd "$build_dir" && pwd)"
vivado -mode batch -journal "$build_dir/package.jou" -log "$build_dir/package.log" \
    -source "$script_dir/package_kernel.tcl" -tclargs "$build_dir"
