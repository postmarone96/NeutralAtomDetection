#!/usr/bin/env bash
# Nine-site, single-CNN XRT build with a verified 400 MHz kernel clock.
set -eo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: bash impl/build_alveo_400.sh <platform.xpfm> [output-directory]" >&2
    exit 2
fi
platform="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rtl_root="$(cd "$script_dir/.." && pwd)"
xrt_root="$rtl_root/xrt"

# This installed U55C platform has already been linked successfully with 2024.2.
# Source before nounset because the vendor setup reads optional environment variables.
if [[ -f /opt/xilinx/vivado-2024.2/Vitis/2024.2/settings64.sh ]]; then
    source /opt/xilinx/vivado-2024.2/Vitis/2024.2/settings64.sh
elif ! command -v vivado >/dev/null 2>&1 || ! command -v v++ >/dev/null 2>&1; then
    echo "Source matching Vivado and Vitis environments before building." >&2
    exit 1
fi
set -u
vivado_version="$(vivado -version | sed -n 's/.*v\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
vitis_version="$(v++ --version | sed -n 's/.*v\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ -z "$vivado_version" || "$vivado_version" != "$vitis_version" ]]; then
    echo "Vivado ($vivado_version) and Vitis ($vitis_version) versions must match." >&2
    exit 1
fi
xclbinutil_tool="$(command -v xclbinutil || true)"
if [[ -z "$xclbinutil_tool" && -x /opt/xilinx/xrt/bin/xclbinutil ]]; then
    xclbinutil_tool=/opt/xilinx/xrt/bin/xclbinutil
fi
if [[ -z "$xclbinutil_tool" ]]; then
    echo "xclbinutil is required to verify the final runtime clock." >&2
    exit 1
fi
output_dir="${2:-$script_dir/alveo_400_single_cnn}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
if [[ -d "$output_dir/temp" || -f "$output_dir/neutral_atom_detector.xclbin" ]]; then
    echo "Output directory already contains a build; choose a fresh directory: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir/logs" "$output_dir/reports" "$output_dir/package"
echo "Building nine sites, one CNN + preprocessing + AXI/XRT at 400 MHz"
echo "Tools: Vivado/Vitis $vivado_version"
echo "Output: $output_dir"
bash "$xrt_root/scripts/package_xo.sh" "$output_dir/package"
python3 - "$output_dir/package/source_files.txt" "$output_dir/source_manifest.sha256" <<'PY'
import hashlib
import sys
from pathlib import Path
paths = Path(sys.argv[1]).read_text().splitlines()
with Path(sys.argv[2]).open("w") as output:
    for name in paths:
        output.write(f"{hashlib.sha256(Path(name).read_bytes()).hexdigest()}  {name}\n")
PY
xo="$output_dir/package/neutral_atom_kernel.xo"
xclbin="$output_dir/neutral_atom_detector.xclbin"
v++ --link --target hw --platform "$platform" \
    --config "$xrt_root/config/link.cfg" \
    --kernel_frequency 0:400 \
    --vivado.prop "run.impl_1.STEPS.INIT_DESIGN.TCL.PRE=$xrt_root/scripts/prepare_clock_400.tcl" \
    --vivado.prop "run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=$xrt_root/scripts/check_kernel_clock_400.tcl" \
    --vivado.prop "run.impl_1.STEPS.PLACE_DESIGN.TCL.POST=$xrt_root/scripts/report_placed_400.tcl" \
    --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore \
    --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true \
    --vivado.prop run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=AggressiveExplore \
    --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore \
    --save-temps --temp_dir "$output_dir/temp" --report_dir "$output_dir/reports" \
    --log_dir "$output_dir/logs" --output "$xclbin" "$xo"
if [[ ! -f "$xclbin" ]]; then echo "No XCLBIN was produced." >&2; exit 1; fi
"$xclbinutil_tool" --info --input "$xclbin" > "$output_dir/xclbin_info.txt"
python3 "$xrt_root/scripts/check_xclbin_clock_400.py" "$output_dir/xclbin_info.txt"
routed_dcp="$(find "$output_dir/temp" -type f -path '*/impl_1/*_routed.dcp' -print | head -n 1 || true)"
if [[ -z "$routed_dcp" ]]; then
    routed_dcp="$(find "$output_dir/temp" -type f -name '*_routed.dcp' -print | head -n 1 || true)"
fi
if [[ -z "$routed_dcp" ]]; then
    echo "Final routed checkpoint is missing; cannot verify complete-design timing." >&2
    exit 1
fi
vivado -mode batch -journal "$output_dir/logs/report_routed.jou" -log "$output_dir/logs/report_routed.log" \
    -source "$script_dir/report_routed_xclbin_400.tcl" -tclargs "$routed_dcp" "$output_dir/reports/final"
cat > "$output_dir/build_summary.txt" <<EOF
top=neutral_atom_kernel
platform=$platform
frequency_mhz=400
compute_units=1
site_count=9
vivado_version=$vivado_version
vitis_version=$vitis_version
runtime_clock_verified=1
routed_timing_pass=1
contents=AXI4_Lite_plus_AXI4_HBM_plus_preprocessing_plus_neighbor_CNN
xclbin=$xclbin
EOF
echo "XCLBIN ready, with verified 400 MHz runtime clock and passing routed timing: $xclbin"
