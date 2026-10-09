#!/usr/bin/env bash
# Implement the preprocessing + CNN detector without the Alveo AXI/XRT shell.
#
# This build is intended for algorithm/core comparisons.  It produces routed
# timing and utilization reports, a routed checkpoint, and an analysis-only
# full-device bitstream.  The .bit file must not be loaded onto an Alveo card;
# use build_alveo_400.sh and its .xclbin for that.
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$script_dir/core"
frequency_mhz="${FREQ_MHZ:-400}"
synth_only="${SYNTH_ONLY:-0}"
implementation_strategy="${CORE_IMPL_STRATEGY:-Default}"

if [[ $# -gt 1 ]]; then
    echo "usage: FREQ_MHZ=400 $0 [output-directory]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    output_dir="$(mkdir -p "$1" && cd "$1" && pwd)"
fi
if [[ ! "$frequency_mhz" =~ ^[1-9][0-9]*$ ]]; then
    echo "FREQ_MHZ must be a positive integer; got '$frequency_mhz'." >&2
    exit 2
fi
if [[ "$synth_only" != "0" && "$synth_only" != "1" ]]; then
    echo "SYNTH_ONLY must be 0 or 1; got '$synth_only'." >&2
    exit 2
fi
case "$implementation_strategy" in
    Default|Performance_Explore) ;;
    *) echo "CORE_IMPL_STRATEGY must be Default or Performance_Explore; got '$implementation_strategy'." >&2; exit 2 ;;
esac

# Xilinx setup scripts inspect optional variables before defining them, so do
# not enable nounset until after the tool environment has been loaded.
if ! command -v vivado >/dev/null 2>&1; then
    if [[ -f /opt/xilinx/vivado-2024.2/Vitis/2024.2/settings64.sh ]]; then
        source /opt/xilinx/vivado-2024.2/Vitis/2024.2/settings64.sh
    else
        echo "Vivado/Vitis settings64.sh was not found." >&2
        exit 1
    fi
fi

set -u
mkdir -p "$output_dir/logs" "$output_dir/reports" "$output_dir/work"

if [[ "$synth_only" == "1" ]]; then
    echo "Synthesizing detector_core at ${frequency_mhz} MHz"
else
    echo "Implementing detector_core at ${frequency_mhz} MHz"
fi
echo "Output: $output_dir"
echo "Implementation strategy: $implementation_strategy"

cd "$output_dir/work"
vivado -mode batch \
    -journal "$output_dir/logs/vivado.jou" \
    -log "$output_dir/logs/vivado.log" \
    -source "$script_dir/build_core.tcl" \
    -tclargs "$output_dir" "$frequency_mhz" "$synth_only" "$implementation_strategy"

echo
if [[ "$synth_only" == "1" ]]; then
    echo "Core synthesis complete."
    echo "  Timing:      $output_dir/reports/timing_synth.rpt"
    echo "  Utilization: $output_dir/reports/utilization_synth.rpt"
    echo "  Hierarchy:   $output_dir/reports/utilization_hierarchical_synth.rpt"
    echo "  Checkpoint:  $output_dir/detector_core_synth.dcp"
else
    echo "Core implementation complete."
    echo "  Timing:      $output_dir/reports/timing_routed.rpt"
    echo "  Utilization: $output_dir/reports/utilization_routed.rpt"
    echo "  Hierarchy:   $output_dir/reports/utilization_hierarchical_routed.rpt"
    echo "  Checkpoint:  $output_dir/detector_core_routed.dcp"
    echo "  Analysis bitstream: $output_dir/detector_core_analysis_only.bit"
    echo "Do not program the analysis-only .bit onto an Alveo card."
fi
