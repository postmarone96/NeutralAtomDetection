set script_dir [file dirname [file normalize [info script]]]
set rtl_root [file normalize [file join $script_dir ..]]

if {[llength $argv] ni {3 4}} {
    error "usage: build_core.tcl <output-directory> <frequency-MHz> <synth-only:0|1> \[Default|Performance_Explore\]"
}

set output_dir [file normalize [lindex $argv 0]]
set report_dir [file join $output_dir reports]
set frequency_mhz [lindex $argv 1]
set synth_only [lindex $argv 2]
set implementation_strategy Default
if {[llength $argv] == 4} {
    set implementation_strategy [lindex $argv 3]
}
if {![string is integer -strict $frequency_mhz] || $frequency_mhz <= 0} {
    error "frequency must be a positive integer in MHz"
}
if {$synth_only ne "0" && $synth_only ne "1"} {
    error "synth-only must be 0 or 1"
}
if {$implementation_strategy ni {Default Performance_Explore}} {
    error "unsupported implementation strategy: $implementation_strategy"
}
set clock_period_ns [expr {1000.0 / double($frequency_mhz)}]

file mkdir $output_dir $report_dir

# Never leave reports from an older successful run beside a failed new run.
foreach old_report [glob -nocomplain [file join $report_dir *.rpt]] {
    file delete -force $old_report
}
foreach old_artifact [list \
    [file join $output_dir detector_core_synth.dcp] \
    [file join $output_dir detector_core_routed.dcp] \
    [file join $output_dir detector_core_analysis_only.bit] \
    [file join $output_dir build_summary.txt] \
] {
    file delete -force $old_artifact
}

# ROM initialization filenames are relative to rtl/src.
cd [file join $rtl_root src]

set source_files [list \
    [file join $rtl_root src helpers_pkg.vhd] \
    [file join $rtl_root src nn_pkg.vhd] \
    [file join $rtl_root src preprocess_pkg.vhd] \
    [file join $rtl_root src activation_map_pkg.vhd] \
    [file join $rtl_root src weights_pkg.vhd] \
    [file join $rtl_root src bias_pkg.vhd] \
    [file join $rtl_root src preprocess.vhd] \
    [file join $rtl_root src act_ram.vhd] \
    [file join $rtl_root src conv_input.vhd] \
    [file join $rtl_root src dense_input.vhd] \
    [file join $rtl_root src weight_rom.vhd] \
    [file join $rtl_root src weights_handler.vhd] \
    [file join $rtl_root src bias_rom.vhd] \
    [file join $rtl_root src biases_handler.vhd] \
    [file join $rtl_root src mac_array.vhd] \
    [file join $rtl_root src requant_array.vhd] \
    [file join $rtl_root src max_pooling.vhd] \
    [file join $rtl_root src layer_output.vhd] \
    [file join $rtl_root src inference_controller.vhd] \
    [file join $rtl_root src inference_core.vhd] \
    [file join $rtl_root src detector_core.vhd] \
]

foreach source_file $source_files {
    if {![file exists $source_file]} {
        error "missing RTL source: $source_file"
    }
    read_vhdl -vhdl2008 $source_file
}

synth_design \
    -top detector_core \
    -part xcu55c-fsvh2892-2L-e \
    -flatten_hierarchy rebuilt

create_clock -name clk -period $clock_period_ns [get_ports clk]

report_utilization \
    -file [file join $report_dir utilization_synth.rpt]
report_utilization -hierarchical -hierarchical_depth 8 \
    -file [file join $report_dir utilization_hierarchical_synth.rpt]
report_timing_summary -delay_type min_max -max_paths 100 -report_unconstrained \
    -file [file join $report_dir timing_synth.rpt]
write_checkpoint -force [file join $output_dir detector_core_synth.dcp]

if {$synth_only eq "1"} {
    set summary [open [file join $output_dir build_summary.txt] w]
    puts $summary "top=detector_core"
    puts $summary "part=xcu55c-fsvh2892-2L-e"
    puts $summary "frequency_mhz=$frequency_mhz"
    puts $summary "clock_period_ns=$clock_period_ns"
    puts $summary "stage=synthesis_only"
    puts $summary "implementation_strategy=$implementation_strategy"
    puts $summary "vivado_version=[version -short]"
    puts $summary "purpose=preprocessing_plus_CNN_resource_and_timing_check"
    close $summary

    puts "Core synthesis reports and checkpoint written to $output_dir"
    exit 0
}

if {$implementation_strategy eq "Performance_Explore"} {
    # AMD Performance_Explore directives, starting from a fresh synthesis.
    opt_design -directive Explore
    place_design -directive Explore
    phys_opt_design -directive Explore
    route_design -directive Explore
} else {
    opt_design
    place_design
    phys_opt_design
    route_design
}

report_route_status \
    -file [file join $report_dir route_status.rpt]
report_utilization \
    -file [file join $report_dir utilization_routed.rpt]
report_utilization -hierarchical -hierarchical_depth 8 \
    -file [file join $report_dir utilization_hierarchical_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 100 -report_unconstrained \
    -file [file join $report_dir timing_routed.rpt]
report_clock_utilization \
    -file [file join $report_dir clock_utilization_routed.rpt]
report_methodology \
    -file [file join $report_dir methodology.rpt]
set setup_wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold_whs [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]

write_checkpoint -force [file join $output_dir detector_core_routed.dcp]

# This top level has no physical board pin constraints because it is only used
# to measure the detector itself.  Lowering these two DRCs permits Vivado to
# emit an analysis artifact, but that artifact is not an Alveo load image.
set_property SEVERITY Warning [get_drc_checks NSTD-1]
set_property SEVERITY Warning [get_drc_checks UCIO-1]
write_bitstream -force [file join $output_dir detector_core_analysis_only.bit]

set summary [open [file join $output_dir build_summary.txt] w]
puts $summary "top=detector_core"
puts $summary "part=xcu55c-fsvh2892-2L-e"
puts $summary "frequency_mhz=$frequency_mhz"
puts $summary "clock_period_ns=$clock_period_ns"
puts $summary "implementation_strategy=$implementation_strategy"
puts $summary "vivado_version=[version -short]"
puts $summary "setup_wns_ns=$setup_wns"
puts $summary "hold_whs_ns=$hold_whs"
puts $summary "internal_setup_and_hold_pass=[expr {$setup_wns >= 0 && $hold_whs >= 0}]"
puts $summary "purpose=preprocessing_plus_CNN_resource_and_timing_comparison"
puts $summary "bitstream=analysis_only_not_for_Alveo_programming"
close $summary

puts "Core reports and artifacts written to $output_dir"
puts "Setup WNS: $setup_wns ns; hold WHS: $hold_whs ns"
