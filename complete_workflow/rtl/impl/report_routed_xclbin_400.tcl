if {[llength $argv] != 2} {
    error "usage: report_routed_xclbin.tcl <routed-checkpoint> <report-directory>"
}

set checkpoint [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
file mkdir $report_dir

open_checkpoint $checkpoint
source [file join [file dirname [file normalize [info script]]] .. xrt scripts check_kernel_clock_400.tcl]

report_route_status \
    -file [file join $report_dir route_status.rpt]
report_utilization \
    -file [file join $report_dir utilization_routed.rpt]
report_utilization -hierarchical -hierarchical_depth 12 \
    -file [file join $report_dir utilization_hierarchical_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 100 -report_unconstrained \
    -file [file join $report_dir timing_routed.rpt]
report_clock_utilization \
    -file [file join $report_dir clock_utilization_routed.rpt]
report_methodology \
    -file [file join $report_dir methodology.rpt]

set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup] || ![llength $hold]} { error "Final timing paths are missing" }
set wns [get_property SLACK [lindex $setup 0]]
set whs [get_property SLACK [lindex $hold 0]]
set summary [open [file join $report_dir timing_summary.txt] w]
puts $summary "setup_slack_ns=$wns"
puts $summary "hold_slack_ns=$whs"
close $summary
if {$wns < 0 || $whs < 0} {
    error "The complete XRT design failed routed timing: setup=$wns ns, hold=$whs ns. Reports saved."
}

close_design
puts "Final routed XCLBIN reports written to $report_dir"
