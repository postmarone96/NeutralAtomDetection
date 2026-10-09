# Verify clock discovery, imported helper replacement and refusal to downclock.
# Vivado placement and hardware clock programming still require the remote build.
namespace eval ocl_util {
    namespace export write_user_impl_clock_constraint report_timing_and_scale_freq
    proc write_user_impl_clock_constraint {args} { error "Broken vendor calculator called" }
    proc report_timing_and_scale_freq {inst design out vpl info binary {is_in_run true}} {
        set ::observed_info $info
        set ::observed_is_in_run $is_in_run
        if {$::negative_slack && ![dict get $info enable_auto_freq_scale]} {
            error "design did not meet timing"
        }
        return true
    }
}
# Match Vitis: the helpers have been imported before the user init hook runs.
namespace import ocl_util::*
proc current_project {} { return Project }
proc get_property {property object} {
    switch $property {
        PART { return $::part }
        NAME { return $object }
        SOURCE_PINS { return $::mmcm_outputs }
        SOURCE { return mmcm/CLKIN1 }
        PERIOD { return $::reference_period }
        default { error "Unexpected property $property" }
    }
}
proc get_pins {args} {
    set name [lindex $args end]
    if {$::missing_port && $name eq "ulp/ucs/aclk_kernel_00"} { return {} }
    return $name
}
proc get_clocks {args} {
    switch [lindex $args end] {
        ulp/ucs/aclk_kernel_00 - mmcm/CLKOUT0 { return kernel_clock }
        mmcm/CLKIN1 { return reference_clock }
        default { return {} }
    }
}
proc create_generated_clock {args} {
    set ::generated_arguments $args
    set factor [lindex $args [expr {[lsearch -exact $args -multiply_by] + 1}]]
    set ::generated_period [expr {$::reference_period / $factor}]
}
proc read_xdc {path} { uplevel #0 [list source $path] }
set here [file dirname [file normalize [info script]]]
set hook [file normalize [file join $here .. scripts prepare_clock_400.tcl]]
source $hook
source $hook
set part xcu55c-fsvh2892-2L-e
set reference_period 10.0
set mmcm_outputs mmcm/CLKOUT0
set missing_port false
set negative_slack false
set output [file join /tmp single_fixed_clock_test]
file mkdir $output
set requests [dict create ucs/aclk_kernel_00 [dict create is_user_set true clk_id 0 freq 400] \
    ucs/aclk_kernel_01 [dict create is_user_set false clk_id 1 freq 500]]
write_user_impl_clock_constraint ulp $requests "" $output false
if {$generated_period != 2.5} { error "Wrong generated clock period" }
if {[lindex $generated_arguments end] ne "mmcm/CLKOUT0"} { error "Wrong clock target" }
puts "PASS: imported helper bypasses the failed calculator and constrains the MMCM to 400 MHz"
foreach {variable value expected} {
    reference_period 8.0 {Expected the U55C 100 MHz reference*}
    mmcm_outputs {mmcm/CLKOUT0 other/CLKOUT0} {Expected one kernel MMCM output pin}
    missing_port true {Expected one platform kernel clock*}
    part wrong_part {The fixed 400 MHz workaround supports only the U55C platform}
} {
    set saved [set $variable]
    set $variable $value
    if {![catch {write_user_impl_clock_constraint ulp $requests "" $output false} message]
        || ![string match $expected $message]} { error "Clock guard failed: $variable=$value: $message" }
    set $variable $saved
}
set bad_requests $requests
dict set bad_requests ucs/aclk_kernel_00 freq 399
if {![catch {write_user_impl_clock_constraint ulp $bad_requests "" $output false}]} {
    error "A different kernel frequency was accepted"
}
set info [dict create enable_auto_freq_scale true skip_timing_and_scaling true \
    error_on_hold_violation false worst_negative_slack -1]
report_timing_and_scale_freq ulp design $output $output $info binary
foreach {key expected} {
    enable_auto_freq_scale false skip_timing_and_scaling false
    error_on_hold_violation true worst_negative_slack 0
} {
    if {[dict get $observed_info $key] ne $expected} { error "Timing policy was not enforced: $key" }
}
if {!$observed_is_in_run} { error "Default timing-helper argument was lost" }
set negative_slack true
if {![catch {report_timing_and_scale_freq ulp design $output $output $info binary false} message]
    || $message ne "design did not meet timing"} { error "Timing failure was hidden" }
if {$observed_is_in_run} { error "Explicit timing-helper argument was lost" }
puts "PASS: invalid clock setups rejected; timing checks enforced; automatic frequency reduction disabled"
