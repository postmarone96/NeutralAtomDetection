# U55C clock 0 is scalable: XRT programs its frequency from the XCLBIN.
# Replace only the failing private Clocking Wizard calculation in Vitis 2024.2.
# Keep the generated clock on the platform MMCM and retain Vitis timing checks.
if {![info exists ::single_fixed_clock_installed]} {
    foreach command {write_user_impl_clock_constraint report_timing_and_scale_freq} {
        if {![llength [info commands ::ocl_util::$command]]} {
            error "Vitis clock helper was not loaded before prepare_clock_400.tcl: $command"
        }
    }

    proc ::ocl_util::write_user_impl_clock_constraint {inst dict_clock_freqs steps_log vivado_output_dir is_versal} {
        if {$is_versal || [get_property PART [current_project]] ne "xcu55c-fsvh2892-2L-e"} {
            error "The fixed 400 MHz workaround supports only the U55C platform"
        }
        set constraints {}
        foreach clock_name [dict keys $dict_clock_freqs] {
            set request [dict get $dict_clock_freqs $clock_name]
            if {![dict get $request is_user_set]} { continue }
            if {[dict get $request clk_id] != 0 || [dict get $request freq] != 400} {
                error "Single-CNN build requires scalable kernel clock 0 at 400 MHz"
            }
            set port [get_pins -quiet "$inst/$clock_name"]
            set clocks [get_clocks -quiet -of_objects $port]
            if {[llength $port] != 1 || [llength $clocks] != 1} {
                error "Expected one platform kernel clock at $inst/$clock_name"
            }
            set output_pin [get_pins -quiet [get_property SOURCE_PINS $clocks]]
            if {[llength $output_pin] != 1} { error "Expected one kernel MMCM output pin" }
            set generated [get_clocks -quiet -of_objects $output_pin]
            if {[llength $generated] != 1} { error "Expected one MMCM generated clock" }
            set source_pin [get_pins -quiet [get_property SOURCE $generated]]
            set reference [get_clocks -quiet -of_objects $source_pin]
            if {[llength $source_pin] != 1 || [llength $reference] != 1} {
                error "Expected one MMCM reference clock"
            }
            set period [get_property PERIOD $reference]
            if {![string is double -strict $period] || abs($period - 10.0) > 0.001} {
                error "Expected the U55C 100 MHz reference (10 ns), found $period ns"
            }
            lappend constraints [list create_generated_clock \
                -name [get_property NAME $generated] -divide_by 1 -multiply_by 4 \
                -source [get_property NAME $source_pin] [get_property NAME $output_pin]]
        }
        if {[llength $constraints] != 1} { error "Exactly one 400 MHz kernel clock request is required" }
        set path [file join $vivado_output_dir _user_impl_clk.xdc]
        set channel [open $path w]
        set status [catch {
            puts $channel "# Fixed U55C scalable kernel clock: 100 MHz reference -> 400 MHz"
            puts $channel [lindex $constraints 0]
        } result options]
        close $channel
        if {$status} { return -options $options $result }
        read_xdc $path
        puts "Applied platform kernel clock constraint: 2.5 ns (400 MHz)"
    }

    # Copy, then redefine in place so existing namespace-import aliases keep working.
    set arguments {}
    foreach argument [info args ::ocl_util::report_timing_and_scale_freq] {
        if {[info default ::ocl_util::report_timing_and_scale_freq $argument default]} {
            lappend arguments [list $argument $default]
        } else { lappend arguments $argument }
    }
    proc ::ocl_util::single_original_timing_check $arguments \
        [info body ::ocl_util::report_timing_and_scale_freq]
    proc ::ocl_util::report_timing_and_scale_freq {ocl_inst_path design_name vivado_output_dir vpl_output_dir clk_info clbinary_name {is_in_run true}} {
        dict set clk_info enable_auto_freq_scale false
        dict set clk_info skip_timing_and_scaling false
        dict set clk_info error_on_hold_violation true
        dict set clk_info worst_negative_slack 0
        return [single_original_timing_check $ocl_inst_path $design_name \
            $vivado_output_dir $vpl_output_dir $clk_info $clbinary_name $is_in_run]
    }
    set ::single_fixed_clock_installed true
    puts "Installed fixed 400 MHz clock setup; automatic frequency reduction disabled"
}
