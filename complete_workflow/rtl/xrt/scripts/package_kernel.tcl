set script_dir [file dirname [file normalize [info script]]]
set xrt_root [file normalize [file join $script_dir ..]]
set rtl_root [file normalize [file join $xrt_root ..]]
set build_root [file normalize [file join $xrt_root build]]
if {[llength $argv] > 1} { error "usage: package_kernel.tcl ?build-directory?" }
if {[llength $argv] == 1} { set build_root [file normalize [lindex $argv 0]] }
set project_dir [file join $build_root package_project]
set ip_dir [file join $build_root packaged_ip]
set xo_path [file join $build_root neutral_atom_kernel.xo]

file mkdir $build_root
file delete -force $project_dir $ip_dir $xo_path
file copy -force [file join $rtl_root src weights.mem] [file join $build_root weights.mem]
file copy -force [file join $rtl_root src biases.mem] [file join $build_root biases.mem]

create_project -force neutral_atom_kernel_package $project_dir -part xcu55c-fsvh2892-2L-e
set_property target_language VHDL [current_project]
set_property simulator_language Mixed [current_project]

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
    [file join $xrt_root src neutral_atom_kernel.vhd] \
]

set manifest [open [file join $build_root source_files.txt] w]
foreach path $source_files { puts $manifest $path }
foreach name {weights.mem biases.mem} { puts $manifest [file join $rtl_root src $name] }
close $manifest

add_files -norecurse $source_files
add_files -norecurse [list \
    [file join $build_root weights.mem] \
    [file join $build_root biases.mem] \
]
set_property file_type {VHDL 2008} [get_files *.vhd]
set_property file_type {Memory Initialization Files} [get_files *.mem]
set_property top neutral_atom_kernel [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project \
    -root_dir $ip_dir \
    -vendor openai.org \
    -library RTLKernel \
    -taxonomy /KernelIP \
    -import_files \
    -set_current false

ipx::unload_core [file join $ip_dir component.xml]
ipx::edit_ip_in_project \
    -upgrade true \
    -name neutral_atom_kernel_ip_edit \
    -directory $ip_dir \
    [file join $ip_dir component.xml]

set core [ipx::current_core]
set_property name neutral_atom_kernel $core
set_property display_name {Neutral Atom Detector RTL Kernel} $core
set_property description {28x28 raw-frame, nine-site neural-network detector} $core
set_property core_revision 1 $core
foreach parameter [ipx::get_user_parameters -of_objects $core] {
    ipx::remove_user_parameter [get_property NAME $parameter] $core
}
set_property sdx_kernel true $core
set_property sdx_kernel_type rtl $core
set_property xpm_libraries {XPM_CDC XPM_MEMORY XPM_FIFO} $core
set_property supported_families {} $core
set_property auto_family_support_level level_2 $core

ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk $core
ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk $core
foreach bus_name {m_axi_gmem s_axi_control} {
    set bus_interface [ipx::get_bus_interfaces $bus_name -of_objects $core]
    set freq_parameter [ipx::get_bus_parameters FREQ_HZ -of_objects $bus_interface]
    if {[llength $freq_parameter] > 0} {
        ipx::remove_bus_parameter FREQ_HZ $bus_interface
    }
}
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity -quiet $core
ipx::update_source_project_archive -component $core
ipx::save_core $core
close_project -delete

package_xo \
    -force \
    -xo_path $xo_path \
    -kernel_name neutral_atom_kernel \
    -ip_directory $ip_dir \
    -kernel_xml [file join $xrt_root config kernel.xml]

puts "Created $xo_path"
