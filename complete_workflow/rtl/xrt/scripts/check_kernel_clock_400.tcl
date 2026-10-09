set detectors [get_cells -hierarchical -filter {ORIG_REF_NAME == detector_core || REF_NAME == detector_core}]
if {[llength $detectors] != 1} {
    error "Expected one nine-site detector inside the XRT kernel; found [llength $detectors]"
}
set prefix "[get_property NAME [lindex $detectors 0]]/"
regsub -all {[][\\.^$*+?(){}|]} $prefix {\\&} detector_regex
# Hierarchical clock ports can disappear during optimization; leaf FD clock pins remain.
set detector_registers [get_cells -quiet -hierarchical -regexp \
    -filter {REF_NAME =~ FD.*} "^${detector_regex}.*"]
if {![llength $detector_registers]} { error "No detector registers found for the kernel clock check" }
set detector_clock_pins [get_pins -quiet -of_objects $detector_registers -filter {REF_PIN_NAME == C}]
if {![llength $detector_clock_pins]} { error "No detector register clock pins found" }
set kernel_clocks [get_clocks -quiet -of_objects $detector_clock_pins]
if {[llength $kernel_clocks] != 1} { error "Expected one clock on the detector" }
set kernel_period [get_property PERIOD [lindex $kernel_clocks 0]]
set requested_period 2.5
if {abs($kernel_period - $requested_period) > 0.01 * $requested_period} {
    error "Kernel clock is constrained to $kernel_period ns, expected $requested_period ns (400 MHz)"
}
puts "Verified detector kernel clock: $kernel_period ns (400 MHz requested), [llength $detector_clock_pins] register clock pins"
