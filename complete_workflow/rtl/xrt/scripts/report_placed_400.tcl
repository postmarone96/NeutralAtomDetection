set report_dir [file join [pwd] single_cnn_reports]
file mkdir $report_dir
write_checkpoint -force [file join $report_dir kernel_placed.dcp]
report_timing_summary -delay_type min_max -max_paths 100 -report_unconstrained \
    -file [file join $report_dir timing_placed.rpt]
report_timing -delay_type max -max_paths 100 -nworst 1 -input_pins -nets \
    -file [file join $report_dir setup_paths_placed.rpt]
report_high_fanout_nets -fanout_greater_than 100 \
    -file [file join $report_dir high_fanout_placed.rpt]
report_design_analysis -congestion \
    -file [file join $report_dir congestion_placed.rpt]
