set report_dir [pwd]
set report_tag "v21"

if {[info exists ::env(POSTROUTE_REPORT_DIR)] && $::env(POSTROUTE_REPORT_DIR) ne ""} {
    set report_dir $::env(POSTROUTE_REPORT_DIR)
}

if {[info exists ::env(POSTROUTE_TAG)] && $::env(POSTROUTE_TAG) ne ""} {
    set report_tag $::env(POSTROUTE_TAG)
}

file mkdir $report_dir

proc safe_report {cmd_list} {
    puts "Running: $cmd_list"
    if {[catch {uplevel #0 $cmd_list} result]} {
        puts "WARNING: report command failed: $result"
    }
}

safe_report [list report_timing_summary -max_paths 50 -file [file join $report_dir timing_summary_${report_tag}.rpt]]
safe_report [list report_utilization -hierarchical -file [file join $report_dir utilization_hier_${report_tag}.rpt]]
safe_report [list report_qor_suggestions -file [file join $report_dir qor_suggestions_${report_tag}.rpt]]
safe_report [list report_design_analysis -congestion -file [file join $report_dir congestion_${report_tag}.rpt]]
safe_report [list report_clock_utilization -file [file join $report_dir clock_util_${report_tag}.rpt]]

puts "Post-route report collection finished in $report_dir with tag $report_tag"
