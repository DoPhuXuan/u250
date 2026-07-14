set work_dir [pwd]
set source_dir [file join $work_dir "src"]
set build_dir [file join $work_dir "build" "hls"]
set report_dir [file join $work_dir "reports" "current"]

if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set part_name $::env(HLS_PART)
} else {
    set part_name "xcu250-figd2104-2L-e"
}

if {[info exists ::env(HLS_CLOCK_NS)] && $::env(HLS_CLOCK_NS) ne ""} {
    set clock_ns $::env(HLS_CLOCK_NS)
} else {
    set clock_ns 3.333
}

set allowed_tops [list \
    "bert_ffn_up_gelu_v21_dotpipe_kernel" \
    "bert_ffn_down_v21_dotpipe_kernel" \
    "bert_ffn_kernel_v21_dotpipe_estimate" \
]

set top_list {}
foreach arg $argv {
    if {$arg eq "" || [string match "-*" $arg] || [string match "*.tcl" $arg]} {
        continue
    }
    if {[lsearch -exact $allowed_tops $arg] < 0} {
        error "Unsupported FFN top '$arg'. This lab keeps only the production V21 tops."
    }
    lappend top_list $arg
}

if {[llength $top_list] == 0} {
    set top_list [lrange $allowed_tops 0 1]
}

file mkdir $build_dir
file mkdir $report_dir

proc report_name_for_top {top_name} {
    if {[string match "*up_gelu*" $top_name]} {
        return "csynth_up_v21.rpt"
    }
    if {[string match "*down*" $top_name]} {
        return "csynth_down_v21.rpt"
    }
    return "csynth_estimate_v21.rpt"
}

proc latest_report_path_for_top {project_dir top_name} {
    set report_dir [file join $project_dir "solution1" "syn" "report"]
    set generic_rpt [file join $report_dir "csynth.rpt"]
    set top_rpt [file join $report_dir "${top_name}_csynth.rpt"]

    if {[file exists $generic_rpt]} {
        return $generic_rpt
    }
    return $top_rpt
}

proc run_one_top {top_name part_name clock_ns work_dir source_dir build_dir report_dir} {
    set project_name "proj_$top_name"
    set project_dir [file join $build_dir $project_name]
    set source_file [file join $source_dir "bert_ffn_kernel_v21.cpp"]
    set header_file [file join $source_dir "bert_ffn_kernel_lab.h"]

    if {![file exists $source_file]} {
        error "Missing V21 source: $source_file"
    }
    if {![file exists $header_file]} {
        error "Missing FFN header: $header_file"
    }

    puts "============================================================"
    puts "Running FFN HLS top: $top_name"
    puts "Part: $part_name"
    puts "Clock: $clock_ns ns"
    puts "Project root: $build_dir"
    puts "============================================================"

    cd $build_dir
    open_project -reset $project_name
    add_files -cflags "-std=c++11" $source_file
    add_files $header_file
    set_top $top_name

    open_solution -reset "solution1"
    set_part $part_name
    create_clock -period $clock_ns -name default
    csynth_design

    set rpt_path [latest_report_path_for_top $project_dir $top_name]
    set out_path [file join $report_dir [report_name_for_top $top_name]]
    if {[file exists $rpt_path]} {
        file copy -force $rpt_path [file join $project_dir "csynth.rpt"]
        file copy -force $rpt_path $out_path
        puts "Copied latest report to $out_path"
    } else {
        puts "WARNING: expected report not found: $rpt_path"
    }

    close_project
    cd $work_dir
}

foreach top_name $top_list {
    run_one_top $top_name $part_name $clock_ns $work_dir $source_dir $build_dir $report_dir
}

exit
