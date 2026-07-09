set work_dir [pwd]

if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set part_name $::env(HLS_PART)
} else {
    set part_name "xcu250-figd2104-2L-e"
}

if {[info exists ::env(HLS_CLOCK_NS)] && $::env(HLS_CLOCK_NS) ne ""} {
    set clock_ns $::env(HLS_CLOCK_NS)
} else {
    set clock_ns 5
}

set top_list {}
foreach arg $argv {
    if {$arg eq ""} {
        continue
    }
    if {[string match "-*" $arg]} {
        continue
    }
    if {[string match "*.tcl" $arg]} {
        continue
    }
    lappend top_list $arg
}

if {[llength $top_list] == 0} {
    set top_list [list \
        "bert_ffn_up_gelu_v21_dotpipe_kernel" \
        "bert_ffn_down_v21_dotpipe_kernel" \
    ]
}

proc uses_packed_cpp {top_name} {
    if {[string match "*packed512*" $top_name]} {
        return 1
    }
    if {[string match "*stream_tile32*" $top_name]} {
        return 1
    }
    if {[string match "*v11*" $top_name]} {
        return 1
    }
    if {[string match "*v12*" $top_name]} {
        return 1
    }
    if {[string match "*v13*" $top_name]} {
        return 1
    }
    if {[string match "*v14*" $top_name]} {
        return 1
    }
    if {[string match "*v15*" $top_name]} {
        return 1
    }
    if {[string match "*v16*" $top_name]} {
        return 1
    }
    if {[string match "*v17*" $top_name]} {
        return 1
    }
    if {[string match "*v18*" $top_name]} {
        return 1
    }
    if {[string match "*v20*" $top_name]} {
        return 1
    }
    if {[string match "*v21*" $top_name]} {
        return 1
    }
    return 0
}

proc version_suffix_for_top {top_name} {
    foreach version {v21 v20 v18 v17 v16 v15 v14 v13 v12 v11 v10 v9 v8 v7 v6 v5 v4 v3 v2 v1} {
        if {[string match "*${version}*" $top_name]} {
            return "_$version"
        }
    }
    return ""
}

proc report_name_for_top {top_name} {
    set suffix [version_suffix_for_top $top_name]
    if {[string match "*up_gelu*" $top_name]} {
        return "csynth_up${suffix}.rpt"
    }
    if {[string match "*down*" $top_name]} {
        return "csynth_down${suffix}.rpt"
    }
    if {[string match "*estimate*" $top_name]} {
        return "csynth_estimate${suffix}.rpt"
    }
    return "csynth${suffix}.rpt"
}

proc latest_report_path_for_top {project_dir top_name} {
    set report_dir [file join $project_dir "solution1" "syn" "report"]
    set generic_rpt [file join $report_dir "csynth.rpt"]
    set top_rpt [file join $report_dir "${top_name}_csynth.rpt"]

    if {[file exists $generic_rpt]} {
        return $generic_rpt
    }
    if {[file exists $top_rpt]} {
        return $top_rpt
    }
    return $generic_rpt
}

proc run_one_top {top_name part_name clock_ns work_dir} {
    set project_name "proj_$top_name"
    set project_dir [file join $work_dir $project_name]

    puts "============================================================"
    puts "Running FFN HLS top: $top_name"
    puts "Part: $part_name"
    puts "Clock: $clock_ns ns"
    puts "============================================================"

    open_project -reset $project_name

    if {[string match "*v21*" $top_name]} {
        set packed_cpp [file join $work_dir "bert_ffn_kernel_v21.cpp"]
    } elseif {[string match "*v20*" $top_name]} {
        set packed_cpp [file join $work_dir "bert_ffn_kernel_v20.cpp"]
    } else {
        set packed_cpp [file join $work_dir "bert_ffn_kernel_hls_packed.cpp"]
    }
    set c_source [file join $work_dir "bert_ffn_kernel_hls.c"]
    set header_file [file join $work_dir "bert_ffn_kernel_lab.h"]

    if {[uses_packed_cpp $top_name]} {
        if {![file exists $packed_cpp]} {
            error "Missing packed C++ source for $top_name: $packed_cpp"
        }
        add_files -cflags "-std=c++11" $packed_cpp
    } else {
        if {![file exists $c_source]} {
            error "Missing bert_ffn_kernel_hls.c. Copy this file into the HLS working directory."
        }
        add_files $c_source
    }

    if {![file exists $header_file]} {
        error "Missing bert_ffn_kernel_lab.h. Copy this file into the HLS working directory."
    }
    add_files $header_file

    set_top $top_name

    open_solution -reset "solution1"
    set_part $part_name
    create_clock -period $clock_ns -name default

    csynth_design

    set rpt_path [latest_report_path_for_top $project_dir $top_name]
    set out_name [report_name_for_top $top_name]
    if {[file exists $rpt_path]} {
        file copy -force $rpt_path [file join $project_dir "csynth.rpt"]
        file copy -force $rpt_path [file join $work_dir $out_name]
        file copy -force $rpt_path [file join $work_dir "csynth.rpt"]
        puts "Copied latest report from $rpt_path"
        puts "Copied latest report to $project_name/csynth.rpt, $out_name, and csynth.rpt"
    } else {
        puts "WARNING: expected report not found: $rpt_path"
    }

    close_project
}

foreach top_name $top_list {
    run_one_top $top_name $part_name $clock_ns $work_dir
}

exit
