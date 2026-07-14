set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set source_dir [file join $root_dir src int8]
set build_dir [file join $root_dir build hls]
set report_dir [file join $root_dir reports int8]

set part_name "xcu250-figd2104-2L-e"
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set part_name $::env(HLS_PART)
}

set clock_ns 3.333
if {[info exists ::env(HLS_CLOCK_NS)] && $::env(HLS_CLOCK_NS) ne ""} {
    set clock_ns $::env(HLS_CLOCK_NS)
}

set k_par 32
if {[info exists ::env(FFN_I8_K_PAR)] && $::env(FFN_I8_K_PAR) ne ""} {
    set k_par $::env(FFN_I8_K_PAR)
}

set out_par 32
if {[info exists ::env(FFN_I8_OUT_PAR)] && $::env(FFN_I8_OUT_PAR) ne ""} {
    set out_par $::env(FFN_I8_OUT_PAR)
}

set gelu_profile 2
if {[info exists ::env(FFN_I8_GELU_PROFILE)] && $::env(FFN_I8_GELU_PROFILE) ne ""} {
    set gelu_profile $::env(FFN_I8_GELU_PROFILE)
}
if {$gelu_profile != 0 && $gelu_profile != 2} {
    error "FFN_I8_GELU_PROFILE must be 0 or 2"
}

set allowed_profiles [list "16x16" "32x16" "32x32" "64x16"]
set selected_profile "${k_par}x${out_par}"
if {[lsearch -exact $allowed_profiles $selected_profile] < 0} {
    error "Unsupported INT8 profile $selected_profile; choose one of $allowed_profiles"
}

array set source_for {
    ffn_up_i8_v1   bert_ffn_up_int8.cpp
    ffn_down_i8_v1 bert_ffn_down_int8.cpp
}

# Only these two files are exported from the HLS projects.
array set report_for {
    ffn_up_i8_v1   csynth_up_int8.rpt
    ffn_down_i8_v1 csynth_down_int8.rpt
}

set all_tops [list ffn_up_i8_v1 ffn_down_i8_v1]
set selected_tops {}
foreach arg $argv {
    if {[info exists source_for($arg)]} {
        lappend selected_tops $arg
    } elseif {$arg ne "" && ![string match "-*" $arg] && ![string match "*.tcl" $arg]} {
        error "Unsupported FFN INT8 top: $arg"
    }
}
if {[llength $selected_tops] == 0} {
    set selected_tops $all_tops
}

file mkdir $build_dir
file mkdir $report_dir

proc export_csynth_report {project_dir output_path top_name} {
    set hls_report_dir [file join $project_dir solution1 syn report]
    set source_report [file join $hls_report_dir csynth.rpt]
    if {![file exists $source_report] || [file size $source_report] == 0} {
        set source_report [file join $hls_report_dir "${top_name}_csynth.rpt"]
    }
    if {![file exists $source_report] || [file size $source_report] == 0} {
        error "csynth.rpt is missing for $top_name in $hls_report_dir"
    }
    file copy -force $source_report $output_path
    puts "Exported: $output_path"
}

proc synth_one_top {top_name} {
    global root_dir source_dir build_dir report_dir part_name clock_ns
    global k_par out_par gelu_profile selected_profile source_for report_for

    set project_name "proj_${top_name}_k${k_par}_o${out_par}_dsp2_g${gelu_profile}_fuse4_down2"
    set project_dir [file join $build_dir $project_name]
    set source_file [file join $source_dir $source_for($top_name)]
    set header_file [file join $source_dir ffn_int8_common.h]
    set output_report [file join $report_dir $report_for($top_name)]
    set cflags "-std=c++11 -DFFN_I8_K_PAR=${k_par} -DFFN_I8_OUT_PAR=${out_par} -DFFN_I8_GELU_PROFILE=${gelu_profile}"
    if {[info exists ::env(FFN_I8_USE_BRAM)] && $::env(FFN_I8_USE_BRAM) eq "1"} {
        append cflags " -DFFN_I8_USE_BRAM=1"
    }

    puts "============================================================"
    puts "Kernel: $top_name"
    puts "Profile: $selected_profile"
    puts "DSP packing: 2 signed INT8 products per multiply"
    puts "GELU profile: G$gelu_profile"
    puts "UP weight/MAC overlap: pre-muxed single-MAC fused schedule fuse4"
    puts "DOWN weight/MAC overlap: pre-muxed fused schedule down2"
    puts "Part: $part_name"
    puts "Clock: $clock_ns ns"
    puts "============================================================"

    set old_dir [pwd]
    set ok 1
    if {[catch {
        cd $build_dir
        open_project -reset $project_name
        add_files -cflags $cflags $source_file
        add_files $header_file
        set_top $top_name
        open_solution -reset solution1 -flow_target vivado
        set_part $part_name
        create_clock -period $clock_ns -name default
        csynth_design
        export_csynth_report $project_dir $output_report $top_name
    } message]} {
        puts stderr "ERROR: $top_name: $message"
        set ok 0
    }
    catch {close_project}
    cd $old_dir
    return $ok
}

set failed_tops {}
foreach top_name $selected_tops {
    if {![synth_one_top $top_name]} {
        lappend failed_tops $top_name
    }
}

if {[llength $failed_tops] != 0} {
    puts stderr "FAILED: [join $failed_tops {, }]"
    exit 1
}

puts "Generated reports:"
foreach top_name $selected_tops {
    puts "  [file join $report_dir $report_for($top_name)]"
}
exit 0
