set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set build_dir [file join $root_dir build hls]
set report_dir [file join $root_dir reports current]

set part_name "xcu250-figd2104-2L-e"
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set part_name $::env(HLS_PART)
}

set clock_ns 3.333
if {[info exists ::env(HLS_CLOCK_NS)] && $::env(HLS_CLOCK_NS) ne ""} {
    if {![string is double -strict $::env(HLS_CLOCK_NS)] ||
        $::env(HLS_CLOCK_NS) <= 0.0} {
        error "HLS_CLOCK_NS must be a positive number"
    }
    set clock_ns [expr {double($::env(HLS_CLOCK_NS))}]
}

set run_rtl_syn 0
if {[info exists ::env(RUN_RTL_SYN)] && $::env(RUN_RTL_SYN) eq "1"} {
    set run_rtl_syn 1
}

array set source_for {
    bert_qkv_kernel                           bert_qkv_kernel.cpp
    bert_attn_core_kernel                     bert_attn_core_kernel.cpp
    bert_attn_out_norm_residual_kernel        bert_attn_out_norm_residual_kernel.cpp
    bert_ffn_up_gelu_v21_dotpipe_kernel       bert_ffn_fp32_kernels.cpp
    bert_ffn_down_residual_norm_fp32_kernel   bert_ffn_fp32_kernels.cpp
    bert_qkv_12layer_kernel                   bert_qkv_kernel.cpp
    bert_attn_core_12layer_kernel             bert_attn_core_kernel.cpp
    bert_attn_out_norm_12layer_kernel         bert_attn_out_norm_residual_kernel.cpp
    bert_ffn_up_gelu_12layer_kernel           bert_ffn_fp32_kernels.cpp
    bert_ffn_down_norm_feedback_12layer_kernel bert_ffn_fp32_kernels.cpp
}

array set report_for {
    bert_qkv_kernel                           csynth_qkv.rpt
    bert_attn_core_kernel                     csynth_attn_core.rpt
    bert_attn_out_norm_residual_kernel        csynth_attn_out_norm_residual.rpt
    bert_ffn_up_gelu_v21_dotpipe_kernel       csynth_ffn_up_fp32.rpt
    bert_ffn_down_residual_norm_fp32_kernel   csynth_ffn_down_residual_norm_fp32.rpt
    bert_qkv_12layer_kernel                   csynth_full_qkv_12layer.rpt
    bert_attn_core_12layer_kernel             csynth_full_attn_core_12layer.rpt
    bert_attn_out_norm_12layer_kernel         csynth_full_attn_out_12layer.rpt
    bert_ffn_up_gelu_12layer_kernel           csynth_full_ffn_up_12layer.rpt
    bert_ffn_down_norm_feedback_12layer_kernel csynth_full_ffn_down_12layer.rpt
}

set all_kernels [list \
    bert_qkv_kernel \
    bert_attn_core_kernel \
    bert_attn_out_norm_residual_kernel \
    bert_ffn_up_gelu_v21_dotpipe_kernel \
    bert_ffn_down_residual_norm_fp32_kernel]

set full_model_kernels [list \
    bert_qkv_12layer_kernel \
    bert_attn_core_12layer_kernel \
    bert_attn_out_norm_12layer_kernel \
    bert_ffn_up_gelu_12layer_kernel \
    bert_ffn_down_norm_feedback_12layer_kernel]

# With no argument, synthesize only the two changed kernels.  Pass "all" to
# rebuild the complete one-layer FP32 chain from the copied locked sources.
set selected {}
set requested $argv
if {[info exists ::env(FP32_KERNELS)] && $::env(FP32_KERNELS) ne ""} {
    set requested [split $::env(FP32_KERNELS) ","]
}
foreach arg $requested {
    set arg [string trim $arg]
    # Vitis HLS 2022.1 keeps its own command-line tokens in $argv, typically
    # "-f scripts/run_vitis_fp32_csynth.tcl".  They are launcher arguments,
    # not kernel selections.
    if {$arg eq "" || [string match "-*" $arg] || [string match "*.tcl" $arg]} {
        continue
    }
    if {$arg eq "all"} {
        set selected $all_kernels
        break
    }
    if {$arg eq "full"} {
        set selected $full_model_kernels
        break
    }
    if {![info exists source_for($arg)]} {
        error "Unknown kernel '$arg'. Valid values: [join [concat $all_kernels $full_model_kernels] {, }], all, full"
    }
    lappend selected $arg
}
if {[llength $selected] == 0} {
    set selected [list \
        bert_attn_out_norm_residual_kernel \
        bert_ffn_down_residual_norm_fp32_kernel]
}

file mkdir $build_dir
file mkdir $report_dir

proc copy_csynth_report {project_dir top destination} {
    set syn_report_dir [file join $project_dir solution1 syn report]
    set source [file join $syn_report_dir csynth.rpt]
    if {![file exists $source] || [file size $source] == 0} {
        set source [file join $syn_report_dir "${top}_csynth.rpt"]
    }
    if {![file exists $source] || [file size $source] == 0} {
        error "Missing csynth report for $top in $syn_report_dir"
    }
    file copy -force $source $destination

    set detail_dir [file join [file dirname $destination] details $top]
    file delete -force $detail_dir
    file mkdir $detail_dir
    foreach path [glob -nocomplain [file join $syn_report_dir *]] {
        if {[file isfile $path] && [file size $path] > 0} {
            file copy -force $path $detail_dir
        }
    }
}

proc synth_one {top source_name report_name} {
    global root_dir build_dir report_dir part_name clock_ns run_rtl_syn
    set source_path [file join $root_dir kernels $source_name]
    set project_name "proj_${top}"
    set project_dir [file join $build_dir $project_name]
    set report_path [file join $report_dir $report_name]

    puts "============================================================"
    puts "Top:    $top"
    puts "Source: $source_path"
    puts "Part:   $part_name"
    puts "Clock:  $clock_ns ns"
    puts "============================================================"

    set old_dir [pwd]
    set ok 1
    if {[catch {
        cd $build_dir
        open_project -reset $project_name
        add_files -cflags "-std=c++14 -I[file join $root_dir include]" $source_path
        set_top $top
        open_solution -reset solution1 -flow_target vivado
        set_part $part_name
        create_clock -period $clock_ns -name default
        csynth_design
        copy_csynth_report $project_dir $top $report_path

        if {$run_rtl_syn} {
            export_design -flow syn -rtl verilog
            set impl_reports [file join $project_dir solution1 impl report]
            set rtl_destination [file join $report_dir rtl_syn $top]
            file delete -force $rtl_destination
            file mkdir [file dirname $rtl_destination]
            file copy -force $impl_reports $rtl_destination
        }
    } message options]} {
        puts stderr "ERROR while synthesizing $top: $message"
        if {[dict exists $options -errorinfo]} {
            puts stderr [dict get $options -errorinfo]
        }
        set ok 0
    }
    catch {close_project}
    cd $old_dir
    return $ok
}

set failed {}
foreach top $selected {
    if {[synth_one $top $source_for($top) $report_for($top)]} {
        puts "PASS: $top"
    } else {
        lappend failed $top
    }
}

if {[llength $failed] != 0} {
    puts stderr "FAILED: [join $failed {, }]"
    exit 1
}
puts "ALL SELECTED KERNELS PASSED CSYNTH"
exit 0
