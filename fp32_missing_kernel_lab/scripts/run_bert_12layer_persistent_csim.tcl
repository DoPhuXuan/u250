set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set build_dir [file join $root_dir build hls]
set project_name proj_bert_12layer_persistent_csim
set project_dir [file join $build_dir $project_name]

set data_dir [file join $root_dir validation data bert_base_uncased_seq128]
if {[info exists ::env(BERT_CSIM_DATA)] && $::env(BERT_CSIM_DATA) ne ""} {
    set data_dir [file normalize $::env(BERT_CSIM_DATA)]
}
set output_dir [file join $root_dir validation outputs persistent_12layer]
set report_dir [file join $root_dir reports csim persistent_12layer]
foreach required [list \
    [file join $data_dir metadata.json] \
    [file join $data_dir input embedding_output.f32] \
    [file join $data_dir weights attention_q_weight.packed.f32] \
    [file join $data_dir weights layer_11 ffn_w2.packed.f32] \
    [file join $data_dir golden encoder_layer_11.f32]] {
    if {![file exists $required] || [file size $required] == 0} {
        error "Missing persistent C-sim input: $required"
    }
}

file delete -force $output_dir
file mkdir $output_dir
file mkdir $report_dir
file mkdir $build_dir

set cflags "-std=c++14 -O2 -pthread -DBERT_CSIM_THREAD_SAFE_STREAM -I[file join $root_dir include]"
set old_dir [pwd]
cd $build_dir
open_project -reset $project_name
add_files -cflags $cflags [file join $root_dir kernels bert_qkv_kernel.cpp]
add_files -cflags $cflags [file join $root_dir kernels bert_attn_core_kernel.cpp]
add_files -cflags $cflags [file join $root_dir kernels bert_attn_out_norm_residual_kernel.cpp]
add_files -cflags $cflags [file join $root_dir kernels bert_ffn_fp32_kernels.cpp]
add_files -tb -cflags $cflags [file join $root_dir validation tb_bert_12layer_persistent_csim.cpp]
set_top bert_qkv_12layer_kernel
open_solution -reset solution1 -flow_target vivado
set_part xcu250-figd2104-2L-e
create_clock -period 3.333 -name default

puts "============================================================"
puts "Persistent full-model C-sim: five CUs, one start each"
puts "Bounded feedback FIFO model enabled"
puts "Data:    $data_dir"
puts "Outputs: $output_dir"
puts "============================================================"
set csim_argv [join [list $data_dir $output_dir] " "]
csim_design -O -clean -ldflags "-pthread" -argv $csim_argv

set csim_report_source [file join $project_dir solution1 csim report]
set csim_report_destination [file join $report_dir vitis_csim]
if {[file isdirectory $csim_report_source]} {
    file delete -force $csim_report_destination
    file mkdir $csim_report_destination
    foreach report_file [glob -nocomplain [file join $csim_report_source *]] {
        if {[file isfile $report_file]} {
            file copy -force $report_file $csim_report_destination
        }
    }
}
file copy -force \
    [file join $output_dir persistent_csim_summary.txt] \
    [file join $report_dir persistent_csim_summary.txt]
close_project
cd $old_dir

set run_compare 1
if {[info exists ::env(RUN_CSIM_COMPARE)] && $::env(RUN_CSIM_COMPARE) eq "0"} {
    set run_compare 0
}
if {$run_compare} {
    set python_bin python3
    if {[info exists ::env(PYTHON_BIN)] && $::env(PYTHON_BIN) ne ""} {
        set python_bin $::env(PYTHON_BIN)
    }
    set compare_command [list \
        $python_bin [file join $root_dir validation compare_persistent_output.py] \
        --data-dir $data_dir \
        --csim-output-dir $output_dir \
        --report-dir $report_dir]
    puts "Comparing persistent final output with Hugging Face layer 11..."
    if {[catch {exec {*}$compare_command 2>@1} compare_output]} {
        puts stderr $compare_output
        error "Persistent C-sim completed, but final accuracy failed"
    }
    puts $compare_output
    puts "Persistent C-sim and accuracy gate passed."
}

