set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set build_dir [file join $root_dir build csim_fp32]
set part_name "xcu250-figd2104-2L-e"
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} { set part_name $::env(HLS_PART) }
file mkdir $build_dir
cd $build_dir
open_project -reset proj_ffn_fp32_real_bert_csim
set cflags "-std=c++11 -DSEQ_LEN=1 -I[file join $root_dir src]"
add_files -cflags $cflags [file join $root_dir src bert_ffn_kernel_v21.cpp]
add_files [file join $root_dir src bert_ffn_kernel_lab.h]
add_files -tb -cflags $cflags [file join $root_dir tests test_bert_ffn_fp32_csim.cpp]
set_top bert_ffn_up_gelu_v21_dotpipe_kernel
open_solution -reset solution1
set_part $part_name
create_clock -period 3.333 -name default
csim_design -O -clean
close_project
