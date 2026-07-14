set work_dir [pwd]
set source_dir [file join $work_dir "src" "int8"]
set test_dir [file join $work_dir "tests"]
set build_dir [file join $work_dir "build" "csim"]

set part_name "xcu250-figd2104-2L-e"
if {[info exists ::env(HLS_PART)] && $::env(HLS_PART) ne ""} {
    set part_name $::env(HLS_PART)
}

file mkdir $build_dir
cd $build_dir
open_project -reset proj_ffn_i8_v1_csim
set cflags "-std=c++11 -I${source_dir} -DFFN_I8_SEQ_LEN=2 -DFFN_I8_HIDDEN_SIZE=64 -DFFN_I8_INTERMEDIATE_SIZE=128 -DFFN_I8_K_PAR=16 -DFFN_I8_OUT_PAR=16 -DFFN_I8_GELU_PROFILE=2 -DFFN_I8_USE_BRAM=1"
add_files -cflags $cflags [file join $source_dir "bert_ffn_up_int8.cpp"]
add_files -cflags $cflags [file join $source_dir "bert_ffn_down_int8.cpp"]
add_files [file join $source_dir "ffn_int8_common.h"]
add_files -tb -cflags $cflags [file join $test_dir "test_ffn_int8_csim.cpp"]
set_top ffn_up_i8_v1
open_solution -reset solution1
set_part $part_name
create_clock -period 3.333 -name default
csim_design -clean
close_project
cd $work_dir
exit
