<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" top="bert_ffn_up_gelu_v21_dotpipe_kernel" name="proj_ffn_fp32_real_bert_csim">
    <files>
        <file name="C:/Users/phudo/Documents/DOCUMENT/DOAN/KLTN/KLTN/hls_ffn_kernel_lab/tests/test_bert_ffn_fp32_csim.cpp" sc="0" tb="1" cflags="-std=c++11 -DSEQ_LEN=1 -IC:/Users/phudo/Documents/DOCUMENT/DOAN/KLTN/KLTN/hls_ffn_kernel_lab/src -Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="C:/Users/phudo/Documents/DOCUMENT/DOAN/KLTN/KLTN/hls_ffn_kernel_lab/src/bert_ffn_kernel_lab.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="C:/Users/phudo/Documents/DOCUMENT/DOAN/KLTN/KLTN/hls_ffn_kernel_lab/src/bert_ffn_kernel_v21.cpp" sc="0" tb="false" cflags="-std=c++11 -DSEQ_LEN=1 -IC:/Users/phudo/Documents/DOCUMENT/DOAN/KLTN/KLTN/hls_ffn_kernel_lab/src" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="true" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

