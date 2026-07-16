#include "bert_stage_kernels.h"

extern "C" {
void bert_attn_core_kernel(
    hidden_stream_t &qkv_stream,
    const int       *attention_mask,
    hidden_stream_t &context_stream)
{
#pragma HLS INTERFACE axis port=qkv_stream
#pragma HLS INTERFACE axis port=context_stream
#pragma HLS INTERFACE m_axi port=attention_mask offset=slave bundle=gmem_mask depth=128
#pragma HLS INTERFACE s_axilite port=attention_mask bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    bert_attn_core_stage(qkv_stream, attention_mask, context_stream);
}
} // extern "C"
