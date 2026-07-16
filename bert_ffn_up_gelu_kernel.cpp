#include "bert_stage_kernels.h"

extern "C" {
void bert_ffn_up_gelu_kernel(
    hidden_stream_t &attn_mid_stream,
    const bus_t     *ffn_up_w_all,
    const bus_t     *ffn_up_b_all,
    int layer_id,
    hidden_stream_t &gelu_stream)
{
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=ffn_up_w_all offset=slave bundle=gmem_ffn_up depth=1769472
#pragma HLS INTERFACE m_axi port=ffn_up_b_all offset=slave bundle=gmem_ffn_up depth=2304
#pragma HLS INTERFACE s_axilite port=ffn_up_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=ffn_up_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=layer_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int weight_offset = layer_id * FFN_UP_W_PACKS;
    const int bias_offset = layer_id * FFN_UP_B_PACKS;
    bert_ffn_up_gelu_stage(
        attn_mid_stream,
        ffn_up_w_all + weight_offset,
        ffn_up_b_all + bias_offset,
        gelu_stream);
}
} // extern "C"
