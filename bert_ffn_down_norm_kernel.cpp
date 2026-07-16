#include "bert_stage_kernels.h"

static void run_ffn_down_to_ddr(
    hidden_stream_t &gelu_stream,
    hidden_stream_t &attn_mid_residual_stream,
    const bus_t     *down_w,
    const bus_t     *down_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    bus_t           *hidden_out)
{
#pragma HLS INLINE off
    hidden_stream_t hidden_out_stream("hidden_out_stream");
#pragma HLS STREAM variable=hidden_out_stream depth=512

#pragma HLS DATAFLOW
    bert_ffn_down_norm_stage(
        gelu_stream, attn_mid_residual_stream, down_w, down_b, norm_gamma, norm_beta,
        hidden_out_stream);
    store_hidden_stream(hidden_out_stream, hidden_out, TOKEN_WORDS);
}

extern "C" {
void bert_ffn_down_norm_kernel(
    hidden_stream_t &gelu_stream,
    hidden_stream_t &attn_mid_residual_stream,
    const bus_t     *ffn_down_w_all,
    const bus_t     *ffn_down_b_all,
    const bus_t     *ffn_norm_gamma_all,
    const bus_t     *ffn_norm_beta_all,
    bus_t           *hidden_out,
    int layer_id)
{
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE axis port=attn_mid_residual_stream
#pragma HLS INTERFACE m_axi port=ffn_down_w_all offset=slave bundle=gmem_ffn_down depth=1769472
#pragma HLS INTERFACE m_axi port=ffn_down_b_all offset=slave bundle=gmem_ffn_down depth=576
#pragma HLS INTERFACE m_axi port=ffn_norm_gamma_all offset=slave bundle=gmem_ffn_norm depth=576
#pragma HLS INTERFACE m_axi port=ffn_norm_beta_all offset=slave bundle=gmem_ffn_norm depth=576
#pragma HLS INTERFACE m_axi port=hidden_out offset=slave bundle=gmem_hidden depth=6144
#pragma HLS INTERFACE s_axilite port=ffn_down_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=ffn_down_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=ffn_norm_gamma_all bundle=control
#pragma HLS INTERFACE s_axilite port=ffn_norm_beta_all bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_out bundle=control
#pragma HLS INTERFACE s_axilite port=layer_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int weight_offset = layer_id * FFN_DN_W_PACKS;
    const int bias_offset = layer_id * FFN_DN_B_PACKS;
    const int norm_offset = layer_id * PACKS;

    run_ffn_down_to_ddr(
        gelu_stream,
        attn_mid_residual_stream,
        ffn_down_w_all + weight_offset,
        ffn_down_b_all + bias_offset,
        ffn_norm_gamma_all + norm_offset,
        ffn_norm_beta_all + norm_offset,
        hidden_out);
}
} // extern "C"
