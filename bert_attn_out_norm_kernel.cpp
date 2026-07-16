#include "bert_stage_kernels.h"

static void run_attn_out_streaming(
    hidden_stream_t &context_stream,
    const bus_t     *residual_in,
    const bus_t     *dense_w,
    const bus_t     *dense_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &attn_mid_stream,
    hidden_stream_t &attn_mid_residual_stream)
{
#pragma HLS INLINE off
    hidden_stream_t norm_stream("attn_norm_stream");
#pragma HLS STREAM variable=norm_stream depth=512

#pragma HLS DATAFLOW
    bert_attn_out_norm_stage(
        context_stream, residual_in, dense_w, dense_b, norm_gamma, norm_beta,
        norm_stream);
    duplicate_stream(norm_stream, attn_mid_stream, attn_mid_residual_stream, TOKEN_WORDS);
}

extern "C" {
void bert_attn_out_norm_kernel(
    hidden_stream_t &context_stream,
    const bus_t     *residual_in,
    volatile unsigned int *residual_ready,
    const bus_t     *attn_o_w_all,
    const bus_t     *attn_o_b_all,
    const bus_t     *attn_norm_gamma_all,
    const bus_t     *attn_norm_beta_all,
    int layer_id,
    hidden_stream_t &attn_mid_stream,
    hidden_stream_t &attn_mid_residual_stream)
{
#pragma HLS INTERFACE axis port=context_stream
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=attn_mid_residual_stream
#pragma HLS INTERFACE m_axi port=residual_in offset=slave bundle=gmem_hidden depth=6144
#pragma HLS INTERFACE m_axi port=residual_ready offset=slave bundle=gmem_hidden depth=1
#pragma HLS INTERFACE m_axi port=attn_o_w_all offset=slave bundle=gmem_o depth=442368
#pragma HLS INTERFACE m_axi port=attn_o_b_all offset=slave bundle=gmem_o depth=576
#pragma HLS INTERFACE m_axi port=attn_norm_gamma_all offset=slave bundle=gmem_norm depth=576
#pragma HLS INTERFACE m_axi port=attn_norm_beta_all offset=slave bundle=gmem_norm depth=576
#pragma HLS INTERFACE s_axilite port=residual_in bundle=control
#pragma HLS INTERFACE s_axilite port=residual_ready bundle=control
#pragma HLS INTERFACE s_axilite port=attn_o_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_o_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_norm_gamma_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_norm_beta_all bundle=control
#pragma HLS INTERFACE s_axilite port=layer_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int weight_offset = layer_id * ATTN_W_PACKS;
    const int bias_offset = layer_id * ATTN_B_PACKS;
    const int norm_offset = layer_id * PACKS;

    // Layer 0 waits for the embedding write; later layers wait for the QKV
    // kernel to publish that the prior layer's hidden buffer is the residual.
    wait_for_stage_token(residual_ready, (stage_token_t)(layer_id + 1));
    run_attn_out_streaming(
        context_stream,
        residual_in,
        attn_o_w_all + weight_offset,
        attn_o_b_all + bias_offset,
        attn_norm_gamma_all + norm_offset,
        attn_norm_beta_all + norm_offset,
        attn_mid_stream,
        attn_mid_residual_stream);
}
} // extern "C"
