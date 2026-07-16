#include "bert_stage_kernels.h"

extern "C" {
void bert_qkv_kernel(
    const bus_t *hidden_in,
    volatile unsigned int *hidden_ready,
    const bus_t *attn_q_w_all,
    const bus_t *attn_q_b_all,
    const bus_t *attn_k_w_all,
    const bus_t *attn_k_b_all,
    const bus_t *attn_v_w_all,
    const bus_t *attn_v_b_all,
    int layer_id,
    hidden_stream_t &qkv_stream)
{
#pragma HLS INTERFACE m_axi port=hidden_in offset=slave bundle=gmem_hidden depth=6144
#pragma HLS INTERFACE m_axi port=hidden_ready offset=slave bundle=gmem_hidden_ready depth=1
#pragma HLS INTERFACE m_axi port=attn_q_w_all offset=slave bundle=gmem_q depth=442368
#pragma HLS INTERFACE m_axi port=attn_q_b_all offset=slave bundle=gmem_q depth=576
#pragma HLS INTERFACE m_axi port=attn_k_w_all offset=slave bundle=gmem_k depth=442368
#pragma HLS INTERFACE m_axi port=attn_k_b_all offset=slave bundle=gmem_k depth=576
#pragma HLS INTERFACE m_axi port=attn_v_w_all offset=slave bundle=gmem_v depth=442368
#pragma HLS INTERFACE m_axi port=attn_v_b_all offset=slave bundle=gmem_v depth=576
#pragma HLS INTERFACE axis port=qkv_stream
#pragma HLS INTERFACE s_axilite port=hidden_in bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_ready bundle=control
#pragma HLS INTERFACE s_axilite port=attn_q_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_q_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_k_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_k_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_v_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_v_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=layer_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int weight_offset = layer_id * ATTN_W_PACKS;
    const int bias_offset = layer_id * ATTN_B_PACKS;
    const bus_t *q_w = attn_q_w_all + weight_offset;
    const bus_t *q_b = attn_q_b_all + bias_offset;
    const bus_t *k_w = attn_k_w_all + weight_offset;
    const bus_t *k_b = attn_k_b_all + bias_offset;
    const bus_t *v_w = attn_v_w_all + weight_offset;
    const bus_t *v_b = attn_v_b_all + bias_offset;

    wait_for_stage_token(hidden_ready, (stage_token_t)(layer_id + 1));

    hidden_stream_t qkv_input_stream("qkv_input_stream");
#pragma HLS STREAM variable=qkv_input_stream depth=64

#pragma HLS DATAFLOW
    load_hidden_stream(hidden_in, qkv_input_stream, TOKEN_WORDS);
    // QKV projection is intentionally isolated in this kernel.  The embedding
    // prep kernel only materializes hidden_in/hidden_ready and never calls
    // bert_qkv_stage, so this remains the single implementation QKV stage.
    bert_qkv_stage(qkv_input_stream, q_w, q_b, k_w, k_b, v_w, v_b, qkv_stream);
}
} // extern "C"
