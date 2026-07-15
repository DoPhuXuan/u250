#ifndef BERT_FULL_MODEL_INTERFACES_H
#define BERT_FULL_MODEL_INTERFACES_H

#include "bert_types.h"

// Persistent 12-layer encoder ABI.  Every CU is started exactly once.  Stream
// ordering and the identical fixed NUM_LAYERS loop keep all CUs synchronized.
extern "C" {
void bert_qkv_12layer_kernel(
    const bus_t *initial_hidden,
    const unsigned int *initial_hidden_ready,
    const qkv_weight_bus_t *attn_q_w_all,
    const bus_t *attn_q_b_all,
    const qkv_weight_bus_t *attn_k_w_all,
    const bus_t *attn_k_b_all,
    const qkv_weight_bus_t *attn_v_w_all,
    const bus_t *attn_v_b_all,
    hidden_stream_t &next_hidden_stream,
    hidden_stream_t &attention_residual_stream,
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream);

void bert_attn_core_12layer_kernel(
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream,
    const int *attention_mask,
    hidden_stream_t &context_stream);

void bert_attn_out_norm_12layer_kernel(
    hidden_stream_t &context_stream,
    hidden_stream_t &attention_residual_stream,
    const bus_t *attn_o_w_all,
    const bus_t *attn_o_b_all,
    const bus_t *attn_norm_gamma_all,
    const bus_t *attn_norm_beta_all,
    hidden_stream_t &attn_mid_stream,
    hidden_stream_t &ffn_residual_stream);

void bert_ffn_up_gelu_12layer_kernel(
    hidden_stream_t &attn_mid_stream,
    bus_t *w1_all,
    bus_t *b1_all,
    hidden_stream_t &gelu_stream);

void bert_ffn_down_norm_feedback_12layer_kernel(
    hidden_stream_t &gelu_stream,
    hidden_stream_t &ffn_residual_stream,
    bus_t *w2_all,
    bus_t *b2_all,
    bus_t *final_norm_gamma_all,
    bus_t *final_norm_beta_all,
    hidden_stream_t &next_hidden_stream,
    bus_t *final_output_hidden);
}

#endif

