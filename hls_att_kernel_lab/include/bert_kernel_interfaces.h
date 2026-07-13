#ifndef BERT_KERNEL_INTERFACES_H
#define BERT_KERNEL_INTERFACES_H

#include "bert_types.h"

// Port names and argument order are the stable link/host ABI. Token pointers
// use native C types; the new baseline performs bounded single accesses only.
extern "C" {
void bert_qkv_kernel(
    const bus_t *hidden_in,
    const unsigned int *hidden_ready,
    const qkv_weight_bus_t *attn_q_w_all,
    const bus_t *attn_q_b_all,
    const qkv_weight_bus_t *attn_k_w_all,
    const bus_t *attn_k_b_all,
    const qkv_weight_bus_t *attn_v_w_all,
    const bus_t *attn_v_b_all,
    int layer_id,
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream);

void bert_attn_core_kernel(
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream,
    const int *attention_mask,
    hidden_stream_t &context_stream);

void bert_attn_out_norm_kernel(
    hidden_stream_t &context_stream,
    const bus_t *residual_in,
    const unsigned int *residual_ready,
    const bus_t *attn_o_w_all,
    const bus_t *attn_o_b_all,
    const bus_t *attn_norm_gamma_all,
    const bus_t *attn_norm_beta_all,
    bus_t *attn_mid_ddr,
    unsigned int *attn_mid_done,
    int layer_id,
    hidden_stream_t &attn_mid_stream);
}

#endif
