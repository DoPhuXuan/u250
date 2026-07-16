#ifndef BERT_STAGE_KERNELS_H
#define BERT_STAGE_KERNELS_H

#include "bert_model.h"

// Monotonic tokens synchronize the two intentional DDR residual hand-offs.
// A token is one 32-bit word in global memory and has no tensor payload.
typedef ap_uint<32> stage_token_t;

// Keep the AXI-visible token as a native 32-bit volatile object.  Vitis HLS
// 2021.2 does not reliably bind ap_uint comparison operators to a volatile
// RHS, even though ap_uint itself is still used for the internal token value.
static void write_stage_token(
    volatile unsigned int *token,
    const stage_token_t value)
{
#pragma HLS INLINE off
    const unsigned int raw_value = value.to_uint();
    token[0] = raw_value;
}

static void wait_for_stage_token(
    volatile unsigned int *token,
    const stage_token_t expected)
{
#pragma HLS INLINE off
    stage_token_t observed = 0;
WAIT_FOR_STAGE_TOKEN:
    while (observed != expected) {
#pragma HLS PIPELINE II=1
        // First read the volatile AXI word into a native local.  Both operands
        // of the ap_uint comparison below are then non-volatile in 2021.2.
        unsigned int raw_observed = token[0];
        observed = raw_observed;
    }
}

#endif // BERT_STAGE_KERNELS_H
