#ifndef BERT_MATH_H
#define BERT_MATH_H

#include "bert_types.h"

#include <stdint.h>

static inline float uint32_to_float(uint32_t x) {
#pragma HLS INLINE
    union { uint32_t i; float f; } conv;
    conv.i = x;
    return conv.f;
}

static inline uint32_t float_to_uint32(float x) {
#pragma HLS INLINE
    union { uint32_t i; float f; } conv;
    conv.f = x;
    return conv.i;
}

static inline void unpack_bus16(const bus_t &word, float lane[PACK_SIZE]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
        lane[i] = uint32_to_float((uint32_t)word.range(31 + i * 32, i * 32));
    }
}

static inline bus_t pack_bus16(const float lane[PACK_SIZE]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    bus_t word = 0;
    for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
        word.range(31 + i * 32, i * 32) = float_to_uint32(lane[i]);
    }
    return word;
}

static inline float fp32_sum8_tree(const float value[8])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
    float s4[4];
    float s2[2];
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
FP32_SUM8_L1:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        s4[i] = value[2 * i] + value[2 * i + 1];
    }
FP32_SUM8_L2:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        s2[i] = s4[2 * i] + s4[2 * i + 1];
    }
    return s2[0] + s2[1];
}

static inline float fp32_sum4_tree(const float value[4])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
    float s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
FP32_SUM4_L1:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        s2[i] = value[2 * i] + value[2 * i + 1];
    }
    return s2[0] + s2[1];
}

static inline float fp32_dot8_tree(
    const float lhs[8],
    const float rhs[8])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
    float product[8];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp
FP32_DOT8_MUL:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }
    return fp32_sum8_tree(product);
}

static inline float fp32_dot4_tree(
    const float lhs[4],
    const float rhs[4])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
    float product[4];
    float s2[2];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp
FP32_DOT4_MUL:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }
FP32_DOT4_L1:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        s2[i] = product[2 * i] + product[2 * i + 1];
    }
    return s2[0] + s2[1];
}

#endif // BERT_MATH_H
