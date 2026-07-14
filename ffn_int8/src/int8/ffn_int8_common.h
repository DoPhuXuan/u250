#ifndef FFN_INT8_COMMON_H
#define FFN_INT8_COMMON_H

#include <ap_int.h>
#include <hls_stream.h>

#ifndef FFN_I8_SEQ_LEN
#define FFN_I8_SEQ_LEN 128
#endif

#ifndef FFN_I8_HIDDEN_SIZE
#define FFN_I8_HIDDEN_SIZE 768
#endif

#ifndef FFN_I8_INTERMEDIATE_SIZE
#define FFN_I8_INTERMEDIATE_SIZE 3072
#endif
#define FFN_I8_AXI_BITS 512
#define FFN_I8_F32_PER_WORD 16
#define FFN_I8_I8_PER_WORD 64
#define FFN_I8_STREAM_TILE 64

#ifndef FFN_I8_K_PAR
#define FFN_I8_K_PAR 16
#endif

#ifndef FFN_I8_OUT_PAR
#define FFN_I8_OUT_PAR 16
#endif

#ifndef FFN_I8_GELU_PROFILE
#define FFN_I8_GELU_PROFILE 0
#endif

#if (FFN_I8_K_PAR != 16) && (FFN_I8_K_PAR != 32) && (FFN_I8_K_PAR != 64)
#error "FFN_I8_K_PAR must be 16, 32, or 64"
#endif

#if (FFN_I8_OUT_PAR != 16) && (FFN_I8_OUT_PAR != 32)
#error "FFN_I8_OUT_PAR must be 16 or 32"
#endif

#if (FFN_I8_GELU_PROFILE != 0) && (FFN_I8_GELU_PROFILE != 2)
#error "FFN_I8_GELU_PROFILE must be 0 (FP32 PWL) or 2 (requantize + LUT)"
#endif

#if (FFN_I8_STREAM_TILE % FFN_I8_K_PAR) != 0
#error "FFN_I8_STREAM_TILE must be divisible by FFN_I8_K_PAR"
#endif

#if (FFN_I8_STREAM_TILE % FFN_I8_OUT_PAR) != 0
#error "FFN_I8_STREAM_TILE must be divisible by FFN_I8_OUT_PAR"
#endif

#if (FFN_I8_HIDDEN_SIZE % FFN_I8_K_PAR) != 0
#error "FFN_I8_HIDDEN_SIZE must be divisible by FFN_I8_K_PAR"
#endif

#if (FFN_I8_INTERMEDIATE_SIZE % FFN_I8_STREAM_TILE) != 0
#error "FFN_I8_INTERMEDIATE_SIZE must be divisible by FFN_I8_STREAM_TILE"
#endif

#if ((FFN_I8_K_PAR * FFN_I8_OUT_PAR) % FFN_I8_I8_PER_WORD) != 0
#error "One weight tile must contain an integer number of AXI words"
#endif

#define FFN_I8_WEIGHT_WORDS_PER_TILE ((FFN_I8_K_PAR * FFN_I8_OUT_PAR) / FFN_I8_I8_PER_WORD)

typedef ap_uint<FFN_I8_AXI_BITS> packed_f32x16_t;
typedef ap_uint<FFN_I8_AXI_BITS> packed_i8x64_t;
typedef hls::stream<packed_f32x16_t> packed_f32_stream_t;
typedef hls::stream<packed_i8x64_t> packed_i8_stream_t;
typedef ap_int<8> ffn_i8_t;
typedef ap_int<32> ffn_acc_t;
typedef ap_uint<32> ffn_i8_requant_param_t;

static float ffn_i8_bits_to_float(ap_uint<32> bits) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } conversion;
    conversion.u = (unsigned int)bits;
    return conversion.f;
}

static ap_uint<32> ffn_i8_float_to_bits(float value) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } conversion;
    conversion.f = value;
    return conversion.u;
}

static int ffn_i8_round_nearest_even(float value) {
#pragma HLS INLINE
    int base = (int)value;
    float fraction = value - (float)base;
    bool base_is_odd = (base & 1) != 0;
    if ((fraction > 0.5f) || ((fraction == 0.5f) && base_is_odd)) {
        return base + 1;
    }
    if ((fraction < -0.5f) || ((fraction == -0.5f) && base_is_odd)) {
        return base - 1;
    }
    return base;
}

static ffn_i8_t ffn_i8_quantize(float value, float inverse_scale) {
#pragma HLS INLINE
    float scaled = value * inverse_scale;
    if (scaled >= 127.0f) {
        return (ffn_i8_t)127;
    }
    if (scaled <= -127.0f) {
        return (ffn_i8_t)-127;
    }
    return (ffn_i8_t)ffn_i8_round_nearest_even(scaled);
}

static float ffn_i8_gelu_pwl(float value) {
#pragma HLS INLINE
    if (value <= -3.0f) {
        return 0.0f;
    }
    if (value >= 3.0f) {
        return value;
    }
    return value * (0.5f + value * (1.0f / 6.0f));
}

static ffn_i8_t ffn_i8_unpack_byte(packed_i8x64_t word, int lane) {
#pragma HLS INLINE
    ffn_i8_t value;
    value.range(7, 0) = word.range(lane * 8 + 7, lane * 8);
    return value;
}

static ffn_i8_t ffn_i8_requantize_dsp(
    ffn_acc_t accumulator,
    ffn_i8_requant_param_t parameter) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    ap_int<18> multiplier;
    multiplier.range(17, 0) = parameter.range(17, 0);
    ap_uint<6> right_shift = parameter.range(23, 18);

    // UP's checked accumulator bound fits 27 signed bits, allowing one
    // DSP48E2 to implement this 27x18 requantization.
    ap_int<27> bounded_accumulator = (ap_int<27>)accumulator;
    ap_int<45> product;
#pragma HLS BIND_OP variable=product op=mul impl=dsp latency=1
    product = bounded_accumulator * multiplier;

    bool negative = product < 0;
    ap_uint<45> magnitude;
    if (negative) {
        ap_int<46> extended_product = (ap_int<46>)product;
        magnitude = (ap_uint<45>)(-extended_product);
    } else {
        magnitude = (ap_uint<45>)product;
    }
    ap_uint<45> rounded_magnitude = 0;
    if (right_shift == 0) {
        rounded_magnitude = magnitude;
    } else if (right_shift < 45) {
        ap_uint<45> quotient = magnitude >> right_shift;
        ap_uint<45> remainder = magnitude - (quotient << right_shift);
        ap_uint<45> half = ((ap_uint<45>)1) << (right_shift - 1);
        bool increment = remainder > half || (remainder == half && quotient[0]);
        rounded_magnitude = quotient;
        if (increment) {
            rounded_magnitude = quotient + 1;
        }
    }

    ap_int<46> rounded;
    if (negative) {
        rounded = -(ap_int<46>)rounded_magnitude;
    } else {
        rounded = (ap_int<46>)rounded_magnitude;
    }
    if (rounded > 127) return (ffn_i8_t)127;
    if (rounded < -127) return (ffn_i8_t)-127;
    return (ffn_i8_t)rounded;
}

static void ffn_i8_load_weight_tile(
    const packed_i8x64_t *weights,
    int output_tile,
    int k_tile,
    int total_k_tiles,
    ffn_i8_t tile[FFN_I8_OUT_PAR][FFN_I8_K_PAR]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=tile complete dim=2
    int base = (output_tile * total_k_tiles + k_tile) * FFN_I8_WEIGHT_WORDS_PER_TILE;

ffn_i8_load_weight_word:
    for (int word_index = 0; word_index < FFN_I8_WEIGHT_WORDS_PER_TILE; ++word_index) {
#pragma HLS PIPELINE II=1
        packed_i8x64_t word = weights[base + word_index];
    ffn_i8_load_weight_lane:
        for (int byte_lane = 0; byte_lane < FFN_I8_I8_PER_WORD; ++byte_lane) {
#pragma HLS UNROLL
            int linear = word_index * FFN_I8_I8_PER_WORD + byte_lane;
            int k_lane = linear / FFN_I8_OUT_PAR;
            int output_lane = linear % FFN_I8_OUT_PAR;
            tile[output_lane][k_lane] = ffn_i8_unpack_byte(word, byte_lane);
        }
    }
}

struct ffn_i8_product_pair_t {
    ap_int<16> first;
    ap_int<16> second;
};

struct ffn_i8_dot_pair_t {
    ffn_acc_t first;
    ffn_acc_t second;
};

// Pack two weights that share one activation into one DSP multiply:
//   lhs * (rhs_first + rhs_second * 2^17).
// A 17-bit lane is wide enough for every signed INT8 product.  A negative
// low product borrows one from the high lane, so compensate that borrow when
// the products are unpacked.
static ffn_i8_product_pair_t ffn_i8_multiply_pair_dsp(
    ffn_i8_t lhs,
    ffn_i8_t rhs_first,
    ffn_i8_t rhs_second) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    ap_int<27> high_weight = (ap_int<27>)rhs_second;
    ap_int<27> packed_weight;
    ap_int<35> packed_product;
#pragma HLS BIND_OP variable=packed_product op=mul impl=dsp latency=1
    high_weight <<= 17;
    packed_weight = high_weight + (ap_int<27>)rhs_first;
    packed_product = packed_weight * (ap_int<8>)lhs;

    ap_int<17> low_product;
    low_product.range(16, 0) = packed_product.range(16, 0);
    ap_int<18> high_product = (ap_int<18>)(packed_product >> 17);

    ffn_i8_product_pair_t result;
    result.first = (ap_int<16>)low_product;
    ap_int<18> corrected_high_product = high_product;
    if (low_product < 0) {
        corrected_high_product = high_product + 1;
    }
    result.second = (ap_int<16>)corrected_high_product;
    return result;
}

static ffn_acc_t ffn_i8_reduce_products(
    const ap_int<16> products_i16[FFN_I8_K_PAR]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=products_i16 complete dim=1
    ffn_acc_t products[FFN_I8_K_PAR];
    ffn_acc_t sum16[16];
    ffn_acc_t sum8[8];
    ffn_acc_t sum4[4];
    ffn_acc_t sum2[2];
#pragma HLS ARRAY_PARTITION variable=products complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum16 complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum2 complete dim=1

ffn_i8_widen_products:
    for (int lane = 0; lane < FFN_I8_K_PAR; ++lane) {
#pragma HLS UNROLL
        products[lane] = (ffn_acc_t)products_i16[lane];
    }

#if FFN_I8_K_PAR == 64
    ffn_acc_t sum32[32];
#pragma HLS ARRAY_PARTITION variable=sum32 complete dim=1
ffn_i8_dot_sum32:
    for (int lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
        sum32[lane] = products[2 * lane] + products[2 * lane + 1];
    }
ffn_i8_dot_sum16_from32:
    for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
        sum16[lane] = sum32[2 * lane] + sum32[2 * lane + 1];
    }
#elif FFN_I8_K_PAR == 32
ffn_i8_dot_sum16_from_products:
    for (int lane = 0; lane < 16; ++lane) {
#pragma HLS UNROLL
        sum16[lane] = products[2 * lane] + products[2 * lane + 1];
    }
#else
ffn_i8_dot_sum8_from_products:
    for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
        sum8[lane] = products[2 * lane] + products[2 * lane + 1];
    }
#endif

#if FFN_I8_K_PAR >= 32
ffn_i8_dot_sum8_from16:
    for (int lane = 0; lane < 8; ++lane) {
#pragma HLS UNROLL
        sum8[lane] = sum16[2 * lane] + sum16[2 * lane + 1];
    }
#endif
ffn_i8_dot_sum4:
    for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
        sum4[lane] = sum8[2 * lane] + sum8[2 * lane + 1];
    }
ffn_i8_dot_sum2:
    for (int lane = 0; lane < 2; ++lane) {
#pragma HLS UNROLL
        sum2[lane] = sum4[2 * lane] + sum4[2 * lane + 1];
    }
    return sum2[0] + sum2[1];
}

static ffn_i8_dot_pair_t ffn_i8_dot_pair(
    const ffn_i8_t lhs[FFN_I8_K_PAR],
    const ffn_i8_t rhs_first[FFN_I8_K_PAR],
    const ffn_i8_t rhs_second[FFN_I8_K_PAR]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs_first complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs_second complete dim=1
    ap_int<16> products_first[FFN_I8_K_PAR];
    ap_int<16> products_second[FFN_I8_K_PAR];
#pragma HLS ARRAY_PARTITION variable=products_first complete dim=1
#pragma HLS ARRAY_PARTITION variable=products_second complete dim=1

ffn_i8_dot_pair_products:
    for (int lane = 0; lane < FFN_I8_K_PAR; ++lane) {
#pragma HLS UNROLL
        ffn_i8_product_pair_t products =
            ffn_i8_multiply_pair_dsp(lhs[lane], rhs_first[lane], rhs_second[lane]);
        products_first[lane] = products.first;
        products_second[lane] = products.second;
    }

    ffn_i8_dot_pair_t result;
    result.first = ffn_i8_reduce_products(products_first);
    result.second = ffn_i8_reduce_products(products_second);
    return result;
}

#endif
