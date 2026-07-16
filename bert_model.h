#ifndef BERT_MODEL_H
#define BERT_MODEL_H

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <stdint.h>

#if !defined(BERT_EXACT_FP_ORDER) && !defined(BERT_BALANCED_FP_TREE)
#define BERT_BALANCED_FP_TREE
#endif

typedef ap_uint<512> bus_t;
typedef hls::stream<bus_t> hidden_stream_t;

// ============================================================================
// BERT Base / packed-bus configuration
// ============================================================================
static const int NUM_LAYERS   = 12;
static const int SEQ_LEN      = 128;
static const int HIDDEN       = 768;
static const int HIDDEN_SIZE  = 768;
static const int NUM_HEADS    = 12;
static const int HEAD_DIM     = 64;
static const int HEAD_PAR     = 2;
static const int HEAD_GROUPS  = NUM_HEADS / HEAD_PAR;
static const int NUM_HEAD_GROUPS = HEAD_GROUPS;
static const int GROUP_DIM    = HEAD_PAR * HEAD_DIM;
static const int FFN_DIM      = 3072;

static const int PACK_SIZE    = 16;                 // 16 fp32 / 512-bit word
static const int PACKS        = HIDDEN / PACK_SIZE; // 48
static const int TOKEN_WORDS  = SEQ_LEN * PACKS;    // 6144
static const int TOTAL_ROWS   = NUM_HEADS * SEQ_LEN;

static const int ATTN_PROJ_TILE_K = 64;
static const int ATTN_PROJ_TILE_O = 64;
static const int ATTN_OUT_TILE    = 32;

// Attention softmax-local packing
static const int COLS         = 128;
static const int LUT_SIZE     = 256;
static const int PACK_FACTOR  = 8;
static const int COL_PACKS    = COLS / PACK_FACTOR;

static_assert(HIDDEN == HIDDEN_SIZE, "HIDDEN and HIDDEN_SIZE must match");
static_assert((NUM_LAYERS % 2) == 0, "ping-pong top assumes an even number of BERT layers");
static_assert(HIDDEN == NUM_HEADS * HEAD_DIM, "HIDDEN must equal NUM_HEADS * HEAD_DIM");
static_assert((NUM_HEADS % HEAD_PAR) == 0, "NUM_HEADS must be divisible by HEAD_PAR");
static_assert((HIDDEN % PACK_SIZE) == 0, "HIDDEN must be divisible by PACK_SIZE");
static_assert((FFN_DIM % PACK_SIZE) == 0, "FFN_DIM must be divisible by PACK_SIZE");
static_assert((GROUP_DIM % PACK_SIZE) == 0, "GROUP_DIM must be pack-aligned");
static_assert((ATTN_PROJ_TILE_K % PACK_SIZE) == 0, "ATTN_PROJ_TILE_K must be pack-aligned");
static_assert((ATTN_PROJ_TILE_O % PACK_SIZE) == 0, "ATTN_PROJ_TILE_O must be pack-aligned");
static_assert((GROUP_DIM % ATTN_PROJ_TILE_O) == 0, "GROUP_DIM must be divisible by ATTN_PROJ_TILE_O");
static_assert((HIDDEN_SIZE % ATTN_OUT_TILE) == 0, "HIDDEN_SIZE must be divisible by ATTN_OUT_TILE");
#ifdef BERT_STREAMING_DATAFLOW
static_assert(HEAD_PAR == 2, "streaming dataflow requires two heads per group");
static_assert(ATTN_PROJ_TILE_K == 64, "streaming dataflow locks attention K tile");
static_assert(ATTN_PROJ_TILE_O == 64, "streaming dataflow locks attention O tile");
static_assert(ATTN_OUT_TILE == 32, "streaming dataflow locks attention output tile");
#endif

// ============================================================================
// Per-layer packed DDR footprints
// ============================================================================
static const int EMB_NORM_PACKS       = PACKS;                   // 48
static const int ADD_NORM_PACKS       = PACKS;                   // 48

static const int ATTN_W_FLOATS        = HIDDEN * HIDDEN;           // 589,824
static const int ATTN_B_FLOATS        = HIDDEN;                    // 768
static const int ATTN_W_PACKS         = ATTN_W_FLOATS / PACK_SIZE; // 36,864
static const int ATTN_B_PACKS         = ATTN_B_FLOATS / PACK_SIZE; // 48
static const int ATTN_ALL_W_PACKS     = 4 * ATTN_W_PACKS;          // q/k/v/o
static const int ATTN_ALL_B_PACKS     = 4 * ATTN_B_PACKS;

static const int FFN_UP_W_FLOATS      = HIDDEN * FFN_DIM;           // 2,359,296
static const int FFN_UP_B_FLOATS      = FFN_DIM;                    // 3,072
static const int FFN_DN_W_FLOATS      = FFN_DIM * HIDDEN;           // 2,359,296
static const int FFN_DN_B_FLOATS      = HIDDEN;                     // 768

static const int FFN_UP_W_PACKS       = FFN_UP_W_FLOATS / PACK_SIZE; // 147,456
static const int FFN_UP_B_PACKS       = FFN_UP_B_FLOATS / PACK_SIZE; // 192
static const int FFN_DN_W_PACKS       = FFN_DN_W_FLOATS / PACK_SIZE; // 147,456
static const int FFN_DN_B_PACKS       = FFN_DN_B_FLOATS / PACK_SIZE; // 48

static const int LAYER_NORM_GAMMA_PACKS = PACKS;
static const int LAYER_NORM_BETA_PACKS  = PACKS;

// ============================================================================
// Internal helper packet used by attention softmax path
// ============================================================================
struct row_pkt_t {
    float v[PACK_FACTOR];
};

// ============================================================================
// Bit conversion helpers
// ============================================================================
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

// Balanced FP32 dot product used by the U250 projection engines.  A fixed
// 16-lane tree avoids the long serial adder chain produced by `sum += a*b`
// after full unrolling.  The reassociation changes only FP32 reduction order;
// neither operands nor results use a reduced-precision representation.
static inline float fp32_sum16_tree(const float value[16])
{
#pragma HLS INLINE
    float s8[8];
    float s4[4];
    float s2[2];
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
FP32_SUM16_L1:
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        s8[i] = value[2 * i] + value[2 * i + 1];
    }
FP32_SUM16_L2:
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        s4[i] = s8[2 * i] + s8[2 * i + 1];
    }
FP32_SUM16_L3:
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        s2[i] = s4[2 * i] + s4[2 * i + 1];
    }
    return s2[0] + s2[1];
}

static inline float fp32_sum32_tree(const float value[32])
{
#pragma HLS INLINE
    float s16[16];
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
#pragma HLS ARRAY_PARTITION variable=s16 complete dim=1
FP32_SUM32_L1:
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        s16[i] = value[2 * i] + value[2 * i + 1];
    }
    return fp32_sum16_tree(s16);
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

static inline float fp32_dot16_tree(
    const float lhs[16],
    const float rhs[16])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1

    float product[16];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp
FP32_DOT16_MUL:
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }
    return fp32_sum16_tree(product);
}

static inline float fp32_dot32_tree(
    const float lhs[32],
    const float rhs[32])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
    float product[32];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp
FP32_DOT32_MUL:
    for (int i = 0; i < 32; ++i) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }
    return fp32_sum32_tree(product);
}

// ============================================================================
// Stream routing helpers
// ============================================================================
static inline void duplicate_stream(
    hidden_stream_t &src,
    hidden_stream_t &dst_main,
    hidden_stream_t &dst_residual,
    const int n_words = TOKEN_WORDS)
{
#pragma HLS INLINE off
DUPLICATE_STREAM:
    for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
        bus_t word = src.read();
        dst_main.write(word);
        dst_residual.write(word);
    }
}

static inline void load_hidden_stream(
    const bus_t *src,
    hidden_stream_t &dst,
    const int n_words = TOKEN_WORDS)
{
#pragma HLS INLINE off
LOAD_HIDDEN_STREAM:
    for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
        dst.write(src[i]);
    }
}

static inline void store_hidden_stream(
    hidden_stream_t &src,
    bus_t *dst,
    const int n_words = TOKEN_WORDS)
{
#pragma HLS INLINE off
STORE_HIDDEN_STREAM:
    for (int i = 0; i < n_words; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = src.read();
    }
}

// ============================================================================
// Matrix-resident residual add + layernorm
// ============================================================================
// Attention and FFN already materialize both the residual input and sublayer
// output as local FP32 matrices.  Applying residual layernorm directly on those
// matrices avoids an extra packed sublayer buffer, an extra stream adapter, and
// one top-level add_norm stage per sublayer.  Math remains FP32 and follows the
// same reduction structure as add_norm_stream.
static void matrix_add_norm_to_stream(
    float residual[SEQ_LEN][HIDDEN_SIZE],
    float sublayer[SEQ_LEN][HIDDEN_SIZE],
    const bus_t *gamma_in,
    const bus_t *beta_in,
    hidden_stream_t &output_stream)
{
#pragma HLS INLINE off

    static const int RED_GROUPS = 4;
    static const int RED_BANKS = RED_GROUPS * PACK_SIZE;
    // Match the physical 16-bank hidden-state layout.  The former 8-lane
    // chunk left `base` as a runtime bank selector in Vitis HLS 2021.2 and
    // generated roughly 180k LUT of crossbar logic across the two norm paths.
    static const int MATRIX_LN_PAR = PACK_SIZE;
    static const float INV_HIDDEN = 1.0f / HIDDEN_SIZE;
    static_assert((PACK_SIZE % MATRIX_LN_PAR) == 0, "PACK_SIZE must be divisible by MATRIX_LN_PAR");

    bus_t gamma_cache[PACKS];
    bus_t beta_cache[PACKS];
#pragma HLS BIND_STORAGE variable=gamma_cache type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=beta_cache type=RAM_2P impl=BRAM

MATRIX_LN_LOAD_GAMMA:
    for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
        gamma_cache[p] = gamma_in[p];
    }

MATRIX_LN_LOAD_BETA:
    for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
        beta_cache[p] = beta_in[p];
    }

MATRIX_LN_TOKEN:
    for (int s = 0; s < SEQ_LEN; ++s) {
        float sum_banks[RED_BANKS];
        float sumsq_banks[RED_BANKS];
#pragma HLS ARRAY_PARTITION variable=sum_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=sumsq_banks complete dim=1

    MATRIX_LN_INIT_BANKS:
        for (int b = 0; b < RED_BANKS; ++b) {
#pragma HLS UNROLL
            sum_banks[b] = 0.0f;
            sumsq_banks[b] = 0.0f;
        }

    MATRIX_LN_ACC_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            const int bank_group = p & (RED_GROUPS - 1);

        MATRIX_LN_ACC_LANE:
            for (int lane = 0; lane < PACK_SIZE; ++lane) {
#pragma HLS UNROLL
                const int col = p * PACK_SIZE + lane;
                const int bank = bank_group * PACK_SIZE + lane;
                const float x = residual[s][col] + sublayer[s][col];
                sum_banks[bank] += x;
                sumsq_banks[bank] += x * x;
            }
        }

        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;

    MATRIX_LN_REDUCE:
        for (int b = 0; b < RED_BANKS; ++b) {
#pragma HLS PIPELINE II=1
            total_sum += sum_banks[b];
            total_sum_sq += sumsq_banks[b];
        }

        const float mean = total_sum * INV_HIDDEN;
        const float variance = (total_sum_sq * INV_HIDDEN) - (mean * mean);
        const float inv_stddev = hls::rsqrtf(variance + 1e-12f);

    MATRIX_LN_APPLY_PACK:
        for (int p = 0; p < PACKS; ++p) {
            float g_lane[PACK_SIZE];
            float b_lane[PACK_SIZE];
            float out_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=g_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_lane complete dim=1

            unpack_bus16(gamma_cache[p], g_lane);
            unpack_bus16(beta_cache[p], b_lane);

#pragma HLS PIPELINE II=1
        MATRIX_LN_APPLY_LANE:
            for (int lane = 0; lane < PACK_SIZE; ++lane) {
#pragma HLS UNROLL
                const int col = p * PACK_SIZE + lane;
                const float x = residual[s][col] + sublayer[s][col];
                const float norm_val = (x - mean) * inv_stddev;
                out_lane[lane] = g_lane[lane] * norm_val + b_lane[lane];
            }

            output_stream.write(pack_bus16(out_lane));
        }
    }
}

// ============================================================================
// Shared stream-based module interfaces
// NOTE:
//  - Data path between modules is always hidden_stream_t (512-bit packed stream)
//  - Weight pointers stay on DDR/HBM and are resolved by top.cpp per layer
//  - Attention / FFN weights are passed as packed 512-bit pointers and cast
//    locally back to float* inside each kernel wrapper, preserving math cores.
// ============================================================================
void bert_embedding_stream(
    const int   *input_ids,
    const int   *token_type_ids,
    const bus_t *token_emb,
    const bus_t *pos_emb,
    const bus_t *seg_emb,
    const bus_t *gamma_in,
    const bus_t *beta_in,
    hidden_stream_t &output_stream);

void bert_self_attention_stream(
    hidden_stream_t &input_hidden,
    const bus_t     *q_w,
    const bus_t     *q_b,
    const bus_t     *k_w,
    const bus_t     *k_b,
    const bus_t     *v_w,
    const bus_t     *v_b,
    const bus_t     *dense_w,
    const bus_t     *dense_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    const int       *attention_mask,
    hidden_stream_t &output_hidden);

// --------------------------------------------------------------------------
// Split-kernel stage interfaces.
// These preserve the FP32 math cores above, but make the large attention and
// FFN working sets private to a single Vitis kernel.  qkv_stream carries
// Q, then K, then V for every (head-group, token, 512-bit pack).
// --------------------------------------------------------------------------
void bert_qkv_stage(
    hidden_stream_t &input_hidden,
    const bus_t     *q_w,
    const bus_t     *q_b,
    const bus_t     *k_w,
    const bus_t     *k_b,
    const bus_t     *v_w,
    const bus_t     *v_b,
    hidden_stream_t &qkv_stream);

void bert_attn_core_stage(
    hidden_stream_t &qkv_stream,
    const int       *attention_mask,
    hidden_stream_t &context_stream);

void bert_attn_out_norm_stage(
    hidden_stream_t &context_stream,
    const bus_t     *residual_in,
    const bus_t     *dense_w,
    const bus_t     *dense_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &attn_mid_stream);

void add_norm_stream(
    hidden_stream_t &residual_in,
    hidden_stream_t &sublayer_in,
    const bus_t     *gamma_in,
    const bus_t     *beta_in,
    hidden_stream_t &output_stream);

void bert_ffn_stream(
    hidden_stream_t &input_hidden,
    const bus_t     *up_w,
    const bus_t     *up_b,
    const bus_t     *down_w,
    const bus_t     *down_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &output_hidden);

void bert_ffn_up_gelu_stage(
    hidden_stream_t &attn_mid_stream,
    const bus_t     *up_w,
    const bus_t     *up_b,
    hidden_stream_t &gelu_stream);

void bert_ffn_down_norm_stage(
    hidden_stream_t &gelu_stream,
    hidden_stream_t &attn_mid_residual_stream,
    const bus_t     *down_w,
    const bus_t     *down_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &hidden_out_stream);

#endif // BERT_MODEL_H
