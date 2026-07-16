#include "bert_model.h"
#include <cfloat>

#ifndef ATTN_MASK_PENALTY
#define ATTN_MASK_PENALTY (-10000.0f)
#endif

#ifndef ATTN_SCALE
#define ATTN_SCALE (0.125f)
#endif

#ifdef BERT_ROUTE_FRIENDLY
static const int ATTN_ARRAY_FACTOR = 4;
static const int ATTN_OUT_PAR = 4;
#else
static const int ATTN_ARRAY_FACTOR = 16;
static const int ATTN_OUT_PAR = 16;
#endif
static const int ATTN_HEAD_FACTOR = HEAD_PAR * ATTN_ARRAY_FACTOR;
static const int ATTN_STORAGE_FACTOR = ATTN_HEAD_FACTOR;
static const int SOFTMAX_REDUCE_GROUPS = 8;
static const int ATTN_QUERY_PAR = 2;

#ifdef BERT_STREAMING_DATAFLOW
struct attention_pair_pkt_t {
    float q0;
    float q1;
};
#endif

static_assert((SEQ_LEN % ATTN_QUERY_PAR) == 0, "SEQ_LEN must be divisible by ATTN_QUERY_PAR");
static_assert(ATTN_QUERY_PAR == 2, "attention split currently implements two-query grouping");
static_assert((SOFTMAX_REDUCE_GROUPS & (SOFTMAX_REDUCE_GROUPS - 1)) == 0,
              "SOFTMAX_REDUCE_GROUPS must be a power of two");
#if defined(BERT_ROUTE_FRIENDLY) && defined(BERT_STREAMING_DATAFLOW)
#error "BERT_ROUTE_FRIENDLY is not compatible with BERT_STREAMING_DATAFLOW"
#endif
#ifdef BERT_STREAMING_DATAFLOW
static_assert(ATTN_ARRAY_FACTOR == 16, "streaming dataflow locks attention lanes");
static_assert(ATTN_OUT_PAR == 16, "streaming dataflow locks attention outputs");
static_assert(ATTN_QUERY_PAR == 2, "streaming dataflow locks query grouping");
#endif

static inline float fp32_attn_dot_tree(
    const float lhs[ATTN_ARRAY_FACTOR],
    const float rhs[ATTN_ARRAY_FACTOR])
{
#pragma HLS INLINE
#ifdef BERT_ROUTE_FRIENDLY
    return fp32_dot4_tree(lhs, rhs);
#else
    return fp32_dot16_tree(lhs, rhs);
#endif
}

static inline float fp32_attn_sum_tree(const float value[ATTN_ARRAY_FACTOR])
{
#pragma HLS INLINE
#ifdef BERT_ROUTE_FRIENDLY
    return fp32_sum4_tree(value);
#else
    return fp32_sum16_tree(value);
#endif
}

static int build_active_index_list(
    const int attention_mask[SEQ_LEN],
    int active_indices[SEQ_LEN])
{
#pragma HLS INLINE

    int active_count = 0;

BUILD_ACTIVE_LIST:
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1
        if (attention_mask[i] != 0) {
            active_indices[active_count] = i;
            ++active_count;
        }
    }

    return active_count;
}

// ============================================================================
// Stream adapters
// ============================================================================

static void load_hidden_stream_to_matrix(
    hidden_stream_t &input_stream,
    float hidden[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off

LOAD_HIDDEN_ROWS:
    for (int s = 0; s < SEQ_LEN; ++s) {
    LOAD_HIDDEN_PACKS:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            bus_t word = input_stream.read();
            unpack_bus16(word, lane);

        LOAD_HIDDEN_LANES:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                hidden[s][p * PACK_SIZE + i] = lane[i];
            }
        }
    }
}

// ============================================================================
// Packed parameter loaders
// ============================================================================

static void load_qkv_bias_tile(
    const bus_t *q_b,
    const bus_t *k_b,
    const bus_t *v_b,
    const int out_base,
    float q_bias[ATTN_PROJ_TILE_O],
    float k_bias[ATTN_PROJ_TILE_O],
    float v_bias[ATTN_PROJ_TILE_O])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=q_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_bias complete dim=1

    static_assert((ATTN_PROJ_TILE_O % PACK_SIZE) == 0,
                  "ATTN_PROJ_TILE_O must be pack-aligned");

LOAD_QKV_BIAS_PACKS:
    for (int p = 0; p < (ATTN_PROJ_TILE_O / PACK_SIZE); ++p) {
#pragma HLS PIPELINE II=1
        float q_lane[PACK_SIZE];
        float k_lane[PACK_SIZE];
        float v_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=q_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1

        const int word_idx = (out_base / PACK_SIZE) + p;
        unpack_bus16(q_b[word_idx], q_lane);
        unpack_bus16(k_b[word_idx], k_lane);
        unpack_bus16(v_b[word_idx], v_lane);

    LOAD_QKV_BIAS_LANES:
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
            const int idx = p * PACK_SIZE + i;
            q_bias[idx] = q_lane[i];
            k_bias[idx] = k_lane[i];
            v_bias[idx] = v_lane[i];
        }
    }
}

static void load_qkv_weight_tile(
    const bus_t *q_w,
    const bus_t *k_w,
    const bus_t *v_w,
    const int out_base,
    const int in_base,
    float wq_tile[ATTN_PROJ_TILE_O][ATTN_PROJ_TILE_K],
    float wk_tile[ATTN_PROJ_TILE_O][ATTN_PROJ_TILE_K],
    float wv_tile[ATTN_PROJ_TILE_O][ATTN_PROJ_TILE_K])
{
#pragma HLS INLINE

    const int TILE_PACKS = (ATTN_PROJ_TILE_K + PACK_SIZE - 1) / PACK_SIZE;
    const int in_pack_base = in_base / PACK_SIZE;

LOAD_QKV_TILE_ROWS:
    for (int oo = 0; oo < ATTN_PROJ_TILE_O; ++oo) {
    LOAD_QKV_TILE_PACKS:
        for (int pk = 0; pk < TILE_PACKS; ++pk) {
#pragma HLS PIPELINE II=1
            float q_lane[PACK_SIZE];
            float k_lane[PACK_SIZE];
            float v_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=q_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1

            const int word_idx = (out_base + oo) * PACKS + in_pack_base + pk;
            unpack_bus16(q_w[word_idx], q_lane);
            unpack_bus16(k_w[word_idx], k_lane);
            unpack_bus16(v_w[word_idx], v_lane);

        LOAD_QKV_TILE_LANES:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int idx = pk * PACK_SIZE + i;
                if (idx < ATTN_PROJ_TILE_K) {
                    wq_tile[oo][idx] = q_lane[i];
                    wk_tile[oo][idx] = k_lane[i];
                    wv_tile[oo][idx] = v_lane[i];
                }
            }
        }
    }
}

static void load_output_weight_tile(
    const bus_t *dense_w,
    const int out_base,
    const int group_base,
    const int in_base,
    float wo_tile[ATTN_OUT_TILE][ATTN_PROJ_TILE_K])
{
#pragma HLS INLINE

    const int TILE_PACKS = (ATTN_PROJ_TILE_K + PACK_SIZE - 1) / PACK_SIZE;
    const int in_pack_base = (group_base + in_base) / PACK_SIZE;

LOAD_OUT_TILE_ROWS:
    for (int oo = 0; oo < ATTN_OUT_TILE; ++oo) {
    LOAD_OUT_TILE_PACKS:
        for (int pk = 0; pk < TILE_PACKS; ++pk) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1

            const int word_idx = (out_base + oo) * PACKS + in_pack_base + pk;
            unpack_bus16(dense_w[word_idx], lane);

        LOAD_OUT_TILE_LANES:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int idx = pk * PACK_SIZE + i;
                if (idx < ATTN_PROJ_TILE_K) {
                    wo_tile[oo][idx] = lane[i];
                }
            }
        }
    }
}

static void load_output_bias_tile(
    const bus_t *dense_b,
    const int out_base,
    float bias_tile[ATTN_OUT_TILE])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1

    static_assert((ATTN_OUT_TILE % PACK_SIZE) == 0,
                  "ATTN_OUT_TILE must be pack-aligned");

LOAD_OUT_BIAS_PACKS:
    for (int p = 0; p < (ATTN_OUT_TILE / PACK_SIZE); ++p) {
#pragma HLS PIPELINE II=1
        float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1

        unpack_bus16(dense_b[(out_base / PACK_SIZE) + p], lane);

    LOAD_OUT_BIAS_LANES:
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
            bias_tile[p * PACK_SIZE + i] = lane[i];
        }
    }
}

// ============================================================================
// Projection helpers
// ============================================================================

static void project_qkv_head_group(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    const bus_t *q_w,
    const bus_t *q_b,
    const bus_t *k_w,
    const bus_t *k_b,
    const bus_t *v_w,
    const bus_t *v_b,
    const int head_group,
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable=input_hidden cyclic factor=ATTN_STORAGE_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2

    const int group_base = head_group * GROUP_DIM;

    float wq_tile[ATTN_PROJ_TILE_O][ATTN_PROJ_TILE_K];
    float wk_tile[ATTN_PROJ_TILE_O][ATTN_PROJ_TILE_K];
    float wv_tile[ATTN_PROJ_TILE_O][ATTN_PROJ_TILE_K];
    float q_acc[SEQ_LEN][ATTN_PROJ_TILE_O];
    float k_acc[SEQ_LEN][ATTN_PROJ_TILE_O];
    float v_acc[SEQ_LEN][ATTN_PROJ_TILE_O];
    float q_bias[ATTN_PROJ_TILE_O];
    float k_bias[ATTN_PROJ_TILE_O];
    float v_bias[ATTN_PROJ_TILE_O];

#ifdef BERT_SCALED_VARIANT
#pragma HLS BIND_STORAGE variable=wq_tile type=RAM_2P impl=LUTRAM
#pragma HLS BIND_STORAGE variable=wk_tile type=RAM_2P impl=LUTRAM
#pragma HLS BIND_STORAGE variable=wv_tile type=RAM_2P impl=LUTRAM
#else
#pragma HLS BIND_STORAGE variable=wq_tile type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=wk_tile type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=wv_tile type=RAM_2P impl=BRAM
#endif
#pragma HLS BIND_STORAGE variable=q_acc   type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=k_acc   type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=v_acc   type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=wq_tile cyclic factor=ATTN_OUT_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wq_tile cyclic factor=ATTN_ARRAY_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=wk_tile cyclic factor=ATTN_OUT_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wk_tile cyclic factor=ATTN_ARRAY_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=wv_tile cyclic factor=ATTN_OUT_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wv_tile cyclic factor=ATTN_ARRAY_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=q_acc cyclic factor=ATTN_OUT_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=k_acc cyclic factor=ATTN_OUT_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=v_acc cyclic factor=ATTN_OUT_PAR dim=2

    static_assert((ATTN_PROJ_TILE_O % ATTN_OUT_PAR) == 0,
                  "ATTN_PROJ_TILE_O must be divisible by ATTN_OUT_PAR");
    static_assert((ATTN_PROJ_TILE_K % ATTN_ARRAY_FACTOR) == 0,
                  "ATTN_PROJ_TILE_K must be divisible by ATTN_ARRAY_FACTOR");

GROUP_OUT_TILE_LOOP:
    for (int ob = 0; ob < GROUP_DIM; ob += ATTN_PROJ_TILE_O) {
        const int out_base = group_base + ob;

        load_qkv_bias_tile(q_b, k_b, v_b, out_base, q_bias, k_bias, v_bias);

    INIT_ACC_SEQ:
        for (int s = 0; s < SEQ_LEN; ++s) {
        INIT_ACC_OUT:
            for (int oo = 0; oo < ATTN_PROJ_TILE_O; ++oo) {
#pragma HLS PIPELINE II=1
                q_acc[s][oo] = q_bias[oo];
                k_acc[s][oo] = k_bias[oo];
                v_acc[s][oo] = v_bias[oo];
            }
        }

    PROJ_K_TILE_LOOP:
        for (int kb = 0; kb < HIDDEN_SIZE; kb += ATTN_PROJ_TILE_K) {
            load_qkv_weight_tile(
                q_w, k_w, v_w,
                out_base,
                kb,
                wq_tile, wk_tile, wv_tile);

        PROJ_SEQ_LOOP:
            for (int s = 0; s < SEQ_LEN; ++s) {
            PROJ_OUT_LOOP:
                for (int oo = 0; oo < ATTN_PROJ_TILE_O; oo += ATTN_OUT_PAR) {
#pragma HLS PIPELINE II=1
                    float qsum[ATTN_OUT_PAR];
                    float ksum[ATTN_OUT_PAR];
                    float vsum[ATTN_OUT_PAR];
#pragma HLS ARRAY_PARTITION variable=qsum complete dim=1
#pragma HLS ARRAY_PARTITION variable=ksum complete dim=1
#pragma HLS ARRAY_PARTITION variable=vsum complete dim=1

                LOAD_PROJ_ACC:
                    for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                        qsum[jj] = q_acc[s][oo + jj];
                        ksum[jj] = k_acc[s][oo + jj];
                        vsum[jj] = v_acc[s][oo + jj];
                    }

                PROJ_IN_LOOP:
                    for (int ii = 0; ii < ATTN_PROJ_TILE_K; ii += ATTN_ARRAY_FACTOR) {
                        float x_vec[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1

                    LOAD_PROJ_X:
                        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                            x_vec[kk] = input_hidden[s][kb + ii + kk];
                        }

                    PROJ_OUT_VEC:
                        for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
#ifdef BERT_BALANCED_FP_TREE
                            float wq_vec[ATTN_ARRAY_FACTOR];
                            float wk_vec[ATTN_ARRAY_FACTOR];
                            float wv_vec[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=wq_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=wk_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=wv_vec complete dim=1
                        LOAD_PROJ_TREE_W:
                            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                                wq_vec[kk] = wq_tile[oo + jj][ii + kk];
                                wk_vec[kk] = wk_tile[oo + jj][ii + kk];
                                wv_vec[kk] = wv_tile[oo + jj][ii + kk];
                            }
                            qsum[jj] += fp32_attn_dot_tree(x_vec, wq_vec);
                            ksum[jj] += fp32_attn_dot_tree(x_vec, wk_vec);
                            vsum[jj] += fp32_attn_dot_tree(x_vec, wv_vec);
#else
                            float qtmp = qsum[jj];
                            float ktmp = ksum[jj];
                            float vtmp = vsum[jj];

                        PROJ_K_VEC:
                            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                                const float x = x_vec[kk];
                                qtmp += x * wq_tile[oo + jj][ii + kk];
                                ktmp += x * wk_tile[oo + jj][ii + kk];
                                vtmp += x * wv_tile[oo + jj][ii + kk];
                            }

                            qsum[jj] = qtmp;
                            ksum[jj] = ktmp;
                            vsum[jj] = vtmp;
#endif
                        }
                    }

                STORE_PROJ_ACC:
                    for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                        q_acc[s][oo + jj] = qsum[jj];
                        k_acc[s][oo + jj] = ksum[jj];
                        v_acc[s][oo + jj] = vsum[jj];
                    }
                }
            }
        }

    // Write one complete physical bank stripe per cycle.  Scalar `oo` writes
    // made HLS 2021.2 build a 16-way destination crossbar (394k LUT in the
    // supplied report) even though the array was already cyclically banked.
    WRITE_QKV_GROUP:
        for (int s = 0; s < SEQ_LEN; ++s) {
        WRITE_QKV_STRIPE:
            for (int oo_base = 0; oo_base < ATTN_PROJ_TILE_O;
                 oo_base += ATTN_HEAD_FACTOR) {
#pragma HLS PIPELINE II=1
            WRITE_QKV_BANK:
                for (int lane = 0; lane < ATTN_HEAD_FACTOR; ++lane) {
#pragma HLS UNROLL
                    const int oo = oo_base + lane;
                    q_group[s][ob + oo] = q_acc[s][oo];
                    k_group[s][ob + oo] = k_acc[s][oo];
                    v_group[s][ob + oo] = v_acc[s][oo];
                }
            }
        }
    }
}

// ============================================================================
// Split attention engines for one head group
// ============================================================================

static void compute_scores_head(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    const int active_indices[SEQ_LEN],
    const int active_count,
    const int h,
    const int q,
    float score_buf[SEQ_LEN])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2

    const int head_base = h * HEAD_DIM;
    static_assert((HEAD_DIM % ATTN_ARRAY_FACTOR) == 0, "HEAD_DIM must be divisible by ATTN_ARRAY_FACTOR");

SCORE_K_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const int k = active_indices[n];
        float dot_lane[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=dot_lane complete dim=1

    INIT_SCORE_LANES:
        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
            dot_lane[kk] = 0.0f;
        }

    SCORE_D_LOOP:
        for (int d = 0; d < HEAD_DIM; d += ATTN_ARRAY_FACTOR) {
        SCORE_D_LANE:
            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                dot_lane[kk] += q_group[q][head_base + d + kk] * k_group[k][head_base + d + kk];
            }
        }

        float dot0 = 0.0f;
    REDUCE_SCORE_LANES:
        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
            dot0 += dot_lane[kk];
        }

        score_buf[n] = dot0 * ATTN_SCALE;
    }
}

static void softmax_scores_head(
    float score_buf[SEQ_LEN],
    const int active_count,
    float prob_buf[SEQ_LEN])
{
#pragma HLS INLINE

    // Break the online-softmax recurrence into banked reductions so the
    // pipeline is no longer constrained by a long exp/max/sum feedback path.
    float max_banks[SOFTMAX_REDUCE_GROUPS];
    float sum_banks[SOFTMAX_REDUCE_GROUPS];
#pragma HLS ARRAY_PARTITION variable=max_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum_banks complete dim=1

INIT_MAX_BANKS:
    for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        max_banks[g] = -FLT_MAX;
    }

SOFTMAX_MAX_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const float score = score_buf[n];
        const int bank = n & (SOFTMAX_REDUCE_GROUPS - 1);
        max_banks[bank] = hls::fmaxf(max_banks[bank], score);
    }

    float max_score = max_banks[0];

REDUCE_MAX_BANKS:
    for (int g = 1; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        max_score = hls::fmaxf(max_score, max_banks[g]);
    }

INIT_SUM_BANKS:
    for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        sum_banks[g] = 0.0f;
    }

SOFTMAX_EXP_SUM_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const float exp_val = hls::expf(score_buf[n] - max_score);
        const int bank = n & (SOFTMAX_REDUCE_GROUPS - 1);
        prob_buf[n] = exp_val;
        sum_banks[bank] += exp_val;
    }

    float sum = 0.0f;

REDUCE_SUM_BANKS:
    for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        sum += sum_banks[g];
    }

    const float inv_sum = (sum == 0.0f) ? 0.0f : (1.0f / sum);

SOFTMAX_NORM_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        prob_buf[n] *= inv_sum;
    }
}

static void apply_scores_to_values_head(
    float prob_buf[SEQ_LEN],
    const int active_indices[SEQ_LEN],
    const int active_count,
    float v_group[SEQ_LEN][GROUP_DIM],
    const int h,
    const int q,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2

    const int head_base = h * HEAD_DIM;
    static_assert((HEAD_DIM % ATTN_ARRAY_FACTOR) == 0, "HEAD_DIM must be divisible by ATTN_ARRAY_FACTOR");

    float acc[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

INIT_CONTEXT_ACC:
    for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
        acc[d] = 0.0f;
    }

CONTEXT_K_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const int k = active_indices[n];
        const float prob = prob_buf[n];

    CONTEXT_D_LOOP:
        for (int d = 0; d < HEAD_DIM; d += ATTN_ARRAY_FACTOR) {
        CONTEXT_D_LANE:
            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                acc[d + kk] += prob * v_group[k][head_base + d + kk];
            }
        }
    }

WRITE_CONTEXT_ACC:
    for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
        context_group[q][head_base + d] = acc[d];
    }
}

static void compute_scores_head_pair(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    const int active_indices[SEQ_LEN],
    const int active_count,
    const int h,
    const int q_base,
    float score_buf0[SEQ_LEN],
    float score_buf1[SEQ_LEN])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2

    const int head_base = h * HEAD_DIM;
    const int q0 = q_base;
    const int q1 = q_base + 1;
    static_assert((HEAD_DIM % ATTN_ARRAY_FACTOR) == 0, "HEAD_DIM must be divisible by ATTN_ARRAY_FACTOR");

SCORE_PAIR_K_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const int k = active_indices[n];
        float dot0_lane[ATTN_ARRAY_FACTOR];
        float dot1_lane[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=dot0_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=dot1_lane complete dim=1

    INIT_SCORE_PAIR_LANES:
        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
            dot0_lane[kk] = 0.0f;
            dot1_lane[kk] = 0.0f;
        }

    SCORE_PAIR_D_LOOP:
        for (int d = 0; d < HEAD_DIM; d += ATTN_ARRAY_FACTOR) {
        SCORE_PAIR_D_LANE:
            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                const float kval = k_group[k][head_base + d + kk];
                dot0_lane[kk] += q_group[q0][head_base + d + kk] * kval;
                dot1_lane[kk] += q_group[q1][head_base + d + kk] * kval;
            }
        }

        float dot0 = 0.0f;
        float dot1 = 0.0f;
    REDUCE_SCORE_PAIR_LANES:
#ifdef BERT_BALANCED_FP_TREE
        dot0 = fp32_attn_sum_tree(dot0_lane);
        dot1 = fp32_attn_sum_tree(dot1_lane);
#else
        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
            dot0 += dot0_lane[kk];
            dot1 += dot1_lane[kk];
        }
#endif

        score_buf0[n] = dot0 * ATTN_SCALE;
        score_buf1[n] = dot1 * ATTN_SCALE;
    }
}

static void softmax_scores_pair(
    float score_buf0[SEQ_LEN],
    float score_buf1[SEQ_LEN],
    const int active_count,
    float prob_buf0[SEQ_LEN],
    float prob_buf1[SEQ_LEN])
{
#pragma HLS INLINE

    float max0_banks[SOFTMAX_REDUCE_GROUPS];
    float max1_banks[SOFTMAX_REDUCE_GROUPS];
    float sum0_banks[SOFTMAX_REDUCE_GROUPS];
    float sum1_banks[SOFTMAX_REDUCE_GROUPS];
#pragma HLS ARRAY_PARTITION variable=max0_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=max1_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum0_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum1_banks complete dim=1

INIT_PAIR_MAX_BANKS:
    for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        max0_banks[g] = -FLT_MAX;
        max1_banks[g] = -FLT_MAX;
    }

SOFTMAX_PAIR_MAX_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const int bank = n & (SOFTMAX_REDUCE_GROUPS - 1);
        max0_banks[bank] = hls::fmaxf(max0_banks[bank], score_buf0[n]);
        max1_banks[bank] = hls::fmaxf(max1_banks[bank], score_buf1[n]);
    }

    float max0 = max0_banks[0];
    float max1 = max1_banks[0];

REDUCE_PAIR_MAX_BANKS:
    for (int g = 1; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        max0 = hls::fmaxf(max0, max0_banks[g]);
        max1 = hls::fmaxf(max1, max1_banks[g]);
    }

INIT_PAIR_SUM_BANKS:
    for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        sum0_banks[g] = 0.0f;
        sum1_banks[g] = 0.0f;
    }

SOFTMAX_PAIR_EXP_SUM_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const float exp0 = hls::expf(score_buf0[n] - max0);
        const float exp1 = hls::expf(score_buf1[n] - max1);
        const int bank = n & (SOFTMAX_REDUCE_GROUPS - 1);
        prob_buf0[n] = exp0;
        prob_buf1[n] = exp1;
        sum0_banks[bank] += exp0;
        sum1_banks[bank] += exp1;
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;

REDUCE_PAIR_SUM_BANKS:
    for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
        sum0 += sum0_banks[g];
        sum1 += sum1_banks[g];
    }

    const float inv0 = (sum0 == 0.0f) ? 0.0f : (1.0f / sum0);
    const float inv1 = (sum1 == 0.0f) ? 0.0f : (1.0f / sum1);

SOFTMAX_PAIR_NORM_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        prob_buf0[n] *= inv0;
        prob_buf1[n] *= inv1;
    }
}

static void apply_scores_to_values_head_pair(
    float prob_buf0[SEQ_LEN],
    float prob_buf1[SEQ_LEN],
    const int active_indices[SEQ_LEN],
    const int active_count,
    float v_group[SEQ_LEN][GROUP_DIM],
    const int h,
    const int q_base,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2

    const int head_base = h * HEAD_DIM;
    const int q0 = q_base;
    const int q1 = q_base + 1;
    static_assert((HEAD_DIM % ATTN_ARRAY_FACTOR) == 0, "HEAD_DIM must be divisible by ATTN_ARRAY_FACTOR");

    float acc0[HEAD_DIM];
    float acc1[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1

INIT_CONTEXT_PAIR_ACC:
    for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
        acc0[d] = 0.0f;
        acc1[d] = 0.0f;
    }

CONTEXT_PAIR_K_LOOP:
    for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
        const int k = active_indices[n];
        const float prob0 = prob_buf0[n];
        const float prob1 = prob_buf1[n];

    CONTEXT_PAIR_D_LOOP:
        for (int d = 0; d < HEAD_DIM; d += ATTN_ARRAY_FACTOR) {
        CONTEXT_PAIR_D_LANE:
            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                const float v = v_group[k][head_base + d + kk];
                acc0[d + kk] += prob0 * v;
                acc1[d + kk] += prob1 * v;
            }
        }
    }

WRITE_CONTEXT_PAIR_STRIPE:
    for (int d_base = 0; d_base < HEAD_DIM; d_base += ATTN_HEAD_FACTOR) {
#pragma HLS PIPELINE II=1
    WRITE_CONTEXT_PAIR_BANK:
        for (int lane = 0; lane < ATTN_HEAD_FACTOR; ++lane) {
#pragma HLS UNROLL
            const int d = d_base + lane;
            context_group[q0][head_base + d] = acc0[d];
            context_group[q1][head_base + d] = acc1[d];
        }
    }
}

#ifdef BERT_STREAMING_DATAFLOW
// Query pipeline: score(q+1), softmax(q), and context(q-1) execute at the same
// time.  The arithmetic lane counts are unchanged; only three already distinct
// stages are allowed to run concurrently.
static void score_pair_stage(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    const int active_indices[SEQ_LEN],
    const int active_count,
    const int h,
    hls::stream<attention_pair_pkt_t> &score_stream)
{
#pragma HLS INLINE off
SCORE_STAGE_QUERY:
    for (int q = 0; q < SEQ_LEN; q += ATTN_QUERY_PAR) {
        const int head_base = h * HEAD_DIM;
    SCORE_STAGE_KEY:
        for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
            const int k = active_indices[n];
            float dot0_lane[ATTN_ARRAY_FACTOR];
            float dot1_lane[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=dot0_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=dot1_lane complete dim=1
        SCORE_STAGE_INIT_LANE:
            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                dot0_lane[kk] = 0.0f;
                dot1_lane[kk] = 0.0f;
            }
        SCORE_STAGE_DIM:
            for (int d = 0; d < HEAD_DIM; d += ATTN_ARRAY_FACTOR) {
            SCORE_STAGE_LANE:
                for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                    const int col = head_base + d + kk;
                    const float kval = k_group[k][col];
                    dot0_lane[kk] += q_group[q][col] * kval;
                    dot1_lane[kk] += q_group[q + 1][col] * kval;
                }
            }
            attention_pair_pkt_t pkt;
#ifdef BERT_BALANCED_FP_TREE
            pkt.q0 = fp32_attn_sum_tree(dot0_lane) * ATTN_SCALE;
            pkt.q1 = fp32_attn_sum_tree(dot1_lane) * ATTN_SCALE;
#else
            float dot0 = 0.0f;
            float dot1 = 0.0f;
        SCORE_STAGE_REDUCE_LANE:
            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                dot0 += dot0_lane[kk];
                dot1 += dot1_lane[kk];
            }
            pkt.q0 = dot0 * ATTN_SCALE;
            pkt.q1 = dot1 * ATTN_SCALE;
#endif
            score_stream.write(pkt);
        }
    }
}

static void softmax_pair_stage(
    const int active_count,
    hls::stream<attention_pair_pkt_t> &score_stream,
    hls::stream<attention_pair_pkt_t> &prob_stream)
{
#pragma HLS INLINE off
SOFTMAX_STAGE_QUERY:
    for (int q = 0; q < SEQ_LEN; q += ATTN_QUERY_PAR) {
        float score0[SEQ_LEN];
        float score1[SEQ_LEN];
        float prob0[SEQ_LEN];
        float prob1[SEQ_LEN];
        float max0_bank[SOFTMAX_REDUCE_GROUPS];
        float max1_bank[SOFTMAX_REDUCE_GROUPS];
        float sum0_bank[SOFTMAX_REDUCE_GROUPS];
        float sum1_bank[SOFTMAX_REDUCE_GROUPS];
#pragma HLS BIND_STORAGE variable=score0 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=score1 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=prob0 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=prob1 type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=max0_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=max1_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum0_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum1_bank complete dim=1

    SOFTMAX_STAGE_INIT_MAX:
        for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
            max0_bank[g] = -FLT_MAX;
            max1_bank[g] = -FLT_MAX;
        }

    SOFTMAX_STAGE_READ:
        for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
            attention_pair_pkt_t pkt = score_stream.read();
            score0[n] = pkt.q0;
            score1[n] = pkt.q1;
            const int bank = n & (SOFTMAX_REDUCE_GROUPS - 1);
            max0_bank[bank] = hls::fmaxf(max0_bank[bank], pkt.q0);
            max1_bank[bank] = hls::fmaxf(max1_bank[bank], pkt.q1);
        }

        float max0 = max0_bank[0];
        float max1 = max1_bank[0];
    SOFTMAX_STAGE_REDUCE_MAX:
        for (int g = 1; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
            max0 = hls::fmaxf(max0, max0_bank[g]);
            max1 = hls::fmaxf(max1, max1_bank[g]);
        }

    SOFTMAX_STAGE_INIT_SUM:
        for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
            sum0_bank[g] = 0.0f;
            sum1_bank[g] = 0.0f;
        }

    SOFTMAX_STAGE_EXP:
        for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
            const float exp0 = hls::expf(score0[n] - max0);
            const float exp1 = hls::expf(score1[n] - max1);
            prob0[n] = exp0;
            prob1[n] = exp1;
        }

    // Separating the exponential pipeline from the banked recurrence prevents
    // the exp latency from forcing the sum loop to II=3.  No extra exp lane or
    // reduction lane is introduced.
    SOFTMAX_STAGE_SUM:
        for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
            const int bank = n & (SOFTMAX_REDUCE_GROUPS - 1);
            sum0_bank[bank] += prob0[n];
            sum1_bank[bank] += prob1[n];
        }

        float sum0 = 0.0f;
        float sum1 = 0.0f;
    SOFTMAX_STAGE_REDUCE_SUM:
        for (int g = 0; g < SOFTMAX_REDUCE_GROUPS; ++g) {
#pragma HLS UNROLL
            sum0 += sum0_bank[g];
            sum1 += sum1_bank[g];
        }
        const float inv0 = (sum0 == 0.0f) ? 0.0f : (1.0f / sum0);
        const float inv1 = (sum1 == 0.0f) ? 0.0f : (1.0f / sum1);

    SOFTMAX_STAGE_WRITE:
        for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
            attention_pair_pkt_t pkt;
            pkt.q0 = prob0[n] * inv0;
            pkt.q1 = prob1[n] * inv1;
            prob_stream.write(pkt);
        }
    }
}

static void context_pair_stage(
    const int active_indices[SEQ_LEN],
    const int active_count,
    float v_group[SEQ_LEN][GROUP_DIM],
    const int h,
    hls::stream<attention_pair_pkt_t> &prob_stream,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
CONTEXT_STAGE_QUERY:
    for (int q = 0; q < SEQ_LEN; q += ATTN_QUERY_PAR) {
        float acc0[HEAD_DIM];
        float acc1[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1
        const int head_base = h * HEAD_DIM;

    CONTEXT_STAGE_INIT:
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
            acc0[d] = 0.0f;
            acc1[d] = 0.0f;
        }

    CONTEXT_STAGE_ACC:
        for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
            attention_pair_pkt_t pkt = prob_stream.read();
            const int k = active_indices[n];
        CONTEXT_STAGE_DIM:
            for (int d = 0; d < HEAD_DIM; d += ATTN_ARRAY_FACTOR) {
            CONTEXT_STAGE_LANE:
                for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                    const int dim = d + kk;
                    const float value = v_group[k][head_base + dim];
                    acc0[dim] += pkt.q0 * value;
                    acc1[dim] += pkt.q1 * value;
                }
            }
        }

    CONTEXT_STAGE_WRITE_STRIPE:
        for (int d_base = 0; d_base < HEAD_DIM; d_base += ATTN_HEAD_FACTOR) {
#pragma HLS PIPELINE II=1
        CONTEXT_STAGE_WRITE_BANK:
            for (int lane = 0; lane < ATTN_HEAD_FACTOR; ++lane) {
#pragma HLS UNROLL
                const int d = d_base + lane;
                context_group[q][head_base + d] = acc0[d];
                context_group[q + 1][head_base + d] = acc1[d];
            }
        }
    }
}

static void attention_head_query_pipeline(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM],
    const int active_indices[SEQ_LEN],
    const int active_count,
    const int h,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
    hls::stream<attention_pair_pkt_t> score_stream("score_pair_stream");
    hls::stream<attention_pair_pkt_t> prob_stream("prob_pair_stream");
#pragma HLS STREAM variable=score_stream depth=256
#pragma HLS STREAM variable=prob_stream depth=256

#pragma HLS DATAFLOW
    score_pair_stage(
        q_group, k_group, active_indices, active_count, h, score_stream);
    softmax_pair_stage(active_count, score_stream, prob_stream);
    context_pair_stage(
        active_indices, active_count, v_group, h, prob_stream, context_group);
}
#endif

static void attention_split_group(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM],
    const int active_indices[SEQ_LEN],
    const int active_count,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2

#ifdef BERT_STREAMING_DATAFLOW
ATTN_HEAD_DATAFLOW_LOOP:
    for (int h = 0; h < HEAD_PAR; ++h) {
#pragma HLS UNROLL
        attention_head_query_pipeline(
            q_group, k_group, v_group,
            active_indices, active_count, h, context_group);
    }
#else
ATTN_HEAD_LOOP:
    for (int h = 0; h < HEAD_PAR; ++h) {
#pragma HLS UNROLL
    ATTN_QUERY_LOOP:
        for (int q = 0; q < SEQ_LEN; q += ATTN_QUERY_PAR) {
            float score_buf0[SEQ_LEN];
            float score_buf1[SEQ_LEN];
            float prob_buf0[SEQ_LEN];
            float prob_buf1[SEQ_LEN];
#pragma HLS BIND_STORAGE variable=score_buf0 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=score_buf1 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=prob_buf0 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=prob_buf1 type=RAM_2P impl=BRAM

            compute_scores_head_pair(
                q_group,
                k_group,
                active_indices,
                active_count,
                h,
                q,
                score_buf0,
                score_buf1);

            softmax_scores_pair(score_buf0, score_buf1, active_count, prob_buf0, prob_buf1);

            apply_scores_to_values_head_pair(
                prob_buf0,
                prob_buf1,
                active_indices,
                active_count,
                v_group,
                h,
                q,
                context_group);
        }
    }
#endif
}

static const int GROUP_PACKS = GROUP_DIM / PACK_SIZE;

static void write_qkv_group_streams(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM],
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream)
{
#pragma HLS INLINE off
WRITE_QKV_STREAM_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    WRITE_QKV_STREAM_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float q_lane[PACK_SIZE];
            float k_lane[PACK_SIZE];
            float v_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=q_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1
        WRITE_QKV_STREAM_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int col = p * PACK_SIZE + i;
                q_lane[i] = q_group[s][col];
                k_lane[i] = k_group[s][col];
                v_lane[i] = v_group[s][col];
            }
            q_stream.write(pack_bus16(q_lane));
            k_stream.write(pack_bus16(k_lane));
            v_stream.write(pack_bus16(v_lane));
        }
    }
}

static void read_qkv_group_streams(
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream,
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
READ_QKV_STREAM_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    READ_QKV_STREAM_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float q_lane[PACK_SIZE];
            float k_lane[PACK_SIZE];
            float v_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=q_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1
            unpack_bus16(q_stream.read(), q_lane);
            unpack_bus16(k_stream.read(), k_lane);
            unpack_bus16(v_stream.read(), v_lane);
        READ_QKV_STREAM_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int col = p * PACK_SIZE + i;
                q_group[s][col] = q_lane[i];
                k_group[s][col] = k_lane[i];
                v_group[s][col] = v_lane[i];
            }
        }
    }
}

// The original dataflow version used three local FIFOs.  The split-kernel
// implementation transports the same packets on one AXI stream in the order
// Q, K, V.  Keeping the packet width and pack order unchanged makes this a
// transport-only refactor: project_qkv_head_group and attention_split_group
// retain their original arithmetic.
static void write_qkv_group_interleaved_stream(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM],
    hidden_stream_t &qkv_stream)
{
#pragma HLS INLINE off
WRITE_QKV_INTERLEAVED_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    WRITE_QKV_INTERLEAVED_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float q_lane[PACK_SIZE];
            float k_lane[PACK_SIZE];
            float v_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=q_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1
        WRITE_QKV_INTERLEAVED_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int col = p * PACK_SIZE + i;
                q_lane[i] = q_group[s][col];
                k_lane[i] = k_group[s][col];
                v_lane[i] = v_group[s][col];
            }
            qkv_stream.write(pack_bus16(q_lane));
            qkv_stream.write(pack_bus16(k_lane));
            qkv_stream.write(pack_bus16(v_lane));
        }
    }
}

static void read_qkv_group_interleaved_stream(
    hidden_stream_t &qkv_stream,
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
READ_QKV_INTERLEAVED_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    READ_QKV_INTERLEAVED_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float q_lane[PACK_SIZE];
            float k_lane[PACK_SIZE];
            float v_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=q_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=v_lane complete dim=1
            unpack_bus16(qkv_stream.read(), q_lane);
            unpack_bus16(qkv_stream.read(), k_lane);
            unpack_bus16(qkv_stream.read(), v_lane);
        READ_QKV_INTERLEAVED_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int col = p * PACK_SIZE + i;
                q_group[s][col] = q_lane[i];
                k_group[s][col] = k_lane[i];
                v_group[s][col] = v_lane[i];
            }
        }
    }
}

static void write_context_group_stream(
    float context_group[SEQ_LEN][GROUP_DIM],
    hidden_stream_t &context_stream)
{
#pragma HLS INLINE off
WRITE_CONTEXT_STREAM_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    WRITE_CONTEXT_STREAM_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
        WRITE_CONTEXT_STREAM_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                lane[i] = context_group[s][p * PACK_SIZE + i];
            }
            context_stream.write(pack_bus16(lane));
        }
    }
}

static void read_context_group_stream(
    hidden_stream_t &context_stream,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
READ_CONTEXT_STREAM_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    READ_CONTEXT_STREAM_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            unpack_bus16(context_stream.read(), lane);
        READ_CONTEXT_STREAM_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                context_group[s][p * PACK_SIZE + i] = lane[i];
            }
        }
    }
}

static void project_all_head_groups_stage(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    const bus_t *q_w,
    const bus_t *q_b,
    const bus_t *k_w,
    const bus_t *k_b,
    const bus_t *v_w,
    const bus_t *v_b,
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream)
{
#pragma HLS INLINE off
    float q_group[SEQ_LEN][GROUP_DIM];
    float k_group[SEQ_LEN][GROUP_DIM];
    float v_group[SEQ_LEN][GROUP_DIM];
#pragma HLS BIND_STORAGE variable=q_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_group type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2

PROJECT_STAGE_GROUP:
    for (int hg = 0; hg < HEAD_GROUPS; ++hg) {
        project_qkv_head_group(
            input_hidden, q_w, q_b, k_w, k_b, v_w, v_b,
            hg, q_group, k_group, v_group);
        write_qkv_group_streams(
            q_group, k_group, v_group, q_stream, k_stream, v_stream);
    }
}

static void attend_all_head_groups_stage(
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream,
    const int active_indices[SEQ_LEN],
    const int active_count,
    hidden_stream_t &context_stream)
{
#pragma HLS INLINE off
    float q_group[SEQ_LEN][GROUP_DIM];
    float k_group[SEQ_LEN][GROUP_DIM];
    float v_group[SEQ_LEN][GROUP_DIM];
    float context_group[SEQ_LEN][GROUP_DIM];
#pragma HLS BIND_STORAGE variable=q_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=context_group type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2

ATTEND_STAGE_GROUP:
    for (int hg = 0; hg < HEAD_GROUPS; ++hg) {
        read_qkv_group_streams(
            q_stream, k_stream, v_stream, q_group, k_group, v_group);
        attention_split_group(
            q_group, k_group, v_group,
            active_indices, active_count, context_group);
        write_context_group_stream(context_group, context_stream);
    }
}

static void project_output_dense_group_accum(
    float context_group[SEQ_LEN][GROUP_DIM],
    const bus_t *dense_w,
    const int head_group,
    float output_hidden[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_RESHAPE variable=output_hidden cyclic factor=ATTN_STORAGE_FACTOR dim=2
    const int group_base = head_group * GROUP_DIM;
    const int num_ib_tiles = GROUP_DIM / ATTN_PROJ_TILE_K;
    float wo_tile[ATTN_OUT_TILE][ATTN_PROJ_TILE_K];
    float out_acc[SEQ_LEN][ATTN_OUT_TILE];
#ifdef BERT_SCALED_VARIANT
#pragma HLS BIND_STORAGE variable=wo_tile type=RAM_2P impl=LUTRAM
#else
#pragma HLS BIND_STORAGE variable=wo_tile type=RAM_2P impl=BRAM
#endif
#pragma HLS BIND_STORAGE variable=out_acc type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=wo_tile cyclic factor=ATTN_OUT_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wo_tile cyclic factor=ATTN_ARRAY_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=ATTN_OUT_PAR dim=2

GROUP_OUT_BLOCK:
    for (int ob = 0; ob < HIDDEN_SIZE; ob += ATTN_OUT_TILE) {
    GROUP_OUT_INIT_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        GROUP_OUT_INIT_COL:
            for (int oo = 0; oo < ATTN_OUT_TILE; ++oo) {
#pragma HLS PIPELINE II=1
                out_acc[s][oo] = output_hidden[s][ob + oo];
            }
        }

    GROUP_OUT_INPUT_TILE:
        for (int ib_idx = 0; ib_idx < num_ib_tiles; ++ib_idx) {
            const int ib = ib_idx * ATTN_PROJ_TILE_K;
            load_output_weight_tile(
                dense_w, ob, group_base, ib, wo_tile);

        GROUP_OUT_ROW:
            for (int s = 0; s < SEQ_LEN; ++s) {
            GROUP_OUT_COL:
                for (int oo = 0; oo < ATTN_OUT_TILE; oo += ATTN_OUT_PAR) {
#pragma HLS PIPELINE II=1
                    float acc_vec[ATTN_OUT_PAR];
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
                GROUP_OUT_LOAD_ACC:
                    for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                        acc_vec[jj] = out_acc[s][oo + jj];
                    }
                GROUP_OUT_K:
                    for (int ii = 0; ii < ATTN_PROJ_TILE_K; ii += ATTN_ARRAY_FACTOR) {
                        float x_vec[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
                    GROUP_OUT_LOAD_X:
                        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                            x_vec[kk] = context_group[s][ib + ii + kk];
                        }
                    GROUP_OUT_MAC:
                        for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
#ifdef BERT_BALANCED_FP_TREE
                            float w_vec[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        GROUP_OUT_LOAD_W_VEC:
                            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                                w_vec[kk] = wo_tile[oo + jj][ii + kk];
                            }
                            acc_vec[jj] += fp32_attn_dot_tree(x_vec, w_vec);
#else
                            float sum = acc_vec[jj];
                        GROUP_OUT_MAC_K:
                            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                                sum += x_vec[kk] * wo_tile[oo + jj][ii + kk];
                            }
                            acc_vec[jj] = sum;
#endif
                        }
                    }
                GROUP_OUT_STORE_ACC:
                    for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                        out_acc[s][oo + jj] = acc_vec[jj];
                    }
                }
            }
        }

    GROUP_OUT_WRITE_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        GROUP_OUT_WRITE_COL:
            for (int oo = 0; oo < ATTN_OUT_TILE; ++oo) {
#pragma HLS PIPELINE II=1
                output_hidden[s][ob + oo] = out_acc[s][oo];
            }
        }
    }
}

static void output_all_head_groups_stage(
    hidden_stream_t &context_stream,
    const bus_t *dense_w,
    const bus_t *dense_b,
    float output_hidden[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float context_group[SEQ_LEN][GROUP_DIM];
#pragma HLS BIND_STORAGE variable=context_group type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2

OUTPUT_STAGE_INIT_BLOCK:
    for (int ob = 0; ob < HIDDEN_SIZE; ob += ATTN_OUT_TILE) {
        float bias_tile[ATTN_OUT_TILE];
#pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
        load_output_bias_tile(dense_b, ob, bias_tile);
    OUTPUT_STAGE_INIT_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        OUTPUT_STAGE_INIT_COL:
            for (int oo = 0; oo < ATTN_OUT_TILE; ++oo) {
#pragma HLS PIPELINE II=1
                output_hidden[s][ob + oo] = bias_tile[oo];
            }
        }
    }

OUTPUT_STAGE_GROUP:
    for (int hg = 0; hg < HEAD_GROUPS; ++hg) {
        read_context_group_stream(context_stream, context_group);
        project_output_dense_group_accum(
            context_group, dense_w, hg, output_hidden);
    }
}
static void store_context_group_full(
    float context_group[SEQ_LEN][GROUP_DIM],
    const int head_group,
    float context_full[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_RESHAPE variable=context_full cyclic factor=ATTN_STORAGE_FACTOR dim=2

    const int group_base = head_group * GROUP_DIM;

STORE_CONTEXT_FULL_SEQ:
    for (int s = 0; s < SEQ_LEN; ++s) {
    STORE_CONTEXT_FULL_STRIPE:
        for (int d_base = 0; d_base < GROUP_DIM;
             d_base += ATTN_STORAGE_FACTOR) {
#pragma HLS PIPELINE II=1
        STORE_CONTEXT_FULL_BANK:
            for (int lane = 0; lane < ATTN_STORAGE_FACTOR; ++lane) {
#pragma HLS UNROLL
                const int d = d_base + lane;
                context_full[s][group_base + d] = context_group[s][d];
            }
        }
    }
}

static void project_output_dense_full(
    float context_full[SEQ_LEN][HIDDEN_SIZE],
    const bus_t *dense_w,
    const bus_t *dense_b,
    float output_hidden[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_RESHAPE variable=context_full cyclic factor=ATTN_STORAGE_FACTOR dim=2
#pragma HLS ARRAY_RESHAPE variable=output_hidden cyclic factor=ATTN_STORAGE_FACTOR dim=2

    const int NUM_IB_TILES = HIDDEN_SIZE / ATTN_PROJ_TILE_K;

    float wo_tile[ATTN_OUT_TILE][ATTN_PROJ_TILE_K];
    float out_acc[SEQ_LEN][ATTN_OUT_TILE];
    float bias_tile[ATTN_OUT_TILE];
#ifdef BERT_SCALED_VARIANT
#pragma HLS BIND_STORAGE variable=wo_tile type=RAM_2P impl=LUTRAM
#else
#pragma HLS BIND_STORAGE variable=wo_tile type=RAM_2P impl=BRAM
#endif
#pragma HLS BIND_STORAGE variable=out_acc type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=wo_tile cyclic factor=ATTN_OUT_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wo_tile cyclic factor=ATTN_ARRAY_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=ATTN_OUT_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1

    static_assert((ATTN_OUT_TILE % ATTN_OUT_PAR) == 0,
                  "ATTN_OUT_TILE must be divisible by ATTN_OUT_PAR");
    static_assert((ATTN_PROJ_TILE_K % ATTN_ARRAY_FACTOR) == 0,
                  "ATTN_PROJ_TILE_K must be divisible by ATTN_ARRAY_FACTOR");

FULL_OUT_BLOCK_LOOP:
    for (int ob = 0; ob < HIDDEN_SIZE; ob += ATTN_OUT_TILE) {
        load_output_bias_tile(dense_b, ob, bias_tile);

    FULL_INIT_OUT_ACC_SEQ:
        for (int s = 0; s < SEQ_LEN; ++s) {
        FULL_INIT_OUT_ACC_O:
            for (int oo = 0; oo < ATTN_OUT_TILE; ++oo) {
#pragma HLS PIPELINE II=1
                out_acc[s][oo] = bias_tile[oo];
            }
        }

    FULL_OUT_IB_TILE_LOOP:
        for (int ib_idx = 0; ib_idx < NUM_IB_TILES; ++ib_idx) {
            const int ib = ib_idx * ATTN_PROJ_TILE_K;

            load_output_weight_tile(
                dense_w,
                ob,
                0,
                ib,
                wo_tile);

        FULL_OUT_SEQ_LOOP:
            for (int s = 0; s < SEQ_LEN; ++s) {
            FULL_OUT_O_LOOP:
                for (int oo = 0; oo < ATTN_OUT_TILE; oo += ATTN_OUT_PAR) {
#pragma HLS PIPELINE II=1
                    float acc_vec[ATTN_OUT_PAR];
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1

                LOAD_FULL_OUT_ACC:
                    for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                        acc_vec[jj] = out_acc[s][oo + jj];
                    }

                FULL_OUT_K_LOOP:
                    for (int ii = 0; ii < ATTN_PROJ_TILE_K; ii += ATTN_ARRAY_FACTOR) {
                        float x_vec[ATTN_ARRAY_FACTOR];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1

                    LOAD_FULL_OUT_X:
                        for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                            x_vec[kk] = context_full[s][ib + ii + kk];
                        }

                FULL_OUT_VEC:
                        for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                            float sum = acc_vec[jj];
                        FULL_OUT_K_VEC:
                            for (int kk = 0; kk < ATTN_ARRAY_FACTOR; ++kk) {
#pragma HLS UNROLL
                                sum += x_vec[kk] * wo_tile[oo + jj][ii + kk];
                            }
                            acc_vec[jj] = sum;
                        }
                    }

                STORE_FULL_OUT_ACC:
                    for (int jj = 0; jj < ATTN_OUT_PAR; ++jj) {
#pragma HLS UNROLL
                        out_acc[s][oo + jj] = acc_vec[jj];
                    }
                }
            }
        }

    FULL_WRITE_OUT_ACC_SEQ:
        for (int s = 0; s < SEQ_LEN; ++s) {
        FULL_WRITE_OUT_ACC_O:
            for (int oo = 0; oo < ATTN_OUT_TILE; ++oo) {
#pragma HLS PIPELINE II=1
                output_hidden[s][ob + oo] = out_acc[s][oo];
            }
        }
    }
}

// ============================================================================
// Top attention kernel
// ============================================================================

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
    hidden_stream_t &output_hidden)
{
#pragma HLS INLINE off

    static float input_hidden_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_hidden_buf[SEQ_LEN][HIDDEN_SIZE];
#ifndef BERT_STREAMING_DATAFLOW
    // These matrices belong only to the legacy full-materialization path.
    // Excluding the declarations in the streaming profile makes the intended
    // URAM lifetime reduction explicit instead of relying on dead-code cleanup.
    static float q_group_buf[SEQ_LEN][GROUP_DIM];
    static float k_group_buf[SEQ_LEN][GROUP_DIM];
    static float v_group_buf[SEQ_LEN][GROUP_DIM];
    static float context_group_buf[SEQ_LEN][GROUP_DIM];
    static float context_full_buf[SEQ_LEN][HIDDEN_SIZE];
#endif
    int attention_mask_buf[SEQ_LEN];
    int active_indices[SEQ_LEN];

#pragma HLS BIND_STORAGE variable=input_hidden_buf   type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=output_hidden_buf  type=RAM_2P impl=URAM
#ifndef BERT_STREAMING_DATAFLOW
#pragma HLS BIND_STORAGE variable=q_group_buf        type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k_group_buf        type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_group_buf        type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=context_group_buf  type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=context_full_buf   type=RAM_2P impl=URAM
#endif
#pragma HLS BIND_STORAGE variable=attention_mask_buf type=RAM_S2P impl=LUTRAM
#pragma HLS BIND_STORAGE variable=active_indices     type=RAM_S2P impl=LUTRAM
#pragma HLS ARRAY_RESHAPE variable=input_hidden_buf cyclic factor=ATTN_STORAGE_FACTOR dim=2
#pragma HLS ARRAY_RESHAPE variable=output_hidden_buf cyclic factor=ATTN_STORAGE_FACTOR dim=2
#ifndef BERT_STREAMING_DATAFLOW
#pragma HLS ARRAY_RESHAPE variable=context_full_buf cyclic factor=ATTN_STORAGE_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=q_group_buf cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group_buf cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group_buf cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=context_group_buf cyclic factor=ATTN_HEAD_FACTOR dim=2
#endif

    load_hidden_stream_to_matrix(input_hidden, input_hidden_buf);

LOAD_MASK:
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1
        attention_mask_buf[i] = attention_mask[i];
    }

    const int active_count = build_active_index_list(attention_mask_buf, active_indices);

#ifdef BERT_STREAMING_DATAFLOW
    hidden_stream_t q_group_stream("q_group_stream");
    hidden_stream_t k_group_stream("k_group_stream");
    hidden_stream_t v_group_stream("v_group_stream");
    hidden_stream_t context_group_stream("context_group_stream");
#pragma HLS STREAM variable=q_group_stream depth=64
#pragma HLS STREAM variable=k_group_stream depth=64
#pragma HLS STREAM variable=v_group_stream depth=64
#pragma HLS STREAM variable=context_group_stream depth=64

#pragma HLS DATAFLOW
    project_all_head_groups_stage(
        input_hidden_buf,
        q_w, q_b, k_w, k_b, v_w, v_b,
        q_group_stream, k_group_stream, v_group_stream);
    attend_all_head_groups_stage(
        q_group_stream, k_group_stream, v_group_stream,
        active_indices, active_count, context_group_stream);
    output_all_head_groups_stage(
        context_group_stream, dense_w, dense_b, output_hidden_buf);
#else
HEAD_GROUP_LOOP:
    for (int hg = 0; hg < HEAD_GROUPS; ++hg) {
        project_qkv_head_group(
            input_hidden_buf,
            q_w, q_b,
            k_w, k_b,
            v_w, v_b,
            hg,
            q_group_buf,
            k_group_buf,
            v_group_buf);

        attention_split_group(
            q_group_buf,
            k_group_buf,
            v_group_buf,
            active_indices,
            active_count,
            context_group_buf);

        store_context_group_full(
            context_group_buf,
            hg,
            context_full_buf);
    }

    project_output_dense_full(
        context_full_buf,
        dense_w,
        dense_b,
        output_hidden_buf);
#endif

    matrix_add_norm_to_stream(
        input_hidden_buf,
        output_hidden_buf,
        norm_gamma,
        norm_beta,
        output_hidden);
}

// ============================================================================
// External split-kernel stages
// ============================================================================

static void load_hidden_ddr_to_matrix(
    const bus_t *input_hidden,
    float hidden[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
LOAD_HIDDEN_DDR_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    LOAD_HIDDEN_DDR_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            unpack_bus16(input_hidden[s * PACKS + p], lane);
        LOAD_HIDDEN_DDR_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                hidden[s][p * PACK_SIZE + i] = lane[i];
            }
        }
    }
}

void bert_qkv_stage(
    hidden_stream_t &input_hidden,
    const bus_t     *q_w,
    const bus_t     *q_b,
    const bus_t     *k_w,
    const bus_t     *k_b,
    const bus_t     *v_w,
    const bus_t     *v_b,
    hidden_stream_t &qkv_stream)
{
#pragma HLS INLINE off
    static float input_hidden_buf[SEQ_LEN][HIDDEN_SIZE];
    static float q_group[SEQ_LEN][GROUP_DIM];
    static float k_group[SEQ_LEN][GROUP_DIM];
    static float v_group[SEQ_LEN][GROUP_DIM];
#pragma HLS BIND_STORAGE variable=input_hidden_buf type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=q_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_group type=RAM_2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=input_hidden_buf cyclic factor=ATTN_STORAGE_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2

    load_hidden_stream_to_matrix(input_hidden, input_hidden_buf);

QKV_STAGE_HEAD_GROUP:
    for (int hg = 0; hg < HEAD_GROUPS; ++hg) {
        project_qkv_head_group(
            input_hidden_buf, q_w, q_b, k_w, k_b, v_w, v_b,
            hg, q_group, k_group, v_group);
        write_qkv_group_interleaved_stream(
            q_group, k_group, v_group, qkv_stream);
    }
}

void bert_attn_core_stage(
    hidden_stream_t &qkv_stream,
    const int       *attention_mask,
    hidden_stream_t &context_stream)
{
#pragma HLS INLINE off
    static float q_group[SEQ_LEN][GROUP_DIM];
    static float k_group[SEQ_LEN][GROUP_DIM];
    static float v_group[SEQ_LEN][GROUP_DIM];
    static float context_group[SEQ_LEN][GROUP_DIM];
    int attention_mask_buf[SEQ_LEN];
    int active_indices[SEQ_LEN];
#pragma HLS BIND_STORAGE variable=q_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=context_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=attention_mask_buf type=RAM_S2P impl=LUTRAM
#pragma HLS BIND_STORAGE variable=active_indices type=RAM_S2P impl=LUTRAM
#pragma HLS ARRAY_PARTITION variable=q_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=k_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=v_group cyclic factor=ATTN_HEAD_FACTOR dim=2
#pragma HLS ARRAY_PARTITION variable=context_group cyclic factor=ATTN_HEAD_FACTOR dim=2

ATTN_STAGE_LOAD_MASK:
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1
        attention_mask_buf[i] = attention_mask[i];
    }
    const int active_count = build_active_index_list(attention_mask_buf, active_indices);

ATTN_STAGE_HEAD_GROUP:
    for (int hg = 0; hg < HEAD_GROUPS; ++hg) {
        read_qkv_group_interleaved_stream(
            qkv_stream, q_group, k_group, v_group);
        attention_split_group(
            q_group, k_group, v_group,
            active_indices, active_count, context_group);
        write_context_group_stream(context_group, context_stream);
    }
}

void bert_attn_out_norm_stage(
    hidden_stream_t &context_stream,
    const bus_t     *residual_in,
    const bus_t     *dense_w,
    const bus_t     *dense_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &attn_mid_stream)
{
#pragma HLS INLINE off
    static float residual_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];
#pragma HLS BIND_STORAGE variable=residual_buf type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=residual_buf cyclic factor=ATTN_STORAGE_FACTOR dim=2
#pragma HLS ARRAY_RESHAPE variable=output_buf cyclic factor=ATTN_STORAGE_FACTOR dim=2

#pragma HLS DATAFLOW
    load_hidden_ddr_to_matrix(residual_in, residual_buf);
    output_all_head_groups_stage(context_stream, dense_w, dense_b, output_buf);
    matrix_add_norm_to_stream(
        residual_buf, output_buf, norm_gamma, norm_beta, attn_mid_stream);
}
