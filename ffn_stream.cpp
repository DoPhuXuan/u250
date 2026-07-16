#include "bert_model.h"

// ====================== MACROS & CONSTANTS ======================
#if !defined(BERT_EXACT_FP_ORDER) && !defined(BERT_BALANCED_FP_TREE)
#define BERT_BALANCED_FP_TREE
#endif

#ifdef BERT_ROUTE_FRIENDLY
static const int TILE_OUT_UP   = 16;
static const int TILE_OUT_DOWN = 16;
static const int TILE_K        = 64;
static const int MAC_GROUP     = 16;
static const int K_PAR         = 8;
#elif defined(BERT_FFN_K8) || defined(BERT_PREROUTE)
static const int TILE_OUT_UP   = 32;
static const int TILE_OUT_DOWN = 32;
static const int TILE_K        = 64;
static const int MAC_GROUP     = 32;
static const int K_PAR         = 8;
#else
static const int TILE_OUT_UP   = 32;
static const int TILE_OUT_DOWN = 32;
static const int TILE_K        = 64;
static const int MAC_GROUP     = 32;
static const int K_PAR         = 32;
#endif

static_assert((HIDDEN % TILE_K) == 0, "HIDDEN must be divisible by TILE_K");
static_assert((FFN_DIM % TILE_K) == 0, "FFN_DIM must be divisible by TILE_K");
static_assert((TILE_OUT_UP % PACK_SIZE) == 0, "TILE_OUT_UP must be pack-aligned");
static_assert((TILE_OUT_DOWN % PACK_SIZE) == 0, "TILE_OUT_DOWN must be pack-aligned");
static_assert((TILE_OUT_UP % MAC_GROUP) == 0, "TILE_OUT_UP must be divisible by MAC_GROUP");
static_assert((TILE_OUT_DOWN % MAC_GROUP) == 0, "TILE_OUT_DOWN must be divisible by MAC_GROUP");
static_assert((TILE_K % K_PAR) == 0, "TILE_K must be divisible by K_PAR");
#if (defined(BERT_ROUTE_FRIENDLY) || defined(BERT_FFN_K8) || defined(BERT_PREROUTE)) && defined(BERT_STREAMING_DATAFLOW)
#error "Reduced-route FFN profiles are not compatible with BERT_STREAMING_DATAFLOW"
#endif
#ifdef BERT_STREAMING_DATAFLOW
static_assert(TILE_OUT_UP == 32, "streaming dataflow locks FFN up tile");
static_assert(TILE_OUT_DOWN == 32, "streaming dataflow locks FFN down tile");
static_assert(TILE_K == 64, "streaming dataflow locks FFN K tile");
static_assert(MAC_GROUP == 32, "streaming dataflow locks FFN output lanes");
static_assert(K_PAR == 32, "streaming dataflow locks FFN K lanes");
#endif

// The low-latency profile uses the shorter balanced tree from bert_model.h.
// Define BERT_EXACT_FP_ORDER only when bit-for-bit reproduction of the old
// left-to-right FP32 accumulation order is more important than timing.
static inline float accumulate_dot16(
    const float accumulator,
    const float lhs[K_PAR],
    const float rhs[K_PAR])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
#ifdef BERT_BALANCED_FP_TREE
#if defined(BERT_ROUTE_FRIENDLY) || defined(BERT_FFN_K8) || defined(BERT_PREROUTE)
    return accumulator + fp32_dot8_tree(lhs, rhs);
#else
    return accumulator + fp32_dot32_tree(lhs, rhs);
#endif
#else
    float sum = accumulator;
ACCUMULATE_DOT16_ORDERED:
    for (int i = 0; i < K_PAR; ++i) {
#pragma HLS UNROLL
        sum += lhs[i] * rhs[i];
    }
    return sum;
#endif
}

// ====================== FAST GELU ======================
static float gelu(float x) {
#pragma HLS INLINE
    float x2 = x * x;
#pragma HLS BIND_OP variable=x2 op=fmul impl=maxdsp
    float x3 = x2 * x;
#pragma HLS BIND_OP variable=x3 op=fmul impl=maxdsp
    float t0 = 0.044715f * x3;
#pragma HLS BIND_OP variable=t0 op=fmul impl=maxdsp
    float t1 = x + t0;
#pragma HLS BIND_OP variable=t1 op=fadd impl=fulldsp
    float inner = 0.79788456f * t1;
#pragma HLS BIND_OP variable=inner op=fmul impl=maxdsp
    float tanh_v = hls::tanhf(inner);
    float one_plus = 1.0f + tanh_v;
#pragma HLS BIND_OP variable=one_plus op=fadd impl=fulldsp
    float half_x = 0.5f * x;
#pragma HLS BIND_OP variable=half_x op=fmul impl=maxdsp
    float out = half_x * one_plus;
#pragma HLS BIND_OP variable=out op=fmul impl=maxdsp
    return out;
}

static void load_input_stream(
    hidden_stream_t &input,
    float buf[SEQ_LEN][HIDDEN]) {
#pragma HLS INLINE off
LOAD_IN_SEQ:
    for (int i = 0; i < SEQ_LEN; i++) {
    LOAD_IN_PACK:
        for (int j_pack = 0; j_pack < HIDDEN / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
            bus_t pack = input.read();
        LOAD_IN_LANE:
            for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                buf[i][j_pack * PACK_SIZE + j_in] =
                    uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
            }
        }
    }
}

// ====================== UP PROJECTION + GELU ======================
static void up_projection_gelu(
    float input[SEQ_LEN][HIDDEN],
    const bus_t *w,
    const bus_t *b,
    float uram_buf[SEQ_LEN][FFN_DIM])
{
#pragma HLS INLINE off

    float w_pp[2][TILE_OUT_UP][TILE_K];
#ifdef BERT_SCALED_VARIANT
#pragma HLS BIND_STORAGE variable=w_pp type=RAM_2P impl=LUTRAM
#else
#pragma HLS BIND_STORAGE variable=w_pp type=RAM_2P impl=BRAM
#endif
#pragma HLS ARRAY_PARTITION variable=w_pp complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_pp cyclic factor=K_PAR dim=3

UP_J_BASE_LOOP:
    for (int j_base = 0; j_base < FFN_DIM; j_base += TILE_OUT_UP) {

        float acc[SEQ_LEN][TILE_OUT_UP];
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=MAC_GROUP dim=2

    UP_INIT_ACC:
        for (int s = 0; s < SEQ_LEN; s++) {
        UP_INIT_PACK:
            for (int j_pack = 0; j_pack < TILE_OUT_UP / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
                bus_t pack = b[(j_base / PACK_SIZE) + j_pack];
            UP_INIT_LANE:
                for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                    acc[s][j_pack * PACK_SIZE + j_in] =
                        uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
                }
            }
        }

        int ping = 0;

    UP_PRELOAD_FIRST:
        for (int tk = 0; tk < TILE_K; tk++) {
        UP_PRELOAD_PACK:
            for (int j_pack = 0; j_pack < TILE_OUT_UP / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
                bus_t pack = w[(tk * FFN_DIM + j_base) / PACK_SIZE + j_pack];
            UP_PRELOAD_LANE:
                for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                    w_pp[ping][j_pack * PACK_SIZE + j_in][tk] =
                        uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
                }
            }
        }

    UP_K_BASE_LOOP:
        for (int k_base = 0; k_base < HIDDEN; k_base += TILE_K) {
            int pong = ping ^ 1;

            if (k_base + TILE_K < HIDDEN) {
            UP_PRELOAD_NEXT_K:
                for (int tk = 0; tk < TILE_K; tk++) {
                UP_PRELOAD_NEXT_PACK:
                    for (int j_pack = 0; j_pack < TILE_OUT_UP / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
                        bus_t pack = w[((k_base + TILE_K + tk) * FFN_DIM + j_base) / PACK_SIZE + j_pack];
                    UP_PRELOAD_NEXT_LANE:
                        for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                            w_pp[pong][j_pack * PACK_SIZE + j_in][tk] =
                                uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
                        }
                    }
                }
            }

        UP_TK_CHUNK_LOOP:
            for (int tk = 0; tk < TILE_K; tk += K_PAR) {
            UP_S_LOOP:
                for (int s = 0; s < SEQ_LEN; ++s) {
                UP_J_LOOP:
                    for (int j = 0; j < TILE_OUT_UP; j += MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float in_vec[K_PAR];
                        float acc_vec[MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=in_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1

                    LOAD_IN_VEC:
                        for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                            in_vec[kk] = input[s][k_base + tk + kk];
                        }

                    LOAD_ACC_VEC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][j + jj];
                        }

                    MAC_VEC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            float sum = acc_vec[jj];
                        MAC_K_PAR:
                            for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                                sum += in_vec[kk] * w_pp[ping][j + jj][tk + kk];
                            }
                            acc_vec[jj] = sum;
                        }

                    STORE_ACC_VEC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc[s][j + jj] = acc_vec[jj];
                        }
                    }
                }
            }
            ping = pong;
        }

    UP_WRITE_GELU:
        for (int s = 0; s < SEQ_LEN; ++s) {
        UP_WRITE_PACK:
            for (int j_pack = 0; j_pack < TILE_OUT_UP; j_pack += PACK_SIZE) {
#pragma HLS PIPELINE II=1
            UP_WRITE_LANE:
                for (int j_in = 0; j_in < PACK_SIZE; ++j_in) {
#pragma HLS UNROLL
                    uram_buf[s][j_base + j_pack + j_in] = gelu(acc[s][j_pack + j_in]);
                }
            }
        }
    }
}

// ====================== DOWN PROJECTION ======================
static void down_projection(
    float uram_buf[SEQ_LEN][FFN_DIM],
    const bus_t *w,
    const bus_t *b,
    float output[SEQ_LEN][HIDDEN])
{
#pragma HLS INLINE off

    float w_pp[2][TILE_OUT_DOWN][TILE_K];
#ifdef BERT_SCALED_VARIANT
#pragma HLS BIND_STORAGE variable=w_pp type=RAM_2P impl=LUTRAM
#else
#pragma HLS BIND_STORAGE variable=w_pp type=RAM_2P impl=BRAM
#endif
#pragma HLS ARRAY_PARTITION variable=w_pp complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_pp cyclic factor=K_PAR dim=3

DOWN_J_BASE_LOOP:
    for (int j_base = 0; j_base < HIDDEN; j_base += TILE_OUT_DOWN) {

        float acc[SEQ_LEN][TILE_OUT_DOWN];
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=MAC_GROUP dim=2

    DOWN_INIT_ACC:
        for (int s = 0; s < SEQ_LEN; s++) {
        DOWN_INIT_PACK:
            for (int j_pack = 0; j_pack < TILE_OUT_DOWN / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
                bus_t pack = b[(j_base / PACK_SIZE) + j_pack];
            DOWN_INIT_LANE:
                for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                    acc[s][j_pack * PACK_SIZE + j_in] =
                        uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
                }
            }
        }

        int ping = 0;

    DOWN_PRELOAD_FIRST:
        for (int tk = 0; tk < TILE_K; tk++) {
        DOWN_PRELOAD_PACK:
            for (int j_pack = 0; j_pack < TILE_OUT_DOWN / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
                bus_t pack = w[(tk * HIDDEN + j_base) / PACK_SIZE + j_pack];
            DOWN_PRELOAD_LANE:
                for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                    w_pp[ping][j_pack * PACK_SIZE + j_in][tk] =
                        uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
                }
            }
        }

    DOWN_K_BASE_LOOP:
        for (int k_base = 0; k_base < FFN_DIM; k_base += TILE_K) {
            int pong = ping ^ 1;

            if (k_base + TILE_K < FFN_DIM) {
            DOWN_PRELOAD_NEXT_K:
                for (int tk = 0; tk < TILE_K; tk++) {
                DOWN_PRELOAD_NEXT_PACK:
                    for (int j_pack = 0; j_pack < TILE_OUT_DOWN / PACK_SIZE; j_pack++) {
#pragma HLS PIPELINE II=1
                        bus_t pack = w[((k_base + TILE_K + tk) * HIDDEN + j_base) / PACK_SIZE + j_pack];
                    DOWN_PRELOAD_NEXT_LANE:
                        for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                            w_pp[pong][j_pack * PACK_SIZE + j_in][tk] =
                                uint32_to_float((uint32_t)pack.range(31 + j_in * 32, j_in * 32));
                        }
                    }
                }
            }

        DOWN_TK_CHUNK_LOOP:
            for (int tk = 0; tk < TILE_K; tk += K_PAR) {
            DOWN_S_LOOP:
                for (int s = 0; s < SEQ_LEN; ++s) {
                DOWN_J_LOOP:
                    for (int j = 0; j < TILE_OUT_DOWN; j += MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float in_vec[K_PAR];
                        float acc_vec[MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=in_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1

                    LOAD_DN_IN_VEC:
                        for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                            in_vec[kk] = uram_buf[s][k_base + tk + kk];
                        }

                    LOAD_DN_ACC_VEC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][j + jj];
                        }

                    DN_MAC_VEC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            float sum = acc_vec[jj];
                        DN_MAC_K_PAR:
                            for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                                sum += in_vec[kk] * w_pp[ping][j + jj][tk + kk];
                            }
                            acc_vec[jj] = sum;
                        }

                    STORE_DN_ACC_VEC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc[s][j + jj] = acc_vec[jj];
                        }
                    }
                }
            }
            ping = pong;
        }

    DOWN_WRITE:
        for (int s = 0; s < SEQ_LEN; s++) {
        DOWN_WRITE_PACK:
            for (int j_pack = 0; j_pack < TILE_OUT_DOWN; j_pack += PACK_SIZE) {
#pragma HLS PIPELINE II=1
            DOWN_WRITE_LANE:
                for (int j_in = 0; j_in < PACK_SIZE; j_in++) {
#pragma HLS UNROLL
                    output[s][j_base + j_pack + j_in] = acc[s][j_pack + j_in];
                }
            }
        }
    }
}

// Produces one existing 32-wide FFN tile at a time.  TILE_OUT_UP, TILE_K,
// MAC_GROUP and K_PAR are unchanged from the 30m implementation.
static void ffn_up_tile_producer(
    float input[SEQ_LEN][HIDDEN],
    const bus_t *w,
    const bus_t *b,
    hidden_stream_t &gelu_tile_stream)
{
#pragma HLS INLINE off
    float w_tile[TILE_OUT_UP][TILE_K];
    float acc[SEQ_LEN][TILE_OUT_UP];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=K_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=MAC_GROUP dim=2

UP_DF_OUTPUT_TILE:
    for (int j_base = 0; j_base < FFN_DIM; j_base += TILE_OUT_UP) {
    UP_DF_INIT_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        UP_DF_INIT_PACK:
            for (int jp = 0; jp < TILE_OUT_UP / PACK_SIZE; ++jp) {
#pragma HLS PIPELINE II=1
                bus_t word = b[j_base / PACK_SIZE + jp];
            UP_DF_INIT_LANE:
                for (int lane = 0; lane < PACK_SIZE; ++lane) {
#pragma HLS UNROLL
                    acc[s][jp * PACK_SIZE + lane] = uint32_to_float(
                        (uint32_t)word.range(31 + lane * 32, lane * 32));
                }
            }
        }

    UP_DF_INPUT_TILE:
        for (int k_base = 0; k_base < HIDDEN; k_base += TILE_K) {
        UP_DF_LOAD_K:
            for (int tk = 0; tk < TILE_K; ++tk) {
            UP_DF_LOAD_PACK:
                for (int jp = 0; jp < TILE_OUT_UP / PACK_SIZE; ++jp) {
#pragma HLS PIPELINE II=1
                    bus_t word = w[((k_base + tk) * FFN_DIM + j_base) / PACK_SIZE + jp];
                UP_DF_LOAD_LANE:
                    for (int lane = 0; lane < PACK_SIZE; ++lane) {
#pragma HLS UNROLL
                        w_tile[jp * PACK_SIZE + lane][tk] = uint32_to_float(
                            (uint32_t)word.range(31 + lane * 32, lane * 32));
                    }
                }
            }

        UP_DF_K_CHUNK:
            for (int tk = 0; tk < TILE_K; tk += K_PAR) {
            UP_DF_ROW:
                for (int s = 0; s < SEQ_LEN; ++s) {
                UP_DF_COL:
                    for (int j = 0; j < TILE_OUT_UP; j += MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[K_PAR];
                        float acc_vec[MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
                    UP_DF_LOAD_X:
                        for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                            x_vec[kk] = input[s][k_base + tk + kk];
                        }
                    UP_DF_LOAD_ACC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][j + jj];
                        }
                    UP_DF_MAC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            float w_vec[K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        UP_DF_LOAD_W:
                            for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[j + jj][tk + kk];
                            }
                            acc_vec[jj] = accumulate_dot16(acc_vec[jj], x_vec, w_vec);
                        }
                    UP_DF_STORE_ACC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc[s][j + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    UP_DF_WRITE_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        UP_DF_WRITE_PACK:
            for (int jp = 0; jp < TILE_OUT_UP / PACK_SIZE; ++jp) {
#pragma HLS PIPELINE II=1
                float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            UP_DF_GELU_LANE:
                for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                    lane[i] = gelu(acc[s][jp * PACK_SIZE + i]);
                }
                gelu_tile_stream.write(pack_bus16(lane));
            }
        }
    }
}

static void ffn_down_tile_consumer(
    hidden_stream_t &gelu_tile_stream,
    const bus_t *w,
    const bus_t *b,
    float output[SEQ_LEN][HIDDEN])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][TILE_OUT_UP];
    float w_tile[TILE_OUT_DOWN][TILE_OUT_UP];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=K_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=K_PAR dim=2

DOWN_DF_INIT_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    DOWN_DF_INIT_PACK:
        for (int p = 0; p < HIDDEN / PACK_SIZE; ++p) {
#pragma HLS PIPELINE II=1
            bus_t word = b[p];
        DOWN_DF_INIT_LANE:
            for (int lane = 0; lane < PACK_SIZE; ++lane) {
#pragma HLS UNROLL
                output[s][p * PACK_SIZE + lane] = uint32_to_float(
                    (uint32_t)word.range(31 + lane * 32, lane * 32));
            }
        }
    }

DOWN_DF_FFN_TILE:
    for (int k_base = 0; k_base < FFN_DIM; k_base += TILE_OUT_UP) {
    DOWN_DF_READ_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        DOWN_DF_READ_PACK:
            for (int p = 0; p < TILE_OUT_UP / PACK_SIZE; ++p) {
#pragma HLS PIPELINE II=1
                float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
                unpack_bus16(gelu_tile_stream.read(), lane);
            DOWN_DF_READ_LANE:
                for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                    activation[s][p * PACK_SIZE + i] = lane[i];
                }
            }
        }

    DOWN_DF_OUTPUT_TILE:
        for (int j_base = 0; j_base < HIDDEN; j_base += TILE_OUT_DOWN) {
        DOWN_DF_LOAD_K:
            for (int tk = 0; tk < TILE_OUT_UP; ++tk) {
            DOWN_DF_LOAD_PACK:
                for (int jp = 0; jp < TILE_OUT_DOWN / PACK_SIZE; ++jp) {
#pragma HLS PIPELINE II=1
                    bus_t word = w[((k_base + tk) * HIDDEN + j_base) / PACK_SIZE + jp];
                DOWN_DF_LOAD_LANE:
                    for (int lane = 0; lane < PACK_SIZE; ++lane) {
#pragma HLS UNROLL
                        w_tile[jp * PACK_SIZE + lane][tk] = uint32_to_float(
                            (uint32_t)word.range(31 + lane * 32, lane * 32));
                    }
                }
            }

        DOWN_DF_K_CHUNK:
            for (int tk = 0; tk < TILE_OUT_UP; tk += K_PAR) {
            DOWN_DF_ROW:
                for (int s = 0; s < SEQ_LEN; ++s) {
                DOWN_DF_COL:
                    for (int j = 0; j < TILE_OUT_DOWN; j += MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[K_PAR];
                        float acc_vec[MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
                    DOWN_DF_LOAD_X:
                        for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }
                    DOWN_DF_LOAD_ACC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            acc_vec[jj] = output[s][j_base + j + jj];
                        }
                    DOWN_DF_MAC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            float w_vec[K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        DOWN_DF_LOAD_W:
                            for (int kk = 0; kk < K_PAR; ++kk) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[j + jj][tk + kk];
                            }
                            acc_vec[jj] = accumulate_dot16(acc_vec[jj], x_vec, w_vec);
                        }
                    DOWN_DF_STORE_ACC:
                        for (int jj = 0; jj < MAC_GROUP; ++jj) {
#pragma HLS UNROLL
                            output[s][j_base + j + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void ffn_projection_dataflow(
    float input[SEQ_LEN][HIDDEN],
    const bus_t *up_w,
    const bus_t *up_b,
    const bus_t *down_w,
    const bus_t *down_b,
    float output[SEQ_LEN][HIDDEN])
{
#pragma HLS INLINE off
    hidden_stream_t gelu_tile_stream("ffn_gelu_tile_stream");
#pragma HLS STREAM variable=gelu_tile_stream depth=512
#pragma HLS DATAFLOW
    ffn_up_tile_producer(input, up_w, up_b, gelu_tile_stream);
    ffn_down_tile_consumer(gelu_tile_stream, down_w, down_b, output);
}
void bert_ffn_stream(
    hidden_stream_t &input_hidden,
    const bus_t     *up_w,
    const bus_t     *up_b,
    const bus_t     *down_w,
    const bus_t     *down_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &output_hidden
) {
#pragma HLS INLINE off

    static float input_buf[SEQ_LEN][HIDDEN];
#pragma HLS BIND_STORAGE variable=input_buf type=RAM_2P impl=URAM latency=3
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=K_PAR dim=2

#ifndef BERT_STREAMING_DATAFLOW
    static float uram_buf[SEQ_LEN][FFN_DIM];
#pragma HLS BIND_STORAGE variable=uram_buf type=RAM_2P impl=URAM latency=3
#pragma HLS ARRAY_PARTITION variable=uram_buf cyclic factor=K_PAR dim=2
#endif

    static float output_buf[SEQ_LEN][HIDDEN];
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=3
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=MAC_GROUP dim=2

    load_input_stream(input_hidden, input_buf);
#ifdef BERT_STREAMING_DATAFLOW
    ffn_projection_dataflow(
        input_buf, up_w, up_b, down_w, down_b, output_buf);
#else
    up_projection_gelu(input_buf, up_w, up_b, uram_buf);
    down_projection(uram_buf, down_w, down_b, output_buf);
#endif
    matrix_add_norm_to_stream(
        input_buf,
        output_buf,
        norm_gamma,
        norm_beta,
        output_hidden);
}

// ============================================================================
// External split-kernel stages
// ============================================================================

static void load_hidden_ddr_to_ffn_matrix(
    const bus_t *input_hidden,
    float hidden[SEQ_LEN][HIDDEN])
{
#pragma HLS INLINE off
LOAD_FFN_RESIDUAL_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    LOAD_FFN_RESIDUAL_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            unpack_bus16(input_hidden[s * PACKS + p], lane);
        LOAD_FFN_RESIDUAL_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                hidden[s][p * PACK_SIZE + i] = lane[i];
            }
        }
    }
}

void bert_ffn_up_gelu_stage(
    hidden_stream_t &attn_mid_stream,
    const bus_t     *up_w,
    const bus_t     *up_b,
    hidden_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    static float input_buf[SEQ_LEN][HIDDEN];
#pragma HLS BIND_STORAGE variable=input_buf type=RAM_2P impl=URAM latency=3
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=K_PAR dim=2

    load_input_stream(attn_mid_stream, input_buf);
    ffn_up_tile_producer(input_buf, up_w, up_b, gelu_stream);
}

void bert_ffn_down_norm_stage(
    hidden_stream_t &gelu_stream,
    hidden_stream_t &attn_mid_residual_stream,
    const bus_t     *down_w,
    const bus_t     *down_b,
    const bus_t     *norm_gamma,
    const bus_t     *norm_beta,
    hidden_stream_t &hidden_out_stream)
{
#pragma HLS INLINE off
    static float residual_buf[SEQ_LEN][HIDDEN];
    static float output_buf[SEQ_LEN][HIDDEN];
#pragma HLS BIND_STORAGE variable=residual_buf type=RAM_2P impl=URAM latency=3
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=3
#pragma HLS ARRAY_PARTITION variable=residual_buf cyclic factor=K_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=MAC_GROUP dim=2

#pragma HLS DATAFLOW
    load_input_stream(attn_mid_residual_stream, residual_buf);
    ffn_down_tile_consumer(gelu_stream, down_w, down_b, output_buf);
    matrix_add_norm_to_stream(
        residual_buf, output_buf, norm_gamma, norm_beta, hidden_out_stream);
}
