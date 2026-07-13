#include "bert_ffn_kernel_lab.h"

#include <ap_int.h>
#include <hls_stream.h>

#define PACK_BITS 512
#define PACK_FLOATS 16
#define TILE_I_PACKS (TILE_I / PACK_FLOATS)
#define OUT_PART_BANKS 4
#define HIDDEN_PACKS (HIDDEN_SIZE / PACK_FLOATS)
#define INTERMEDIATE_PACKS (INTERMEDIATE_SIZE / PACK_FLOATS)
#define V10_TILE_OUT 32
#define V10_TILE_K 64
#define V10_K_PAR 16
#define V10_MAC_GROUP 16
#define V10_GELU_STREAM_DEPTH 512
#define V13_TILE_I 16
#define V13_TILE_O 32
#define V15_W1_TILE_PACKS (V13_TILE_I / PACK_FLOATS)
#define V15_W2_TILE_PACKS (V13_TILE_O / PACK_FLOATS)
#define V15_INTERMEDIATE_TILES (INTERMEDIATE_SIZE / V13_TILE_I)
#define V15_HIDDEN_TILES (HIDDEN_SIZE / V13_TILE_O)
#define V14_MAC_GROUP 8
#define V21_TILE_I 96
#define V21_TILE_O 96
#define V21_MAC_GROUP 24
#define V21_W1_TILE_PACKS (V21_TILE_I / PACK_FLOATS)
#define V21_W2_TILE_PACKS (V21_TILE_O / PACK_FLOATS)
#define V21_WEIGHT_GROUP_PACKS 2
#define V21_W1_WEIGHT_GROUPS (V21_TILE_I / V21_MAC_GROUP)
#define V21_W2_WEIGHT_GROUPS (V21_TILE_O / V21_MAC_GROUP)
#define V21_W1_WEIGHT_PACKS (V21_W1_WEIGHT_GROUPS * V21_WEIGHT_GROUP_PACKS)
#define V21_W2_WEIGHT_PACKS (V21_W2_WEIGHT_GROUPS * V21_WEIGHT_GROUP_PACKS)
#define V21_INTERMEDIATE_TILES (INTERMEDIATE_SIZE / V21_TILE_I)
#define V21_HIDDEN_TILES (HIDDEN_SIZE / V21_TILE_O)

#if ((HIDDEN_SIZE % V10_TILE_K) != 0)
#error "HIDDEN_SIZE must be divisible by V10_TILE_K"
#endif

#if ((HIDDEN_SIZE % V10_TILE_OUT) != 0)
#error "HIDDEN_SIZE must be divisible by V10_TILE_OUT"
#endif

#if ((INTERMEDIATE_SIZE % V10_TILE_OUT) != 0)
#error "INTERMEDIATE_SIZE must be divisible by V10_TILE_OUT"
#endif

#if ((V10_TILE_K % V10_K_PAR) != 0)
#error "V10_TILE_K must be divisible by V10_K_PAR"
#endif

#if ((V10_TILE_OUT % V10_MAC_GROUP) != 0)
#error "V10_TILE_OUT must be divisible by V10_MAC_GROUP"
#endif

#if ((V10_TILE_OUT % PACK_FLOATS) != 0)
#error "V10_TILE_OUT must be divisible by PACK_FLOATS"
#endif

#if ((INTERMEDIATE_SIZE % V13_TILE_I) != 0)
#error "INTERMEDIATE_SIZE must be divisible by V13_TILE_I"
#endif

#if ((V13_TILE_I % PACK_FLOATS) != 0)
#error "V13_TILE_I must be divisible by PACK_FLOATS"
#endif

#if ((V13_TILE_I % V10_K_PAR) != 0)
#error "V13_TILE_I must be divisible by V10_K_PAR"
#endif

#if ((HIDDEN_SIZE % V13_TILE_O) != 0)
#error "HIDDEN_SIZE must be divisible by V13_TILE_O"
#endif

#if ((V13_TILE_O % V10_MAC_GROUP) != 0)
#error "V13_TILE_O must be divisible by V10_MAC_GROUP"
#endif

#if ((V13_TILE_O % PACK_FLOATS) != 0)
#error "V13_TILE_O must be divisible by PACK_FLOATS"
#endif

#if (V15_W2_TILE_PACKS != 2)
#error "V15 W2 load timing path assumes two packed words per output tile"
#endif

#if ((V10_TILE_OUT % V14_MAC_GROUP) != 0)
#error "V10_TILE_OUT must be divisible by V14_MAC_GROUP"
#endif

#if ((INTERMEDIATE_SIZE % V21_TILE_I) != 0)
#error "INTERMEDIATE_SIZE must be divisible by V21_TILE_I"
#endif

#if ((HIDDEN_SIZE % V21_TILE_O) != 0)
#error "HIDDEN_SIZE must be divisible by V21_TILE_O"
#endif

#if ((V21_TILE_I % V21_MAC_GROUP) != 0)
#error "V21_TILE_I must be divisible by V21_MAC_GROUP"
#endif

#if ((V21_TILE_O % V21_MAC_GROUP) != 0)
#error "V21_TILE_O must be divisible by V21_MAC_GROUP"
#endif

#if ((V21_TILE_I % PACK_FLOATS) != 0)
#error "V21_TILE_I must be divisible by PACK_FLOATS"
#endif

#if ((V21_TILE_O % PACK_FLOATS) != 0)
#error "V21_TILE_O must be divisible by PACK_FLOATS"
#endif

#if ((V21_MAC_GROUP != 24) || (V21_WEIGHT_GROUP_PACKS != 2))
#error "V21 padded weight loader requires 24 lanes stored in two packed words"
#endif

#if ((V21_W1_WEIGHT_PACKS != 8) || (V21_W2_WEIGHT_PACKS != 8))
#error "V21 padded weight loader address decode requires eight words per tile row"
#endif

typedef ap_uint<PACK_BITS> packed_float16_t;
typedef hls::stream<packed_float16_t> packed_float_stream_t;

static float gelu_pwl_packed(float x) {
#pragma HLS INLINE
    float gate;

    if (x <= -3.0f) {
        return 0.0f;
    }
    if (x >= 3.0f) {
        return x;
    }

    gate = 0.5f + x * (1.0f / 6.0f);
    return x * gate;
}

static float bits_to_float(ap_uint<32> bits) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } conv;

    conv.u = (unsigned int)bits;
    return conv.f;
}

static ap_uint<32> float_to_bits(float value) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } conv;

    conv.f = value;
    return conv.u;
}

static float get_packed_float(packed_float16_t word, int lane) {
#pragma HLS INLINE
    switch (lane) {
    case 0:
        return bits_to_float(word.range(31, 0));
    case 1:
        return bits_to_float(word.range(63, 32));
    case 2:
        return bits_to_float(word.range(95, 64));
    case 3:
        return bits_to_float(word.range(127, 96));
    case 4:
        return bits_to_float(word.range(159, 128));
    case 5:
        return bits_to_float(word.range(191, 160));
    case 6:
        return bits_to_float(word.range(223, 192));
    case 7:
        return bits_to_float(word.range(255, 224));
    case 8:
        return bits_to_float(word.range(287, 256));
    case 9:
        return bits_to_float(word.range(319, 288));
    case 10:
        return bits_to_float(word.range(351, 320));
    case 11:
        return bits_to_float(word.range(383, 352));
    case 12:
        return bits_to_float(word.range(415, 384));
    case 13:
        return bits_to_float(word.range(447, 416));
    case 14:
        return bits_to_float(word.range(479, 448));
    default:
        return bits_to_float(word.range(511, 480));
    }
}

static void set_packed_float(packed_float16_t *word, int lane, float value) {
#pragma HLS INLINE
    ap_uint<32> bits = float_to_bits(value);

    switch (lane) {
    case 0:
        word->range(31, 0) = bits;
        break;
    case 1:
        word->range(63, 32) = bits;
        break;
    case 2:
        word->range(95, 64) = bits;
        break;
    case 3:
        word->range(127, 96) = bits;
        break;
    case 4:
        word->range(159, 128) = bits;
        break;
    case 5:
        word->range(191, 160) = bits;
        break;
    case 6:
        word->range(223, 192) = bits;
        break;
    case 7:
        word->range(255, 224) = bits;
        break;
    case 8:
        word->range(287, 256) = bits;
        break;
    case 9:
        word->range(319, 288) = bits;
        break;
    case 10:
        word->range(351, 320) = bits;
        break;
    case 11:
        word->range(383, 352) = bits;
        break;
    case 12:
        word->range(415, 384) = bits;
        break;
    case 13:
        word->range(447, 416) = bits;
        break;
    case 14:
        word->range(479, 448) = bits;
        break;
    default:
        word->range(511, 480) = bits;
        break;
    }
}

static void unpack_packed_float16(packed_float16_t word, float values[PACK_FLOATS])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
    values[0] = bits_to_float(word.range(31, 0));
    values[1] = bits_to_float(word.range(63, 32));
    values[2] = bits_to_float(word.range(95, 64));
    values[3] = bits_to_float(word.range(127, 96));
    values[4] = bits_to_float(word.range(159, 128));
    values[5] = bits_to_float(word.range(191, 160));
    values[6] = bits_to_float(word.range(223, 192));
    values[7] = bits_to_float(word.range(255, 224));
    values[8] = bits_to_float(word.range(287, 256));
    values[9] = bits_to_float(word.range(319, 288));
    values[10] = bits_to_float(word.range(351, 320));
    values[11] = bits_to_float(word.range(383, 352));
    values[12] = bits_to_float(word.range(415, 384));
    values[13] = bits_to_float(word.range(447, 416));
    values[14] = bits_to_float(word.range(479, 448));
    values[15] = bits_to_float(word.range(511, 480));
}

static packed_float16_t pack_packed_float16(const float values[PACK_FLOATS])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
    packed_float16_t word = 0;
    word.range(31, 0) = float_to_bits(values[0]);
    word.range(63, 32) = float_to_bits(values[1]);
    word.range(95, 64) = float_to_bits(values[2]);
    word.range(127, 96) = float_to_bits(values[3]);
    word.range(159, 128) = float_to_bits(values[4]);
    word.range(191, 160) = float_to_bits(values[5]);
    word.range(223, 192) = float_to_bits(values[6]);
    word.range(255, 224) = float_to_bits(values[7]);
    word.range(287, 256) = float_to_bits(values[8]);
    word.range(319, 288) = float_to_bits(values[9]);
    word.range(351, 320) = float_to_bits(values[10]);
    word.range(383, 352) = float_to_bits(values[11]);
    word.range(415, 384) = float_to_bits(values[12]);
    word.range(447, 416) = float_to_bits(values[13]);
    word.range(479, 448) = float_to_bits(values[14]);
    word.range(511, 480) = float_to_bits(values[15]);
    return word;
}

static float v10_dot16_tree(
    const float lhs[V10_K_PAR],
    const float rhs[V10_K_PAR])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
    float product[V10_K_PAR];
    float s8[8];
    float s4[4];
    float s2[2];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1

v10_dot_mul:
    for (int i = 0; i < V10_K_PAR; i++) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }

v10_dot_l1:
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        s8[i] = product[2 * i] + product[2 * i + 1];
    }

v10_dot_l2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        s4[i] = s8[2 * i] + s8[2 * i + 1];
    }

v10_dot_l3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        s2[i] = s4[2 * i] + s4[2 * i + 1];
    }

    return s2[0] + s2[1];
}

static void v10_load_input_matrix(
    packed_float16_t *input_hidden,
    float input_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off

v10_load_input_s:
    for (int s = 0; s < SEQ_LEN; s++) {
        int input_index = s * HIDDEN_PACKS;
    v10_load_input_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            packed_float16_t word = input_hidden[input_index + h_pack];
            unpack_packed_float16(word, lanes);
        v10_load_input_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                input_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }
}

static void v10_store_output_matrix(
    float output_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *output_hidden)
{
#pragma HLS INLINE off

v10_store_output_s:
    for (int s = 0; s < SEQ_LEN; s++) {
        int output_index = s * HIDDEN_PACKS;
    v10_store_output_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            int h_base = h_pack << 4;
        v10_store_output_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                lanes[lane] = output_buf[s][h_base + lane];
            }
            output_hidden[output_index + h_pack] = pack_packed_float16(lanes);
        }
    }
}

static void v16_store_output_matrix_direct(
    float output_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *output_hidden)
{
#pragma HLS INLINE off

v16_store_output_s:
    for (int s = 0; s < SEQ_LEN; s++) {
        int output_index = s * HIDDEN_PACKS;
    v16_store_output_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            int h = h_pack << 4;
            packed_float16_t word = 0;

            word.range(31, 0) = float_to_bits(output_buf[s][h + 0]);
            word.range(63, 32) = float_to_bits(output_buf[s][h + 1]);
            word.range(95, 64) = float_to_bits(output_buf[s][h + 2]);
            word.range(127, 96) = float_to_bits(output_buf[s][h + 3]);
            word.range(159, 128) = float_to_bits(output_buf[s][h + 4]);
            word.range(191, 160) = float_to_bits(output_buf[s][h + 5]);
            word.range(223, 192) = float_to_bits(output_buf[s][h + 6]);
            word.range(255, 224) = float_to_bits(output_buf[s][h + 7]);
            word.range(287, 256) = float_to_bits(output_buf[s][h + 8]);
            word.range(319, 288) = float_to_bits(output_buf[s][h + 9]);
            word.range(351, 320) = float_to_bits(output_buf[s][h + 10]);
            word.range(383, 352) = float_to_bits(output_buf[s][h + 11]);
            word.range(415, 384) = float_to_bits(output_buf[s][h + 12]);
            word.range(447, 416) = float_to_bits(output_buf[s][h + 13]);
            word.range(479, 448) = float_to_bits(output_buf[s][h + 14]);
            word.range(511, 480) = float_to_bits(output_buf[s][h + 15]);

            output_hidden[output_index + h_pack] = word;
        }
    }
}

static void v18_store_output_matrix_staged(
    float output_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *output_hidden)
{
#pragma HLS INLINE off
    packed_float16_t out_word_buf[HIDDEN_PACKS];
#pragma HLS BIND_STORAGE variable=out_word_buf type=RAM_2P impl=BRAM latency=2

v18_store_output_s:
    for (int s = 0; s < SEQ_LEN; s++) {
        int output_index = s * HIDDEN_PACKS;
    v18_store_output_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            int h = h_pack << 4;
            packed_float16_t word = 0;

            word.range(31, 0) = float_to_bits(output_buf[s][h + 0]);
            word.range(63, 32) = float_to_bits(output_buf[s][h + 1]);
            word.range(95, 64) = float_to_bits(output_buf[s][h + 2]);
            word.range(127, 96) = float_to_bits(output_buf[s][h + 3]);
            word.range(159, 128) = float_to_bits(output_buf[s][h + 4]);
            word.range(191, 160) = float_to_bits(output_buf[s][h + 5]);
            word.range(223, 192) = float_to_bits(output_buf[s][h + 6]);
            word.range(255, 224) = float_to_bits(output_buf[s][h + 7]);
            word.range(287, 256) = float_to_bits(output_buf[s][h + 8]);
            word.range(319, 288) = float_to_bits(output_buf[s][h + 9]);
            word.range(351, 320) = float_to_bits(output_buf[s][h + 10]);
            word.range(383, 352) = float_to_bits(output_buf[s][h + 11]);
            word.range(415, 384) = float_to_bits(output_buf[s][h + 12]);
            word.range(447, 416) = float_to_bits(output_buf[s][h + 13]);
            word.range(479, 448) = float_to_bits(output_buf[s][h + 14]);
            word.range(511, 480) = float_to_bits(output_buf[s][h + 15]);

            out_word_buf[h_pack] = word;
        }

    v18_store_output_write:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            output_hidden[output_index + h_pack] = out_word_buf[h_pack];
        }
    }
}

static void v11_load_input_stream_matrix(
    packed_float_stream_t &input_stream,
    float input_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off

v11_load_input_stream_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v11_load_input_stream_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            packed_float16_t word = input_stream.read();
            unpack_packed_float16(word, lanes);
        v11_load_input_stream_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                input_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }
}

static void v10_ffn_up_tile_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V10_TILE_OUT][V10_TILE_K];
    float acc[SEQ_LEN][V10_TILE_OUT];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

v10_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V10_TILE_OUT) {
    v10_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v10_up_init_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = b1[i0 / PACK_FLOATS + i_pack];
            v10_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v10_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
        v10_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
            v10_up_load_w_pack:
                for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = ((h0 + tk) * INTERMEDIATE_SIZE + i0) / PACK_FLOATS + i_pack;
                    packed_float16_t word = w1[packed_index];
                v10_up_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[i_pack * PACK_FLOATS + lane][tk] = get_packed_float(word, lane);
                    }
                }
            }

        v10_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v10_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v10_up_mac_group:
                    for (int io = 0; io < V10_TILE_OUT; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1

                    v10_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v10_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v10_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v10_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v10_dot16_tree(x_vec, w_vec);
                        }

                    v10_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v10_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v10_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = 0;
            v10_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    float value = gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                    set_packed_float(&word, lane, value);
                }
                gelu_stream.write(word);
            }
        }
    }
}

static void v10_ffn_down_tile_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V10_TILE_OUT];
    float w_tile[V10_TILE_OUT][V10_TILE_OUT];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2

v10_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v10_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = b2[h_pack];
        v10_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
            }
        }
    }

v10_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V10_TILE_OUT) {
    v10_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v10_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = gelu_stream.read();
            v10_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v10_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V10_TILE_OUT) {
        v10_down_load_w_k:
            for (int tk = 0; tk < V10_TILE_OUT; tk++) {
            v10_down_load_w_pack:
                for (int o_pack = 0; o_pack < V10_TILE_OUT / PACK_FLOATS; o_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = ((i0 + tk) * HIDDEN_SIZE + o0) / PACK_FLOATS + o_pack;
                    packed_float16_t word = w2[packed_index];
                v10_down_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[o_pack * PACK_FLOATS + lane][tk] = get_packed_float(word, lane);
                    }
                }
            }

        v10_down_k_chunk:
            for (int tk = 0; tk < V10_TILE_OUT; tk += V10_K_PAR) {
            v10_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v10_down_mac_group:
                    for (int oo = 0; oo < V10_TILE_OUT; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1

                    v10_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v10_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v10_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v10_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v10_dot16_tree(x_vec, w_vec);
                        }

                    v10_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v10_projection_dataflow(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v10_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v10_ffn_up_tile_producer(input_buf, w1, b1, gelu_stream);
    v10_ffn_down_tile_consumer(gelu_stream, w2, b2, output_buf);
}

static float v12_dot16_tree_timing(
    const float lhs[V10_K_PAR],
    const float rhs[V10_K_PAR])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
    float product[V10_K_PAR];
    float s8[8];
    float s4[4];
    float s2[2];
    float sum;
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp latency=4
#pragma HLS BIND_OP variable=s8 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=s4 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=s2 op=fadd impl=fulldsp latency=5
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=5

v12_dot_mul:
    for (int i = 0; i < V10_K_PAR; i++) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }

v12_dot_l1:
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        s8[i] = product[2 * i] + product[2 * i + 1];
    }

v12_dot_l2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        s4[i] = s8[2 * i] + s8[2 * i + 1];
    }

v12_dot_l3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        s2[i] = s4[2 * i] + s4[2 * i + 1];
    }

    sum = s2[0] + s2[1];
    return sum;
}

static float v13_dot16_tree_margin(
    const float lhs[V10_K_PAR],
    const float rhs[V10_K_PAR])
{
#ifdef FFN_V21_DOT_PIPELINE
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#else
#pragma HLS INLINE
#endif
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
#ifdef FFN_V21_DOT_PIPELINE
    float p0, p1, p2, p3, p4, p5, p6, p7;
    float p8, p9, p10, p11, p12, p13, p14, p15;
#else
    float product[V10_K_PAR];
#endif
    float s8[8];
    float s4[4];
    float s2[2];
    float sum;
#ifndef FFN_V21_DOT_PIPELINE
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#endif
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#ifdef FFN_V21_DOT_PIPELINE
#pragma HLS BIND_OP variable=p0 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p1 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p2 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p3 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p4 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p5 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p6 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p7 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p8 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p9 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p10 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p11 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p12 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p13 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p14 op=fmul impl=meddsp latency=7
#pragma HLS BIND_OP variable=p15 op=fmul impl=meddsp latency=7
#elif defined(FFN_V20_COMPUTE_MARGIN)
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp latency=5
#else
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp latency=4
#endif
#pragma HLS BIND_OP variable=s8 op=fadd impl=fulldsp latency=6
#pragma HLS BIND_OP variable=s4 op=fadd impl=fulldsp latency=6
#pragma HLS BIND_OP variable=s2 op=fadd impl=fulldsp latency=6
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=6

#ifdef FFN_V21_DOT_PIPELINE
    p0 = lhs[0] * rhs[0];
    p1 = lhs[1] * rhs[1];
    p2 = lhs[2] * rhs[2];
    p3 = lhs[3] * rhs[3];
    p4 = lhs[4] * rhs[4];
    p5 = lhs[5] * rhs[5];
    p6 = lhs[6] * rhs[6];
    p7 = lhs[7] * rhs[7];
    p8 = lhs[8] * rhs[8];
    p9 = lhs[9] * rhs[9];
    p10 = lhs[10] * rhs[10];
    p11 = lhs[11] * rhs[11];
    p12 = lhs[12] * rhs[12];
    p13 = lhs[13] * rhs[13];
    p14 = lhs[14] * rhs[14];
    p15 = lhs[15] * rhs[15];

    s8[0] = p0 + p1;
    s8[1] = p2 + p3;
    s8[2] = p4 + p5;
    s8[3] = p6 + p7;
    s8[4] = p8 + p9;
    s8[5] = p10 + p11;
    s8[6] = p12 + p13;
    s8[7] = p14 + p15;
#else
v13_dot_mul:
    for (int i = 0; i < V10_K_PAR; i++) {
#pragma HLS UNROLL
        product[i] = lhs[i] * rhs[i];
    }

v13_dot_l1:
    for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
        s8[i] = product[2 * i] + product[2 * i + 1];
    }
#endif

v13_dot_l2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        s4[i] = s8[2 * i] + s8[2 * i + 1];
    }

v13_dot_l3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        s2[i] = s4[2 * i] + s4[2 * i + 1];
    }

    sum = s2[0] + s2[1];
    return sum;
}

static void v12_ffn_up_tile_producer_timing(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V10_TILE_OUT][V10_TILE_K];
    float acc[SEQ_LEN][V10_TILE_OUT];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

v12_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V10_TILE_OUT) {
    v12_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v12_up_init_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = b1[i0 / PACK_FLOATS + i_pack];
            v12_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v12_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
        v12_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
            v12_up_load_w_pack:
                for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = ((h0 + tk) * INTERMEDIATE_SIZE + i0) / PACK_FLOATS + i_pack;
                    packed_float16_t word = w1[packed_index];
                v12_up_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[i_pack * PACK_FLOATS + lane][tk] = get_packed_float(word, lane);
                    }
                }
            }

        v12_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v12_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v12_up_mac_group:
                    for (int io = 0; io < V10_TILE_OUT; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v12_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v12_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v12_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v12_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v12_dot16_tree_timing(x_vec, w_vec);
                        }

                    v12_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v12_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v12_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = 0;
            v12_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    float value = gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                    set_packed_float(&word, lane, value);
                }
                gelu_stream.write(word);
            }
        }
    }
}

static void v12_ffn_down_tile_consumer_timing(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V10_TILE_OUT];
    float w_tile[V10_TILE_OUT][V10_TILE_OUT];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2

v12_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v12_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = b2[h_pack];
        v12_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
            }
        }
    }

v12_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V10_TILE_OUT) {
    v12_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v12_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = gelu_stream.read();
            v12_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v12_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V10_TILE_OUT) {
        v12_down_load_w_k:
            for (int tk = 0; tk < V10_TILE_OUT; tk++) {
            v12_down_load_w_pack:
                for (int o_pack = 0; o_pack < V10_TILE_OUT / PACK_FLOATS; o_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = ((i0 + tk) * HIDDEN_SIZE + o0) / PACK_FLOATS + o_pack;
                    packed_float16_t word = w2[packed_index];
                v12_down_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[o_pack * PACK_FLOATS + lane][tk] = get_packed_float(word, lane);
                    }
                }
            }

        v12_down_k_chunk:
            for (int tk = 0; tk < V10_TILE_OUT; tk += V10_K_PAR) {
            v12_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v12_down_mac_group:
                    for (int oo = 0; oo < V10_TILE_OUT; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v12_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v12_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v12_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v12_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v12_dot16_tree_timing(x_vec, w_vec);
                        }

                    v12_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v12_projection_dataflow_timing(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v12_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v12_ffn_up_tile_producer_timing(input_buf, w1, b1, gelu_stream);
    v12_ffn_down_tile_consumer_timing(gelu_stream, w2, b2, output_buf);
}

static void v13_ffn_up_tile16_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V13_TILE_I][V10_TILE_K];
    float acc[SEQ_LEN][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

v13_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
    v13_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v13_up_init_pack:
            for (int i_pack = 0; i_pack < V13_TILE_I / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = b1[i0_pack + i_pack];
                unpack_packed_float16(word, lanes);
            v13_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

    v13_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
            int packed_base = h0 * INTERMEDIATE_PACKS + i0_pack;
        v13_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
            v13_up_load_w_pack:
                for (int i_pack = 0; i_pack < V13_TILE_I / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                    float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                    packed_float16_t word = w1[packed_base + i_pack];
                    unpack_packed_float16(word, lanes);
                v13_up_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[i_pack * PACK_FLOATS + lane][tk] = lanes[lane];
                    }
                }
                packed_base += INTERMEDIATE_PACKS;
            }

        v13_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v13_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v13_up_mac_group:
                    for (int io = 0; io < V13_TILE_I; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v13_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v13_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v13_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v13_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v13_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v13_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v13_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V13_TILE_I / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            v13_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    lanes[lane] = gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                }
                packed_float16_t word = pack_packed_float16(lanes);
                gelu_stream.write(word);
            }
        }
    }
}

static void v13_ffn_down_tile16_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V13_TILE_I];
    float w_tile[V13_TILE_O][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=2

v13_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v13_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            packed_float16_t word = b2[h_pack];
            unpack_packed_float16(word, lanes);
        v13_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }

v13_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
    v13_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v13_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V13_TILE_I / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = gelu_stream.read();
                unpack_packed_float16(word, lanes);
            v13_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

    v13_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V13_TILE_O) {
            int o0_pack = o0 / PACK_FLOATS;
            int packed_base = i0 * HIDDEN_PACKS + o0_pack;
        v13_down_load_w_k:
            for (int tk = 0; tk < V13_TILE_I; tk++) {
            v13_down_load_w_pack:
                for (int o_pack = 0; o_pack < V13_TILE_O / PACK_FLOATS; o_pack++) {
#pragma HLS PIPELINE II=1
                    float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                    packed_float16_t word = w2[packed_base + o_pack];
                    unpack_packed_float16(word, lanes);
                v13_down_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[o_pack * PACK_FLOATS + lane][tk] = lanes[lane];
                    }
                }
                packed_base += HIDDEN_PACKS;
            }

        v13_down_k_chunk:
            for (int tk = 0; tk < V13_TILE_I; tk += V10_K_PAR) {
            v13_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v13_down_mac_group:
                    for (int oo = 0; oo < V13_TILE_O; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v13_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v13_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v13_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v13_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v13_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v13_projection_dataflow_tile16(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v13_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v13_ffn_up_tile16_producer(input_buf, w1, b1, gelu_stream);
    v13_ffn_down_tile16_consumer(gelu_stream, w2, b2, output_buf);
}

static void v15_ffn_up_tilemajor_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V13_TILE_I][V10_TILE_K];
    float acc[SEQ_LEN][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

v15_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
        int i_tile = i0 / V13_TILE_I;
    v15_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v15_up_init_pack:
            for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = b1[i0_pack + i_pack];
                unpack_packed_float16(word, lanes);
            v15_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

    v15_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
            int packed_base = i_tile * HIDDEN_SIZE * V15_W1_TILE_PACKS + h0 * V15_W1_TILE_PACKS;
        v15_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
            v15_up_load_w_pack:
                for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                    float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                    packed_float16_t word = w1[packed_base + i_pack];
                    unpack_packed_float16(word, lanes);
                v15_up_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[i_pack * PACK_FLOATS + lane][tk] = lanes[lane];
                    }
                }
                packed_base += V15_W1_TILE_PACKS;
            }

        v15_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v15_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v15_up_mac_group:
                    for (int io = 0; io < V13_TILE_I; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v15_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v15_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v15_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v15_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v15_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v15_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v15_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            v15_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    lanes[lane] = gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                }
                packed_float16_t word = pack_packed_float16(lanes);
                gelu_stream.write(word);
            }
        }
    }
}

static void v15_ffn_down_tilemajor_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V13_TILE_I];
    float w_tile[V13_TILE_O][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=2

v15_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v15_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            packed_float16_t word = b2[h_pack];
            unpack_packed_float16(word, lanes);
        v15_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }

v15_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i_tile = i0 / V13_TILE_I;
    v15_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v15_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = gelu_stream.read();
                unpack_packed_float16(word, lanes);
            v15_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

        v15_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V13_TILE_O) {
            int o_tile = o0 / V13_TILE_O;
            int packed_base = ((i_tile * V15_HIDDEN_TILES + o_tile) * V13_TILE_I) * V15_W2_TILE_PACKS;
        v15_down_load_w_k:
            for (int packed_offset = 0; packed_offset < V13_TILE_I * V15_W2_TILE_PACKS; packed_offset++) {
#pragma HLS PIPELINE II=1
                int tk = packed_offset >> 1;
                int o_pack = packed_offset & 1;
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = w2[packed_base + packed_offset];
                unpack_packed_float16(word, lanes);
            v15_down_load_w_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    w_tile[o_pack * PACK_FLOATS + lane][tk] = lanes[lane];
                }
            }

        v15_down_k_chunk:
            for (int tk = 0; tk < V13_TILE_I; tk += V10_K_PAR) {
            v15_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v15_down_mac_group:
                    for (int oo = 0; oo < V13_TILE_O; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v15_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v15_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v15_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v15_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v15_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v15_projection_dataflow_tilemajor(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v15_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v15_ffn_up_tilemajor_producer(input_buf, w1, b1, gelu_stream);
    v15_ffn_down_tilemajor_consumer(gelu_stream, w2, b2, output_buf);
}

#ifdef FFN_V21_TILE96_EXPERIMENT
static void v21_ffn_up_tile96_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V21_TILE_I][V10_TILE_K];
    float acc[SEQ_LEN][V21_TILE_I];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_1P impl=LUTRAM latency=1
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=24 dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=24 dim=2

v21_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V21_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
        int i_tile = i0 / V21_TILE_I;
    v21_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v21_up_init_pack:
            for (int i_pack = 0; i_pack < V21_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = b1[i0_pack + i_pack];
                unpack_packed_float16(word, lanes);
            v21_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

        v21_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
            int packed_base =
                i_tile * HIDDEN_SIZE * V21_W1_WEIGHT_PACKS
                + h0 * V21_W1_WEIGHT_PACKS;
        v21_up_load_w_full:
            for (int word_offset = 0;
                 word_offset < V10_TILE_K * V21_W1_WEIGHT_GROUPS;
                 word_offset++) {
#pragma HLS PIPELINE II=1
                int tk = word_offset >> 2;
                int group = word_offset & 3;
                int group_base = group * V21_MAC_GROUP;
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                unpack_packed_float16(
                    w1[packed_base + tk * V21_W1_WEIGHT_PACKS + group * V21_WEIGHT_GROUP_PACKS],
                    lanes);
            v21_up_load_w_full_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    w_tile[group_base + lane][tk] = lanes[lane];
                }
            }

        v21_up_load_w_tail:
            for (int word_offset = 0;
                 word_offset < V10_TILE_K * V21_W1_WEIGHT_GROUPS;
                 word_offset++) {
#pragma HLS PIPELINE II=1
                int tk = word_offset >> 2;
                int group = word_offset & 3;
                int group_base = group * V21_MAC_GROUP + PACK_FLOATS;
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                unpack_packed_float16(
                    w1[packed_base + tk * V21_W1_WEIGHT_PACKS + group * V21_WEIGHT_GROUP_PACKS + 1],
                    lanes);
            v21_up_load_w_tail_lane:
                for (int lane = 0; lane < V21_MAC_GROUP - PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    w_tile[group_base + lane][tk] = lanes[lane];
                }
            }

        v21_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v21_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v21_up_mac_group:
                    for (int io = 0; io < V21_TILE_I; io += V21_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V21_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v21_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v21_up_load_acc_vec:
                        for (int jj = 0; jj < V21_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v21_up_mac_vec:
                        for (int jj = 0; jj < V21_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v21_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v21_up_store_acc_vec:
                        for (int jj = 0; jj < V21_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v21_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v21_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V21_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            v21_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    lanes[lane] =
                        gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                }
                gelu_stream.write(pack_packed_float16(lanes));
            }
        }
    }
}

static void v21_ffn_down_tile96_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V21_TILE_I];
    float w_tile[V21_TILE_O][V21_TILE_I];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_1P impl=LUTRAM latency=1
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=24 dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2

v21_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v21_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            unpack_packed_float16(b2[h_pack], lanes);
        v21_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }

v21_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V21_TILE_I) {
        int i_tile = i0 / V21_TILE_I;
    v21_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v21_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V21_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                unpack_packed_float16(gelu_stream.read(), lanes);
            v21_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

        v21_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V21_TILE_O) {
            int o_tile = o0 / V21_TILE_O;
            int packed_base =
                ((i_tile * V21_HIDDEN_TILES + o_tile) * V21_TILE_I)
                * V21_W2_WEIGHT_PACKS;
        v21_down_load_w_full:
            for (int word_offset = 0;
                 word_offset < V21_TILE_I * V21_W2_WEIGHT_GROUPS;
                 word_offset++) {
#pragma HLS PIPELINE II=1
                int tk = word_offset >> 2;
                int group = word_offset & 3;
                int group_base = group * V21_MAC_GROUP;
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                unpack_packed_float16(
                    w2[packed_base + tk * V21_W2_WEIGHT_PACKS + group * V21_WEIGHT_GROUP_PACKS],
                    lanes);
            v21_down_load_w_full_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    w_tile[group_base + lane][tk] = lanes[lane];
                }
            }

        v21_down_load_w_tail:
            for (int word_offset = 0;
                 word_offset < V21_TILE_I * V21_W2_WEIGHT_GROUPS;
                 word_offset++) {
#pragma HLS PIPELINE II=1
                int tk = word_offset >> 2;
                int group = word_offset & 3;
                int group_base = group * V21_MAC_GROUP + PACK_FLOATS;
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                unpack_packed_float16(
                    w2[packed_base + tk * V21_W2_WEIGHT_PACKS + group * V21_WEIGHT_GROUP_PACKS + 1],
                    lanes);
            v21_down_load_w_tail_lane:
                for (int lane = 0; lane < V21_MAC_GROUP - PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    w_tile[group_base + lane][tk] = lanes[lane];
                }
            }

        v21_down_k_chunk:
            for (int tk = 0; tk < V21_TILE_I; tk += V10_K_PAR) {
            v21_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v21_down_mac_group:
                    for (int oo = 0; oo < V21_TILE_O; oo += V21_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V21_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v21_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v21_down_load_acc_vec:
                        for (int jj = 0; jj < V21_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v21_down_mac_vec:
                        for (int jj = 0; jj < V21_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v21_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v21_down_store_acc_vec:
                        for (int jj = 0; jj < V21_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v21_projection_dataflow_tile96(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v21_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v21_ffn_up_tile96_producer(input_buf, w1, b1, gelu_stream);
    v21_ffn_down_tile96_consumer(gelu_stream, w2, b2, output_buf);
}
#endif

static void v16_ffn_up_tilemajor_directpack_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V13_TILE_I][V10_TILE_K];
    float acc[SEQ_LEN][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

v16_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
        int i_tile = i0 / V13_TILE_I;
    v16_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = b1[i0_pack];
            acc[s][0] = bits_to_float(word.range(31, 0));
            acc[s][1] = bits_to_float(word.range(63, 32));
            acc[s][2] = bits_to_float(word.range(95, 64));
            acc[s][3] = bits_to_float(word.range(127, 96));
            acc[s][4] = bits_to_float(word.range(159, 128));
            acc[s][5] = bits_to_float(word.range(191, 160));
            acc[s][6] = bits_to_float(word.range(223, 192));
            acc[s][7] = bits_to_float(word.range(255, 224));
            acc[s][8] = bits_to_float(word.range(287, 256));
            acc[s][9] = bits_to_float(word.range(319, 288));
            acc[s][10] = bits_to_float(word.range(351, 320));
            acc[s][11] = bits_to_float(word.range(383, 352));
            acc[s][12] = bits_to_float(word.range(415, 384));
            acc[s][13] = bits_to_float(word.range(447, 416));
            acc[s][14] = bits_to_float(word.range(479, 448));
            acc[s][15] = bits_to_float(word.range(511, 480));
        }

    v16_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
            int packed_base = i_tile * HIDDEN_SIZE + h0;
        v16_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = w1[packed_base + tk];
                w_tile[0][tk] = bits_to_float(word.range(31, 0));
                w_tile[1][tk] = bits_to_float(word.range(63, 32));
                w_tile[2][tk] = bits_to_float(word.range(95, 64));
                w_tile[3][tk] = bits_to_float(word.range(127, 96));
                w_tile[4][tk] = bits_to_float(word.range(159, 128));
                w_tile[5][tk] = bits_to_float(word.range(191, 160));
                w_tile[6][tk] = bits_to_float(word.range(223, 192));
                w_tile[7][tk] = bits_to_float(word.range(255, 224));
                w_tile[8][tk] = bits_to_float(word.range(287, 256));
                w_tile[9][tk] = bits_to_float(word.range(319, 288));
                w_tile[10][tk] = bits_to_float(word.range(351, 320));
                w_tile[11][tk] = bits_to_float(word.range(383, 352));
                w_tile[12][tk] = bits_to_float(word.range(415, 384));
                w_tile[13][tk] = bits_to_float(word.range(447, 416));
                w_tile[14][tk] = bits_to_float(word.range(479, 448));
                w_tile[15][tk] = bits_to_float(word.range(511, 480));
            }

        v16_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v16_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v16_up_mac_group:
                    for (int io = 0; io < V13_TILE_I; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v16_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v16_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v16_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v16_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v16_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v16_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = 0;
            word.range(31, 0) = float_to_bits(gelu_pwl_packed(acc[s][0]));
            word.range(63, 32) = float_to_bits(gelu_pwl_packed(acc[s][1]));
            word.range(95, 64) = float_to_bits(gelu_pwl_packed(acc[s][2]));
            word.range(127, 96) = float_to_bits(gelu_pwl_packed(acc[s][3]));
            word.range(159, 128) = float_to_bits(gelu_pwl_packed(acc[s][4]));
            word.range(191, 160) = float_to_bits(gelu_pwl_packed(acc[s][5]));
            word.range(223, 192) = float_to_bits(gelu_pwl_packed(acc[s][6]));
            word.range(255, 224) = float_to_bits(gelu_pwl_packed(acc[s][7]));
            word.range(287, 256) = float_to_bits(gelu_pwl_packed(acc[s][8]));
            word.range(319, 288) = float_to_bits(gelu_pwl_packed(acc[s][9]));
            word.range(351, 320) = float_to_bits(gelu_pwl_packed(acc[s][10]));
            word.range(383, 352) = float_to_bits(gelu_pwl_packed(acc[s][11]));
            word.range(415, 384) = float_to_bits(gelu_pwl_packed(acc[s][12]));
            word.range(447, 416) = float_to_bits(gelu_pwl_packed(acc[s][13]));
            word.range(479, 448) = float_to_bits(gelu_pwl_packed(acc[s][14]));
            word.range(511, 480) = float_to_bits(gelu_pwl_packed(acc[s][15]));
            gelu_stream.write(word);
        }
    }
}

static void v16_ffn_down_tilemajor_directpack_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V13_TILE_I];
    float w_tile[V13_TILE_O][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=2

v16_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v16_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            int h = h_pack << 4;
            packed_float16_t word = b2[h_pack];
            output_buf[s][h + 0] = bits_to_float(word.range(31, 0));
            output_buf[s][h + 1] = bits_to_float(word.range(63, 32));
            output_buf[s][h + 2] = bits_to_float(word.range(95, 64));
            output_buf[s][h + 3] = bits_to_float(word.range(127, 96));
            output_buf[s][h + 4] = bits_to_float(word.range(159, 128));
            output_buf[s][h + 5] = bits_to_float(word.range(191, 160));
            output_buf[s][h + 6] = bits_to_float(word.range(223, 192));
            output_buf[s][h + 7] = bits_to_float(word.range(255, 224));
            output_buf[s][h + 8] = bits_to_float(word.range(287, 256));
            output_buf[s][h + 9] = bits_to_float(word.range(319, 288));
            output_buf[s][h + 10] = bits_to_float(word.range(351, 320));
            output_buf[s][h + 11] = bits_to_float(word.range(383, 352));
            output_buf[s][h + 12] = bits_to_float(word.range(415, 384));
            output_buf[s][h + 13] = bits_to_float(word.range(447, 416));
            output_buf[s][h + 14] = bits_to_float(word.range(479, 448));
            output_buf[s][h + 15] = bits_to_float(word.range(511, 480));
        }
    }

v16_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i_tile = i0 / V13_TILE_I;
    v16_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = gelu_stream.read();
            activation[s][0] = bits_to_float(word.range(31, 0));
            activation[s][1] = bits_to_float(word.range(63, 32));
            activation[s][2] = bits_to_float(word.range(95, 64));
            activation[s][3] = bits_to_float(word.range(127, 96));
            activation[s][4] = bits_to_float(word.range(159, 128));
            activation[s][5] = bits_to_float(word.range(191, 160));
            activation[s][6] = bits_to_float(word.range(223, 192));
            activation[s][7] = bits_to_float(word.range(255, 224));
            activation[s][8] = bits_to_float(word.range(287, 256));
            activation[s][9] = bits_to_float(word.range(319, 288));
            activation[s][10] = bits_to_float(word.range(351, 320));
            activation[s][11] = bits_to_float(word.range(383, 352));
            activation[s][12] = bits_to_float(word.range(415, 384));
            activation[s][13] = bits_to_float(word.range(447, 416));
            activation[s][14] = bits_to_float(word.range(479, 448));
            activation[s][15] = bits_to_float(word.range(511, 480));
        }

    v16_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V13_TILE_O) {
            int o_tile = o0 / V13_TILE_O;
            int packed_base = ((i_tile * V15_HIDDEN_TILES + o_tile) * V13_TILE_I) * V15_W2_TILE_PACKS;
        v16_down_load_w_k:
            for (int tk = 0; tk < V13_TILE_I; tk++) {
#pragma HLS PIPELINE II=1
                int packed_index = packed_base + (tk << 1);
                packed_float16_t word0 = w2[packed_index];
                packed_float16_t word1 = w2[packed_index + 1];

                w_tile[0][tk] = bits_to_float(word0.range(31, 0));
                w_tile[1][tk] = bits_to_float(word0.range(63, 32));
                w_tile[2][tk] = bits_to_float(word0.range(95, 64));
                w_tile[3][tk] = bits_to_float(word0.range(127, 96));
                w_tile[4][tk] = bits_to_float(word0.range(159, 128));
                w_tile[5][tk] = bits_to_float(word0.range(191, 160));
                w_tile[6][tk] = bits_to_float(word0.range(223, 192));
                w_tile[7][tk] = bits_to_float(word0.range(255, 224));
                w_tile[8][tk] = bits_to_float(word0.range(287, 256));
                w_tile[9][tk] = bits_to_float(word0.range(319, 288));
                w_tile[10][tk] = bits_to_float(word0.range(351, 320));
                w_tile[11][tk] = bits_to_float(word0.range(383, 352));
                w_tile[12][tk] = bits_to_float(word0.range(415, 384));
                w_tile[13][tk] = bits_to_float(word0.range(447, 416));
                w_tile[14][tk] = bits_to_float(word0.range(479, 448));
                w_tile[15][tk] = bits_to_float(word0.range(511, 480));

                w_tile[16][tk] = bits_to_float(word1.range(31, 0));
                w_tile[17][tk] = bits_to_float(word1.range(63, 32));
                w_tile[18][tk] = bits_to_float(word1.range(95, 64));
                w_tile[19][tk] = bits_to_float(word1.range(127, 96));
                w_tile[20][tk] = bits_to_float(word1.range(159, 128));
                w_tile[21][tk] = bits_to_float(word1.range(191, 160));
                w_tile[22][tk] = bits_to_float(word1.range(223, 192));
                w_tile[23][tk] = bits_to_float(word1.range(255, 224));
                w_tile[24][tk] = bits_to_float(word1.range(287, 256));
                w_tile[25][tk] = bits_to_float(word1.range(319, 288));
                w_tile[26][tk] = bits_to_float(word1.range(351, 320));
                w_tile[27][tk] = bits_to_float(word1.range(383, 352));
                w_tile[28][tk] = bits_to_float(word1.range(415, 384));
                w_tile[29][tk] = bits_to_float(word1.range(447, 416));
                w_tile[30][tk] = bits_to_float(word1.range(479, 448));
                w_tile[31][tk] = bits_to_float(word1.range(511, 480));
            }

        v16_down_k_chunk:
            for (int tk = 0; tk < V13_TILE_I; tk += V10_K_PAR) {
            v16_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v16_down_mac_group:
                    for (int oo = 0; oo < V13_TILE_O; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v16_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v16_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v16_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v16_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v16_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v16_projection_dataflow_tilemajor_directpack(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v16_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v16_ffn_up_tilemajor_directpack_producer(input_buf, w1, b1, gelu_stream);
    v16_ffn_down_tilemajor_directpack_consumer(gelu_stream, w2, b2, output_buf);
}

static void v18_ffn_up_staged_tilemajor_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V13_TILE_I][V10_TILE_K];
    float acc[SEQ_LEN][V13_TILE_I];
    packed_float16_t w1_word_buf[V10_TILE_K];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_word_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

v18_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
        int i_tile = i0 / V13_TILE_I;
    v18_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = b1[i0_pack];
            acc[s][0] = bits_to_float(word.range(31, 0));
            acc[s][1] = bits_to_float(word.range(63, 32));
            acc[s][2] = bits_to_float(word.range(95, 64));
            acc[s][3] = bits_to_float(word.range(127, 96));
            acc[s][4] = bits_to_float(word.range(159, 128));
            acc[s][5] = bits_to_float(word.range(191, 160));
            acc[s][6] = bits_to_float(word.range(223, 192));
            acc[s][7] = bits_to_float(word.range(255, 224));
            acc[s][8] = bits_to_float(word.range(287, 256));
            acc[s][9] = bits_to_float(word.range(319, 288));
            acc[s][10] = bits_to_float(word.range(351, 320));
            acc[s][11] = bits_to_float(word.range(383, 352));
            acc[s][12] = bits_to_float(word.range(415, 384));
            acc[s][13] = bits_to_float(word.range(447, 416));
            acc[s][14] = bits_to_float(word.range(479, 448));
            acc[s][15] = bits_to_float(word.range(511, 480));
        }

    v18_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
            int packed_base = i_tile * HIDDEN_SIZE + h0;
        v18_up_load_w_words:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
#pragma HLS PIPELINE II=1
                w1_word_buf[tk] = w1[packed_base + tk];
            }

        v18_up_unpack_w_words:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = w1_word_buf[tk];
                w_tile[0][tk] = bits_to_float(word.range(31, 0));
                w_tile[1][tk] = bits_to_float(word.range(63, 32));
                w_tile[2][tk] = bits_to_float(word.range(95, 64));
                w_tile[3][tk] = bits_to_float(word.range(127, 96));
                w_tile[4][tk] = bits_to_float(word.range(159, 128));
                w_tile[5][tk] = bits_to_float(word.range(191, 160));
                w_tile[6][tk] = bits_to_float(word.range(223, 192));
                w_tile[7][tk] = bits_to_float(word.range(255, 224));
                w_tile[8][tk] = bits_to_float(word.range(287, 256));
                w_tile[9][tk] = bits_to_float(word.range(319, 288));
                w_tile[10][tk] = bits_to_float(word.range(351, 320));
                w_tile[11][tk] = bits_to_float(word.range(383, 352));
                w_tile[12][tk] = bits_to_float(word.range(415, 384));
                w_tile[13][tk] = bits_to_float(word.range(447, 416));
                w_tile[14][tk] = bits_to_float(word.range(479, 448));
                w_tile[15][tk] = bits_to_float(word.range(511, 480));
            }

        v18_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v18_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v18_up_mac_group:
                    for (int io = 0; io < V13_TILE_I; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v18_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v18_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v18_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v18_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v18_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v18_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = 0;
            word.range(31, 0) = float_to_bits(gelu_pwl_packed(acc[s][0]));
            word.range(63, 32) = float_to_bits(gelu_pwl_packed(acc[s][1]));
            word.range(95, 64) = float_to_bits(gelu_pwl_packed(acc[s][2]));
            word.range(127, 96) = float_to_bits(gelu_pwl_packed(acc[s][3]));
            word.range(159, 128) = float_to_bits(gelu_pwl_packed(acc[s][4]));
            word.range(191, 160) = float_to_bits(gelu_pwl_packed(acc[s][5]));
            word.range(223, 192) = float_to_bits(gelu_pwl_packed(acc[s][6]));
            word.range(255, 224) = float_to_bits(gelu_pwl_packed(acc[s][7]));
            word.range(287, 256) = float_to_bits(gelu_pwl_packed(acc[s][8]));
            word.range(319, 288) = float_to_bits(gelu_pwl_packed(acc[s][9]));
            word.range(351, 320) = float_to_bits(gelu_pwl_packed(acc[s][10]));
            word.range(383, 352) = float_to_bits(gelu_pwl_packed(acc[s][11]));
            word.range(415, 384) = float_to_bits(gelu_pwl_packed(acc[s][12]));
            word.range(447, 416) = float_to_bits(gelu_pwl_packed(acc[s][13]));
            word.range(479, 448) = float_to_bits(gelu_pwl_packed(acc[s][14]));
            word.range(511, 480) = float_to_bits(gelu_pwl_packed(acc[s][15]));
            gelu_stream.write(word);
        }
    }
}

static void v18_ffn_down_staged_tilemajor_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V13_TILE_I];
    float w_tile[V13_TILE_O][V13_TILE_I];
    packed_float16_t b2_word_buf[HIDDEN_PACKS];
    packed_float16_t w2_word_buf[V13_TILE_I * V15_W2_TILE_PACKS];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=b2_word_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w2_word_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=2

v18_down_preload_b2:
    for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
        b2_word_buf[h_pack] = b2[h_pack];
    }

v18_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v18_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            int h = h_pack << 4;
            packed_float16_t word = b2_word_buf[h_pack];
            output_buf[s][h + 0] = bits_to_float(word.range(31, 0));
            output_buf[s][h + 1] = bits_to_float(word.range(63, 32));
            output_buf[s][h + 2] = bits_to_float(word.range(95, 64));
            output_buf[s][h + 3] = bits_to_float(word.range(127, 96));
            output_buf[s][h + 4] = bits_to_float(word.range(159, 128));
            output_buf[s][h + 5] = bits_to_float(word.range(191, 160));
            output_buf[s][h + 6] = bits_to_float(word.range(223, 192));
            output_buf[s][h + 7] = bits_to_float(word.range(255, 224));
            output_buf[s][h + 8] = bits_to_float(word.range(287, 256));
            output_buf[s][h + 9] = bits_to_float(word.range(319, 288));
            output_buf[s][h + 10] = bits_to_float(word.range(351, 320));
            output_buf[s][h + 11] = bits_to_float(word.range(383, 352));
            output_buf[s][h + 12] = bits_to_float(word.range(415, 384));
            output_buf[s][h + 13] = bits_to_float(word.range(447, 416));
            output_buf[s][h + 14] = bits_to_float(word.range(479, 448));
            output_buf[s][h + 15] = bits_to_float(word.range(511, 480));
        }
    }

v18_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i_tile = i0 / V13_TILE_I;
    v18_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = gelu_stream.read();
            activation[s][0] = bits_to_float(word.range(31, 0));
            activation[s][1] = bits_to_float(word.range(63, 32));
            activation[s][2] = bits_to_float(word.range(95, 64));
            activation[s][3] = bits_to_float(word.range(127, 96));
            activation[s][4] = bits_to_float(word.range(159, 128));
            activation[s][5] = bits_to_float(word.range(191, 160));
            activation[s][6] = bits_to_float(word.range(223, 192));
            activation[s][7] = bits_to_float(word.range(255, 224));
            activation[s][8] = bits_to_float(word.range(287, 256));
            activation[s][9] = bits_to_float(word.range(319, 288));
            activation[s][10] = bits_to_float(word.range(351, 320));
            activation[s][11] = bits_to_float(word.range(383, 352));
            activation[s][12] = bits_to_float(word.range(415, 384));
            activation[s][13] = bits_to_float(word.range(447, 416));
            activation[s][14] = bits_to_float(word.range(479, 448));
            activation[s][15] = bits_to_float(word.range(511, 480));
        }

    v18_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V13_TILE_O) {
            int o_tile = o0 / V13_TILE_O;
            int packed_base = ((i_tile * V15_HIDDEN_TILES + o_tile) * V13_TILE_I) * V15_W2_TILE_PACKS;
        v18_down_load_w_words:
            for (int p = 0; p < V13_TILE_I * V15_W2_TILE_PACKS; p++) {
#pragma HLS PIPELINE II=1
                w2_word_buf[p] = w2[packed_base + p];
            }

        v18_down_unpack_w_pack0:
            for (int tk = 0; tk < V13_TILE_I; tk++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = w2_word_buf[tk << 1];
                w_tile[0][tk] = bits_to_float(word.range(31, 0));
                w_tile[1][tk] = bits_to_float(word.range(63, 32));
                w_tile[2][tk] = bits_to_float(word.range(95, 64));
                w_tile[3][tk] = bits_to_float(word.range(127, 96));
                w_tile[4][tk] = bits_to_float(word.range(159, 128));
                w_tile[5][tk] = bits_to_float(word.range(191, 160));
                w_tile[6][tk] = bits_to_float(word.range(223, 192));
                w_tile[7][tk] = bits_to_float(word.range(255, 224));
                w_tile[8][tk] = bits_to_float(word.range(287, 256));
                w_tile[9][tk] = bits_to_float(word.range(319, 288));
                w_tile[10][tk] = bits_to_float(word.range(351, 320));
                w_tile[11][tk] = bits_to_float(word.range(383, 352));
                w_tile[12][tk] = bits_to_float(word.range(415, 384));
                w_tile[13][tk] = bits_to_float(word.range(447, 416));
                w_tile[14][tk] = bits_to_float(word.range(479, 448));
                w_tile[15][tk] = bits_to_float(word.range(511, 480));
            }

        v18_down_unpack_w_pack1:
            for (int tk = 0; tk < V13_TILE_I; tk++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = w2_word_buf[(tk << 1) + 1];
                w_tile[16][tk] = bits_to_float(word.range(31, 0));
                w_tile[17][tk] = bits_to_float(word.range(63, 32));
                w_tile[18][tk] = bits_to_float(word.range(95, 64));
                w_tile[19][tk] = bits_to_float(word.range(127, 96));
                w_tile[20][tk] = bits_to_float(word.range(159, 128));
                w_tile[21][tk] = bits_to_float(word.range(191, 160));
                w_tile[22][tk] = bits_to_float(word.range(223, 192));
                w_tile[23][tk] = bits_to_float(word.range(255, 224));
                w_tile[24][tk] = bits_to_float(word.range(287, 256));
                w_tile[25][tk] = bits_to_float(word.range(319, 288));
                w_tile[26][tk] = bits_to_float(word.range(351, 320));
                w_tile[27][tk] = bits_to_float(word.range(383, 352));
                w_tile[28][tk] = bits_to_float(word.range(415, 384));
                w_tile[29][tk] = bits_to_float(word.range(447, 416));
                w_tile[30][tk] = bits_to_float(word.range(479, 448));
                w_tile[31][tk] = bits_to_float(word.range(511, 480));
            }

        v18_down_k_chunk:
            for (int tk = 0; tk < V13_TILE_I; tk += V10_K_PAR) {
            v18_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v18_down_mac_group:
                    for (int oo = 0; oo < V13_TILE_O; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v18_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v18_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v18_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v18_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v18_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v18_projection_dataflow_staged_tilemajor(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v18_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v18_ffn_up_staged_tilemajor_producer(input_buf, w1, b1, gelu_stream);
    v18_ffn_down_staged_tilemajor_consumer(gelu_stream, w2, b2, output_buf);
}

static void v14_ffn_up_tile32_mac8_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V10_TILE_OUT][V10_TILE_K];
    float acc[SEQ_LEN][V10_TILE_OUT];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

v14_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V10_TILE_OUT) {
    v14_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v14_up_init_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = b1[i0 / PACK_FLOATS + i_pack];
            v14_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v14_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
        v14_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
            v14_up_load_w_pack:
                for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = ((h0 + tk) * INTERMEDIATE_SIZE + i0) / PACK_FLOATS + i_pack;
                    packed_float16_t word = w1[packed_index];
                v14_up_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[i_pack * PACK_FLOATS + lane][tk] = get_packed_float(word, lane);
                    }
                }
            }

        v14_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v14_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v14_up_mac_group:
                    for (int io = 0; io < V10_TILE_OUT; io += V14_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V14_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v14_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v14_up_load_acc_vec:
                        for (int jj = 0; jj < V14_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v14_up_mac_vec:
                        for (int jj = 0; jj < V14_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v14_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v12_dot16_tree_timing(x_vec, w_vec);
                        }

                    v14_up_store_acc_vec:
                        for (int jj = 0; jj < V14_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v14_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v14_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = 0;
            v14_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    float value = gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                    set_packed_float(&word, lane, value);
                }
                gelu_stream.write(word);
            }
        }
    }
}

static void v14_ffn_down_tile32_mac8_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V10_TILE_OUT];
    float w_tile[V10_TILE_OUT][V10_TILE_OUT];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2

v14_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v14_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
            packed_float16_t word = b2[h_pack];
        v14_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
            }
        }
    }

v14_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V10_TILE_OUT) {
    v14_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v14_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V10_TILE_OUT / PACK_FLOATS; i_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = gelu_stream.read();
            v14_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v14_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V10_TILE_OUT) {
        v14_down_load_w_k:
            for (int tk = 0; tk < V10_TILE_OUT; tk++) {
            v14_down_load_w_pack:
                for (int o_pack = 0; o_pack < V10_TILE_OUT / PACK_FLOATS; o_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = ((i0 + tk) * HIDDEN_SIZE + o0) / PACK_FLOATS + o_pack;
                    packed_float16_t word = w2[packed_index];
                v14_down_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[o_pack * PACK_FLOATS + lane][tk] = get_packed_float(word, lane);
                    }
                }
            }

        v14_down_k_chunk:
            for (int tk = 0; tk < V10_TILE_OUT; tk += V10_K_PAR) {
            v14_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v14_down_mac_group:
                    for (int oo = 0; oo < V10_TILE_OUT; oo += V14_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V14_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v14_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v14_down_load_acc_vec:
                        for (int jj = 0; jj < V14_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v14_down_mac_vec:
                        for (int jj = 0; jj < V14_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v14_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v12_dot16_tree_timing(x_vec, w_vec);
                        }

                    v14_down_store_acc_vec:
                        for (int jj = 0; jj < V14_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v14_projection_dataflow_slr_fit(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v14_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v14_ffn_up_tile32_mac8_producer(input_buf, w1, b1, gelu_stream);
    v14_ffn_down_tile32_mac8_consumer(gelu_stream, w2, b2, output_buf);
}

extern "C" {
void bert_ffn_kernel_v7_packed512_t64_i128_h32_o16(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_tile[TILE_T_V5][HIDDEN_SIZE];
    static float out_acc[TILE_T_V5][HIDDEN_SIZE];
    static float w1_tile[TILE_I][HIDDEN_SIZE];
    static float gelu_tile[TILE_T_V5][TILE_I];
    static float w2_tile[TILE_I][TILE_O];
    static float b2_local[HIDDEN_SIZE];
    float b1_tile[TILE_I];
    float out_vec[TILE_O];

#pragma HLS BIND_STORAGE variable=x_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=out_acc type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=gelu_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w2_tile type=RAM_2P impl=BRAM latency=2

#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w1_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=gelu_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w2_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=b2_local cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=b1_tile cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1

    int t0;
    int tt;
    int h;
    int h_pack;
    int i0;
    int ii;
    int ii_pack;
    int o0;
    int oo;
    int oo_pack;
    int lane;

v7_load_b2_pack:
    for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
        packed_float16_t word = b2[h_pack];
    v7_load_b2_lane:
        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
            b2_local[h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
        }
    }

v7_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T_V5) {
    v7_load_input_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v7_load_input_pack:
            for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = input_hidden[((t0 + tt) * HIDDEN_SIZE) / PACK_FLOATS + h_pack];
            v7_unpack_input_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    x_tile[tt][h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v7_init_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v7_init_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                out_acc[tt][h] = b2_local[h];
            }
        }

    v7_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v7_load_b1_pack:
            for (ii_pack = 0; ii_pack < TILE_I / PACK_FLOATS; ii_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = b1[i0 / PACK_FLOATS + ii_pack];
            v7_load_b1_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    b1_tile[ii_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }

        v7_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v7_load_w1_tile_pack:
                for (ii_pack = 0; ii_pack < TILE_I / PACK_FLOATS; ii_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = (h * INTERMEDIATE_SIZE + i0) / PACK_FLOATS + ii_pack;
                    packed_float16_t word = w1[packed_index];
                v7_unpack_w1_lane:
                    for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w1_tile[ii_pack * PACK_FLOATS + lane][h] = get_packed_float(word, lane);
                    }
                }
            }

        v7_compute_gelu_t:
            for (tt = 0; tt < TILE_T_V5; tt++) {
            v7_compute_gelu_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    float acc = b1_tile[ii];
                v7_compute_w1_reduce_h:
                    for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                        acc += x_tile[tt][h] * w1_tile[ii][h];
                    }
                    gelu_tile[tt][ii] = gelu_pwl_packed(acc);
                }
            }

        v7_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v7_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v7_load_w2_tile_pack:
                    for (oo_pack = 0; oo_pack < TILE_O / PACK_FLOATS; oo_pack++) {
#pragma HLS PIPELINE II=1
                        int packed_index = ((i0 + ii) * HIDDEN_SIZE + o0) / PACK_FLOATS + oo_pack;
                        packed_float16_t word = w2[packed_index];
                    v7_unpack_w2_lane:
                        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                            w2_tile[ii][oo_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                        }
                    }
                }

            v7_update_output_t:
                for (tt = 0; tt < TILE_T_V5; tt++) {
                v7_load_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_vec[oo] = out_acc[tt][o0 + oo];
                    }

                v7_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                        float g = gelu_tile[tt][ii];
                    v7_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_vec[oo] += g * w2_tile[ii][oo];
                        }
                    }

                v7_store_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_acc[tt][o0 + oo] = out_vec[oo];
                    }
                }
            }
        }

    v7_store_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v7_store_output_pack:
            for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = 0;
            v7_pack_output_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    set_packed_float(&word, lane, out_acc[tt][h_pack * PACK_FLOATS + lane]);
                }
                output_hidden[((t0 + tt) * HIDDEN_SIZE) / PACK_FLOATS + h_pack] = word;
            }
        }
    }
}

void bert_ffn_kernel_v8_packed512_w1word_t64_i128_h32_o16(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_tile[TILE_T_V5][HIDDEN_SIZE];
    static float out_acc[TILE_T_V5][HIDDEN_SIZE];
    static packed_float16_t w1_word_tile[TILE_I_PACKS][HIDDEN_SIZE];
    static float gelu_tile[TILE_T_V5][TILE_I];
    static float w2_tile[TILE_I][TILE_O];
    static float b2_local[HIDDEN_SIZE];
    float b1_tile[TILE_I];
    float out_vec[TILE_O];

#pragma HLS BIND_STORAGE variable=x_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=out_acc type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_word_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=gelu_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w2_tile type=RAM_2P impl=BRAM latency=2

#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w1_word_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=gelu_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w2_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=b2_local cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=b1_tile cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1

    int t0;
    int tt;
    int h;
    int h_pack;
    int i0;
    int ii;
    int ii_pack;
    int o0;
    int oo;
    int oo_pack;
    int lane;

v8_load_b2_pack:
    for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
        packed_float16_t word = b2[h_pack];
    v8_load_b2_lane:
        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
            b2_local[h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
        }
    }

v8_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T_V5) {
    v8_load_input_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v8_load_input_pack:
            for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = input_hidden[((t0 + tt) * HIDDEN_SIZE) / PACK_FLOATS + h_pack];
            v8_unpack_input_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    x_tile[tt][h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v8_init_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v8_init_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                out_acc[tt][h] = b2_local[h];
            }
        }

    v8_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v8_load_b1_pack:
            for (ii_pack = 0; ii_pack < TILE_I_PACKS; ii_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = b1[i0 / PACK_FLOATS + ii_pack];
            v8_load_b1_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    b1_tile[ii_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }

        v8_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v8_load_w1_tile_pack:
                for (ii_pack = 0; ii_pack < TILE_I_PACKS; ii_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = (h * INTERMEDIATE_SIZE + i0) / PACK_FLOATS + ii_pack;
                    w1_word_tile[ii_pack][h] = w1[packed_index];
                }
            }

        v8_compute_gelu_t:
            for (tt = 0; tt < TILE_T_V5; tt++) {
            v8_compute_gelu_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    int w1_pack = ii / PACK_FLOATS;
                    int w1_lane = ii - w1_pack * PACK_FLOATS;
                    float acc = b1_tile[ii];
                v8_compute_w1_reduce_h:
                    for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                        float w1_value = get_packed_float(w1_word_tile[w1_pack][h], w1_lane);
                        acc += x_tile[tt][h] * w1_value;
                    }
                    gelu_tile[tt][ii] = gelu_pwl_packed(acc);
                }
            }

        v8_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v8_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v8_load_w2_tile_pack:
                    for (oo_pack = 0; oo_pack < TILE_O / PACK_FLOATS; oo_pack++) {
#pragma HLS PIPELINE II=1
                        int packed_index = ((i0 + ii) * HIDDEN_SIZE + o0) / PACK_FLOATS + oo_pack;
                        packed_float16_t word = w2[packed_index];
                    v8_unpack_w2_lane:
                        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                            w2_tile[ii][oo_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                        }
                    }
                }

            v8_update_output_t:
                for (tt = 0; tt < TILE_T_V5; tt++) {
                v8_load_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_vec[oo] = out_acc[tt][o0 + oo];
                    }

                v8_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                        float g = gelu_tile[tt][ii];
                    v8_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_vec[oo] += g * w2_tile[ii][oo];
                        }
                    }

                v8_store_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_acc[tt][o0 + oo] = out_vec[oo];
                    }
                }
            }
        }

    v8_store_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v8_store_output_pack:
            for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = 0;
            v8_pack_output_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    set_packed_float(&word, lane, out_acc[tt][h_pack * PACK_FLOATS + lane]);
                }
                output_hidden[((t0 + tt) * HIDDEN_SIZE) / PACK_FLOATS + h_pack] = word;
            }
        }
    }
}

void bert_ffn_kernel_v9_packed512_w1vec_outpart_t64_i128_h32_o16(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_tile[TILE_T_V5][HIDDEN_SIZE];
    static float out_acc[TILE_T_V5][HIDDEN_SIZE];
    static packed_float16_t w1_word_tile[TILE_I_PACKS][HIDDEN_SIZE];
    static float w1_vec[HIDDEN_SIZE];
    static float gelu_tile[TILE_T_V5][TILE_I];
    static float w2_tile[TILE_I][TILE_O];
    static float b2_local[HIDDEN_SIZE];
    float b1_tile[TILE_I];
    float out_vec[TILE_O];
    float out_part[OUT_PART_BANKS][TILE_O];

#pragma HLS BIND_STORAGE variable=x_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=out_acc type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_word_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_vec type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=gelu_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w2_tile type=RAM_2P impl=BRAM latency=2

#pragma HLS ARRAY_PARTITION variable=x_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w1_word_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=w1_vec cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=gelu_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w2_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=b2_local cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=b1_tile cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=out_part complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_part cyclic factor=16 dim=2

    int t0;
    int tt;
    int h;
    int h0;
    int h_pack;
    int i0;
    int ii;
    int ii_pack;
    int o0;
    int oo;
    int oo_pack;
    int oo_base;
    int lane;
    int u;
    int p;
    int bank;

v9_load_b2_pack:
    for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
        packed_float16_t word = b2[h_pack];
    v9_load_b2_lane:
        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
            b2_local[h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
        }
    }

v9_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T_V5) {
    v9_load_input_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v9_load_input_pack:
            for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = input_hidden[((t0 + tt) * HIDDEN_SIZE) / PACK_FLOATS + h_pack];
            v9_unpack_input_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    x_tile[tt][h_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }
        }

    v9_init_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v9_init_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                out_acc[tt][h] = b2_local[h];
            }
        }

    v9_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v9_load_b1_pack:
            for (ii_pack = 0; ii_pack < TILE_I_PACKS; ii_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = b1[i0 / PACK_FLOATS + ii_pack];
            v9_load_b1_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    b1_tile[ii_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                }
            }

        v9_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v9_load_w1_tile_pack:
                for (ii_pack = 0; ii_pack < TILE_I_PACKS; ii_pack++) {
#pragma HLS PIPELINE II=1
                    int packed_index = (h * INTERMEDIATE_SIZE + i0) / PACK_FLOATS + ii_pack;
                    w1_word_tile[ii_pack][h] = w1[packed_index];
                }
            }

        v9_compute_gelu_i:
            for (ii = 0; ii < TILE_I; ii++) {
                int w1_pack = ii / PACK_FLOATS;
                int w1_lane = ii - w1_pack * PACK_FLOATS;

            v9_unpack_w1_vec_h0:
                for (h0 = 0; h0 < HIDDEN_SIZE; h0 += 32) {
#pragma HLS PIPELINE II=1
                v9_unpack_w1_vec_u:
                    for (u = 0; u < 32; u++) {
#pragma HLS UNROLL
                        w1_vec[h0 + u] = get_packed_float(w1_word_tile[w1_pack][h0 + u], w1_lane);
                    }
                }

            v9_compute_gelu_t:
                for (tt = 0; tt < TILE_T_V5; tt++) {
#pragma HLS PIPELINE II=1
                    float acc = b1_tile[ii];
                v9_compute_w1_reduce_h:
                    for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                        acc += x_tile[tt][h] * w1_vec[h];
                    }
                    gelu_tile[tt][ii] = gelu_pwl_packed(acc);
                }
            }

        v9_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v9_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v9_load_w2_tile_pack:
                    for (oo_pack = 0; oo_pack < TILE_O / PACK_FLOATS; oo_pack++) {
#pragma HLS PIPELINE II=1
                        int packed_index = ((i0 + ii) * HIDDEN_SIZE + o0) / PACK_FLOATS + oo_pack;
                        packed_float16_t word = w2[packed_index];
                    v9_unpack_w2_lane:
                        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                            w2_tile[ii][oo_pack * PACK_FLOATS + lane] = get_packed_float(word, lane);
                        }
                    }
                }

            v9_update_output_t:
                for (tt = 0; tt < TILE_T_V5; tt++) {
                v9_load_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_vec[oo] = out_acc[tt][o0 + oo];
                    }

                v9_out_part_init_o:
                    for (oo_base = 0; oo_base < TILE_O; oo_base += PACK_FLOATS) {
#pragma HLS PIPELINE II=1
                    v9_out_part_init_p:
                        for (p = 0; p < OUT_PART_BANKS; p++) {
#pragma HLS UNROLL
                        v9_out_part_init_lane:
                            for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                                out_part[p][oo_base + lane] = 0.0f;
                            }
                        }
                    }

                v9_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=out_part inter false
                        bank = ii & (OUT_PART_BANKS - 1);
                        float g = gelu_tile[tt][ii];
                    v9_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_part[bank][oo] += g * w2_tile[ii][oo];
                        }
                    }

                v9_sum_out_vec_o:
                    for (oo_base = 0; oo_base < TILE_O; oo_base += PACK_FLOATS) {
#pragma HLS PIPELINE II=1
                    v9_sum_out_vec_lane:
                        for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                            float acc = out_vec[oo_base + lane];
                        v9_sum_out_vec_p:
                            for (p = 0; p < OUT_PART_BANKS; p++) {
#pragma HLS UNROLL
                                acc += out_part[p][oo_base + lane];
                            }
                            out_vec[oo_base + lane] = acc;
                        }
                    }

                v9_store_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_acc[tt][o0 + oo] = out_vec[oo];
                    }
                }
            }
        }

    v9_store_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v9_store_output_pack:
            for (h_pack = 0; h_pack < HIDDEN_SIZE / PACK_FLOATS; h_pack++) {
#pragma HLS PIPELINE II=1
                packed_float16_t word = 0;
            v9_pack_output_lane:
                for (lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    set_packed_float(&word, lane, out_acc[tt][h_pack * PACK_FLOATS + lane]);
                }
                output_hidden[((t0 + tt) * HIDDEN_SIZE) / PACK_FLOATS + h_pack] = word;
            }
        }
    }
}

void bert_ffn_kernel_v10_stream_tile32_k64_p16_m16(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v10_projection_dataflow(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_up_gelu_v11_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v10_ffn_up_tile_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v11_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_ffn_down_tile_consumer(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v11_split_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v10_projection_dataflow(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_up_gelu_v12_uram_bindop_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v12_ffn_up_tile_producer_timing(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v12_uram_bindop_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v12_ffn_down_tile_consumer_timing(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v12_uram_bindop_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v12_projection_dataflow_timing(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_up_gelu_v13_tile16_uram_bindop_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v13_ffn_up_tile16_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v13_tile16_uram_bindop_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v13_ffn_down_tile16_consumer(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v13_tile16_uram_bindop_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v13_projection_dataflow_tile16(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_up_gelu_v15_tilemajor_uram_bindop_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v15_ffn_up_tilemajor_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v15_tilemajor_uram_bindop_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v15_ffn_down_tilemajor_consumer(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v15_tilemajor_uram_bindop_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v15_projection_dataflow_tilemajor(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

#ifdef FFN_V20_COMPUTE_MARGIN
void bert_ffn_up_gelu_v20_fmul5_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v15_ffn_up_tilemajor_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v20_fmul5_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v15_ffn_down_tilemajor_consumer(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v20_fmul5_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v15_projection_dataflow_tilemajor(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}
#endif

#ifdef FFN_V21_DOT_PIPELINE
void bert_ffn_up_gelu_v21_dotpipe_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v15_ffn_up_tilemajor_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v21_dotpipe_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v15_ffn_down_tilemajor_consumer(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v21_dotpipe_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v15_projection_dataflow_tilemajor(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}
#endif

void bert_ffn_up_gelu_v16_tilemajor_directpack_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v16_ffn_up_tilemajor_directpack_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v16_tilemajor_directpack_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v16_ffn_down_tilemajor_directpack_consumer(gelu_stream, w2, b2, output_buf);
    v16_store_output_matrix_direct(output_buf, output_hidden);
}

void bert_ffn_kernel_v16_tilemajor_directpack_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v16_projection_dataflow_tilemajor_directpack(input_buf, w1, b1, w2, b2, output_buf);
    v16_store_output_matrix_direct(output_buf, output_hidden);
}

void bert_ffn_up_gelu_v18_staged_tilemajor_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v18_ffn_up_staged_tilemajor_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v18_staged_tilemajor_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v18_ffn_down_staged_tilemajor_consumer(gelu_stream, w2, b2, output_buf);
    v18_store_output_matrix_staged(output_buf, output_hidden);
}

void bert_ffn_kernel_v18_staged_tilemajor_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v18_projection_dataflow_staged_tilemajor(input_buf, w1, b1, w2, b2, output_buf);
    v18_store_output_matrix_staged(output_buf, output_hidden);
}

void bert_ffn_up_gelu_v14_slr2_tile32_mac8_uram_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v14_ffn_up_tile32_mac8_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v14_slr3_tile32_mac8_uram_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v14_ffn_down_tile32_mac8_consumer(gelu_stream, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v14_slr_split_tile32_mac8_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v14_projection_dataflow_slr_fit(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}
}
