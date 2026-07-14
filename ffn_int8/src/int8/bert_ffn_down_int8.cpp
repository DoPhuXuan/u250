#include "ffn_int8_common.h"

static void ffn_down_i8_initialize(
    const int *b2,
    const float *w2_scale,
    ffn_acc_t output_accumulator[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE],
    float scale_buffer[FFN_I8_HIDDEN_SIZE]) {
#pragma HLS INLINE off

ffn_down_i8_load_scale:
    for (int hidden = 0; hidden < FFN_I8_HIDDEN_SIZE; ++hidden) {
#pragma HLS PIPELINE II=1
        scale_buffer[hidden] = w2_scale[hidden];
    }

ffn_down_i8_bias_tile:
    for (int hidden_base = 0; hidden_base < FFN_I8_HIDDEN_SIZE; hidden_base += FFN_I8_OUT_PAR) {
        ffn_acc_t bias_tile[FFN_I8_OUT_PAR];
#pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
    ffn_down_i8_load_bias:
        for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS PIPELINE II=1
            bias_tile[output_lane] = (ffn_acc_t)b2[hidden_base + output_lane];
        }
    ffn_down_i8_initialize_sequence:
        for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
        ffn_down_i8_initialize_lane:
            for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS UNROLL
                output_accumulator[sequence][hidden_base + output_lane] = bias_tile[output_lane];
            }
        }
    }
}

static void ffn_down_i8_prefetch_and_mac_tile(
    const packed_i8x64_t *w2,
    int output_tile,
    int current_k_subtile,
    int next_global_k_tile,
    int output_base,
    ffn_i8_t activation_tile[FFN_I8_SEQ_LEN][FFN_I8_STREAM_TILE],
    ffn_acc_t output_accumulator[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE],
    ffn_i8_t weight_tiles[2][FFN_I8_OUT_PAR][FFN_I8_K_PAR]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=3
    int current_bank = current_k_subtile & 1;
    int next_bank = current_bank ^ 1;
    int next_weight_base =
        (output_tile * (FFN_I8_INTERMEDIATE_SIZE / FFN_I8_K_PAR) + next_global_k_tile) *
        FFN_I8_WEIGHT_WORDS_PER_TILE;

ffn_down_i8_sequence_mac:
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
        if (sequence < FFN_I8_WEIGHT_WORDS_PER_TILE) {
            packed_i8x64_t next_weight_word = w2[next_weight_base + sequence];
        ffn_down_i8_prefetch_weight_lane:
            for (int byte_lane = 0; byte_lane < FFN_I8_I8_PER_WORD; ++byte_lane) {
#pragma HLS UNROLL
                int linear = sequence * FFN_I8_I8_PER_WORD + byte_lane;
                int k_lane = linear / FFN_I8_OUT_PAR;
                int output_lane = linear % FFN_I8_OUT_PAR;
                weight_tiles[next_bank][output_lane][k_lane] =
                    ffn_i8_unpack_byte(next_weight_word, byte_lane);
            }
        }
        ffn_i8_t input_vector[FFN_I8_K_PAR];
        ffn_acc_t next_accumulator[FFN_I8_OUT_PAR];
#pragma HLS ARRAY_PARTITION variable=input_vector complete dim=1
#pragma HLS ARRAY_PARTITION variable=next_accumulator complete dim=1
    ffn_down_i8_read_input_vector:
        for (int k_lane = 0; k_lane < FFN_I8_K_PAR; ++k_lane) {
#pragma HLS UNROLL
            input_vector[k_lane] =
                activation_tile[sequence][current_k_subtile * FFN_I8_K_PAR + k_lane];
        }
    ffn_down_i8_output_mac_pair:
        for (int output_pair = 0; output_pair < FFN_I8_OUT_PAR / 2; ++output_pair) {
#pragma HLS UNROLL
            int first_lane = 2 * output_pair;
            int second_lane = first_lane + 1;
            ffn_i8_t selected_weight_first[FFN_I8_K_PAR];
            ffn_i8_t selected_weight_second[FFN_I8_K_PAR];
#pragma HLS ARRAY_PARTITION variable=selected_weight_first complete dim=1
#pragma HLS ARRAY_PARTITION variable=selected_weight_second complete dim=1
        ffn_down_i8_select_weight_bank:
            for (int k_lane = 0; k_lane < FFN_I8_K_PAR; ++k_lane) {
#pragma HLS UNROLL
                if (current_bank == 0) {
                    selected_weight_first[k_lane] = weight_tiles[0][first_lane][k_lane];
                    selected_weight_second[k_lane] = weight_tiles[0][second_lane][k_lane];
                } else {
                    selected_weight_first[k_lane] = weight_tiles[1][first_lane][k_lane];
                    selected_weight_second[k_lane] = weight_tiles[1][second_lane][k_lane];
                }
            }
            ffn_i8_dot_pair_t dot = ffn_i8_dot_pair(
                input_vector,
                selected_weight_first,
                selected_weight_second);
            next_accumulator[first_lane] =
                output_accumulator[sequence][output_base + first_lane] + dot.first;
            next_accumulator[second_lane] =
                output_accumulator[sequence][output_base + second_lane] + dot.second;
        }
    ffn_down_i8_write_accumulator:
        for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS UNROLL
            output_accumulator[sequence][output_base + output_lane] =
                next_accumulator[output_lane];
        }
    }
}

static void ffn_down_i8_compute(
    packed_i8_stream_t &gelu_i8_stream,
    const packed_i8x64_t *w2,
    ffn_acc_t output_accumulator[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE]) {
#pragma HLS INLINE off
    ffn_i8_t activation_tile[FFN_I8_SEQ_LEN][FFN_I8_STREAM_TILE];
    ffn_i8_t weight_tiles[2][FFN_I8_OUT_PAR][FFN_I8_K_PAR];
#pragma HLS ARRAY_PARTITION variable=activation_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=3

ffn_down_i8_stream_tile:
    for (int stream_base = 0; stream_base < FFN_I8_INTERMEDIATE_SIZE; stream_base += FFN_I8_STREAM_TILE) {
    ffn_down_i8_read_stream:
        for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
            packed_i8x64_t word = gelu_i8_stream.read();
        ffn_down_i8_unpack_stream_lane:
            for (int lane = 0; lane < FFN_I8_STREAM_TILE; ++lane) {
#pragma HLS UNROLL
                activation_tile[sequence][lane] = ffn_i8_unpack_byte(word, lane);
            }
        }

    ffn_down_i8_output_tile:
        for (int output_base = 0; output_base < FFN_I8_HIDDEN_SIZE; output_base += FFN_I8_OUT_PAR) {
            int output_tile = output_base / FFN_I8_OUT_PAR;
            ffn_i8_load_weight_tile(
                w2,
                output_tile,
                stream_base / FFN_I8_K_PAR,
                FFN_I8_INTERMEDIATE_SIZE / FFN_I8_K_PAR,
                weight_tiles[0]);
        ffn_down_i8_k_subtile_overlap:
            for (int k_subtile = 0;
                 k_subtile < FFN_I8_STREAM_TILE / FFN_I8_K_PAR;
                 ++k_subtile) {
                int next_k_subtile = k_subtile + 1;
                if (next_k_subtile == FFN_I8_STREAM_TILE / FFN_I8_K_PAR) {
                    next_k_subtile = 0;
                }
                int next_global_k_tile =
                    stream_base / FFN_I8_K_PAR + next_k_subtile;
                ffn_down_i8_prefetch_and_mac_tile(
                    w2, output_tile, k_subtile, next_global_k_tile, output_base,
                    activation_tile, output_accumulator, weight_tiles);
            }
        }
    }
}

static void ffn_down_i8_store_output(
    ffn_acc_t output_accumulator[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE],
    float scale_buffer[FFN_I8_HIDDEN_SIZE],
    float gelu_scale,
    packed_f32x16_t *output_hidden) {
#pragma HLS INLINE off

ffn_down_i8_store_sequence:
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
    ffn_down_i8_store_pack:
        for (int hidden_pack = 0; hidden_pack < FFN_I8_HIDDEN_SIZE / FFN_I8_F32_PER_WORD; ++hidden_pack) {
#pragma HLS PIPELINE II=1
            packed_f32x16_t word = 0;
        ffn_down_i8_store_lane:
            for (int lane = 0; lane < FFN_I8_F32_PER_WORD; ++lane) {
#pragma HLS UNROLL
                int hidden = hidden_pack * FFN_I8_F32_PER_WORD + lane;
                float value = (float)output_accumulator[sequence][hidden] *
                              (gelu_scale * scale_buffer[hidden]);
                word.range(lane * 32 + 31, lane * 32) = ffn_i8_float_to_bits(value);
            }
            output_hidden[sequence * (FFN_I8_HIDDEN_SIZE / FFN_I8_F32_PER_WORD) + hidden_pack] = word;
        }
    }
}

extern "C" {

void ffn_down_i8_v1(
    packed_i8_stream_t &gelu_i8_stream,
    const packed_i8x64_t *w2,
    const int *b2,
    const float *w2_scale,
    float gelu_scale,
    packed_f32x16_t *output_hidden) {
#pragma HLS INTERFACE axis port=gelu_i8_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 depth=36864
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 depth=768
#pragma HLS INTERFACE m_axi port=w2_scale offset=slave bundle=gmem_w2 depth=768
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out depth=6144
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=w2_scale bundle=control
#pragma HLS INTERFACE s_axilite port=gelu_scale bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static ffn_acc_t output_accumulator[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE];
    static float scale_buffer[FFN_I8_HIDDEN_SIZE];
#ifdef FFN_I8_USE_BRAM
#pragma HLS BIND_STORAGE variable=output_accumulator type=RAM_2P impl=BRAM latency=2
#else
#pragma HLS BIND_STORAGE variable=output_accumulator type=RAM_2P impl=URAM latency=2
#endif
#if FFN_I8_OUT_PAR == 16
#pragma HLS ARRAY_PARTITION variable=output_accumulator cyclic factor=16 dim=2
#else
#pragma HLS ARRAY_PARTITION variable=output_accumulator cyclic factor=32 dim=2
#endif
#pragma HLS BIND_STORAGE variable=scale_buffer type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=scale_buffer cyclic factor=16 dim=1

    ffn_down_i8_initialize(b2, w2_scale, output_accumulator, scale_buffer);
    ffn_down_i8_compute(gelu_i8_stream, w2, output_accumulator);
    ffn_down_i8_store_output(output_accumulator, scale_buffer, gelu_scale, output_hidden);
}

}
