#include "ffn_int8_common.h"

static void ffn_up_i8_load_input(
    packed_f32_stream_t &attn_mid_stream,
    float input_inverse_scale,
    ffn_i8_t input_buffer[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE]) {
#pragma HLS INLINE off

ffn_up_i8_load_input_sequence:
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
    ffn_up_i8_load_input_word:
        for (int pack = 0; pack < FFN_I8_HIDDEN_SIZE / FFN_I8_F32_PER_WORD; ++pack) {
#pragma HLS PIPELINE II=1
            packed_f32x16_t word = attn_mid_stream.read();
        ffn_up_i8_load_input_lane:
            for (int lane = 0; lane < FFN_I8_F32_PER_WORD; ++lane) {
#pragma HLS UNROLL
                float value = ffn_i8_bits_to_float(word.range(lane * 32 + 31, lane * 32));
                input_buffer[sequence][pack * FFN_I8_F32_PER_WORD + lane] =
                    ffn_i8_quantize(value, input_inverse_scale);
            }
        }
    }
}

static void ffn_up_i8_prefetch_and_mac_tile(
    const packed_i8x64_t *w1,
    int output_tile,
    int current_k_tile,
    int next_k_tile,
    ffn_i8_t input_buffer[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE],
    ffn_acc_t accumulator[FFN_I8_SEQ_LEN][FFN_I8_OUT_PAR],
    ffn_i8_t weight_tiles[2][FFN_I8_OUT_PAR][FFN_I8_K_PAR]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=3
    int current_bank = current_k_tile & 1;
    int next_bank = current_bank ^ 1;
    int next_weight_base =
        (output_tile * (FFN_I8_HIDDEN_SIZE / FFN_I8_K_PAR) + next_k_tile) *
        FFN_I8_WEIGHT_WORDS_PER_TILE;

ffn_up_i8_sequence_mac:
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
        if (sequence < FFN_I8_WEIGHT_WORDS_PER_TILE) {
            packed_i8x64_t next_weight_word = w1[next_weight_base + sequence];
        ffn_up_i8_prefetch_weight_lane:
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
    ffn_up_i8_read_input_vector:
        for (int k_lane = 0; k_lane < FFN_I8_K_PAR; ++k_lane) {
#pragma HLS UNROLL
            input_vector[k_lane] =
                input_buffer[sequence][current_k_tile * FFN_I8_K_PAR + k_lane];
        }
    ffn_up_i8_output_mac_pair:
        for (int output_pair = 0; output_pair < FFN_I8_OUT_PAR / 2; ++output_pair) {
#pragma HLS UNROLL
            int first_lane = 2 * output_pair;
            int second_lane = first_lane + 1;
            ffn_i8_t selected_weight_first[FFN_I8_K_PAR];
            ffn_i8_t selected_weight_second[FFN_I8_K_PAR];
#pragma HLS ARRAY_PARTITION variable=selected_weight_first complete dim=1
#pragma HLS ARRAY_PARTITION variable=selected_weight_second complete dim=1
        ffn_up_i8_select_weight_bank:
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
                accumulator[sequence][first_lane] + dot.first;
            next_accumulator[second_lane] =
                accumulator[sequence][second_lane] + dot.second;
        }
    ffn_up_i8_write_accumulator:
        for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS UNROLL
            accumulator[sequence][output_lane] = next_accumulator[output_lane];
        }
    }
}

static void ffn_up_i8_compute(
    ffn_i8_t input_buffer[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE],
    const packed_i8x64_t *w1,
    const int *b1,
    const float *w1_scale,
    const ffn_i8_requant_param_t *w1_to_gelu_requant,
    const packed_i8x64_t *gelu_lut_packed,
    float input_scale,
    float gelu_inverse_scale,
    packed_i8_stream_t &gelu_i8_stream) {
#pragma HLS INLINE off
    ffn_acc_t accumulator[FFN_I8_SEQ_LEN][FFN_I8_OUT_PAR];
    ffn_i8_t weight_tiles[2][FFN_I8_OUT_PAR][FFN_I8_K_PAR];
    ffn_i8_t gelu_tile[FFN_I8_SEQ_LEN][FFN_I8_STREAM_TILE];
#pragma HLS BIND_STORAGE variable=accumulator type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_tiles complete dim=3
#pragma HLS ARRAY_PARTITION variable=gelu_tile complete dim=2

#if FFN_I8_GELU_PROFILE == 2
    packed_i8x64_t gelu_lut_words[4];
    ffn_i8_t gelu_lut[FFN_I8_OUT_PAR][256];
#pragma HLS ARRAY_PARTITION variable=gelu_lut_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=gelu_lut complete dim=1
#pragma HLS BIND_STORAGE variable=gelu_lut type=RAM_1P impl=BRAM latency=1
ffn_up_i8_load_gelu_lut_words:
    for (int word = 0; word < 4; ++word) {
#pragma HLS PIPELINE II=1
        gelu_lut_words[word] = gelu_lut_packed[word];
    }
ffn_up_i8_expand_gelu_lut:
    for (int index = 0; index < 256; ++index) {
#pragma HLS PIPELINE II=1
        ffn_i8_t value = ffn_i8_unpack_byte(gelu_lut_words[index / 64], index % 64);
    ffn_up_i8_replicate_gelu_lut:
        for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS UNROLL
            gelu_lut[output_lane][index] = value;
        }
    }
#endif

ffn_up_i8_stream_tile:
    for (int stream_base = 0; stream_base < FFN_I8_INTERMEDIATE_SIZE; stream_base += FFN_I8_STREAM_TILE) {
    ffn_up_i8_compute_subtile:
        for (int sub_base = 0; sub_base < FFN_I8_STREAM_TILE; sub_base += FFN_I8_OUT_PAR) {
            int output_base = stream_base + sub_base;
            int output_tile = output_base / FFN_I8_OUT_PAR;
            ffn_acc_t bias_tile[FFN_I8_OUT_PAR];
#if FFN_I8_GELU_PROFILE == 2
            ffn_i8_requant_param_t scale_tile[FFN_I8_OUT_PAR];
#else
            float scale_tile[FFN_I8_OUT_PAR];
#endif
#pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=scale_tile complete dim=1

        ffn_up_i8_load_bias:
            for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS PIPELINE II=1
                bias_tile[output_lane] = (ffn_acc_t)b1[output_base + output_lane];
            }
        ffn_up_i8_load_scale:
            for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS PIPELINE II=1
#if FFN_I8_GELU_PROFILE == 2
                scale_tile[output_lane] = w1_to_gelu_requant[output_base + output_lane];
#else
                scale_tile[output_lane] = w1_scale[output_base + output_lane];
#endif
            }
        ffn_up_i8_initialize_accumulator:
            for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
            ffn_up_i8_initialize_lane:
                for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS UNROLL
                    accumulator[sequence][output_lane] = bias_tile[output_lane];
                }
            }

            ffn_i8_load_weight_tile(
                w1,
                output_tile,
                0,
                FFN_I8_HIDDEN_SIZE / FFN_I8_K_PAR,
                weight_tiles[0]);
        ffn_up_i8_k_tile_overlap:
            for (int k_tile = 0;
                 k_tile < FFN_I8_HIDDEN_SIZE / FFN_I8_K_PAR;
                 ++k_tile) {
                int next_k_tile = k_tile + 1;
                if (next_k_tile == FFN_I8_HIDDEN_SIZE / FFN_I8_K_PAR) {
                    // The final load is intentionally a harmless prefetch of
                    // tile zero so every K tile uses the same fused pipeline.
                    next_k_tile = 0;
                }
                ffn_up_i8_prefetch_and_mac_tile(
                    w1, output_tile, k_tile, next_k_tile,
                    input_buffer, accumulator, weight_tiles);
            }

        ffn_up_i8_gelu_requantize:
            for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
            ffn_up_i8_gelu_lane:
                for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
#pragma HLS UNROLL
#if FFN_I8_GELU_PROFILE == 2
                    ffn_i8_t pre_gelu = ffn_i8_requantize_dsp(
                        accumulator[sequence][output_lane], scale_tile[output_lane]);
                    ap_uint<8> lut_index = (ap_uint<8>)((ap_int<9>)pre_gelu + 128);
                    gelu_tile[sequence][sub_base + output_lane] =
                        gelu_lut[output_lane][lut_index];
#else
                    float dequantized = (float)accumulator[sequence][output_lane] *
                                        (input_scale * scale_tile[output_lane]);
                    float activated = ffn_i8_gelu_pwl(dequantized);
                    gelu_tile[sequence][sub_base + output_lane] =
                        ffn_i8_quantize(activated, gelu_inverse_scale);
#endif
                }
            }
        }

    ffn_up_i8_write_stream:
        for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
#pragma HLS PIPELINE II=1
            packed_i8x64_t word = 0;
        ffn_up_i8_pack_stream_lane:
            for (int lane = 0; lane < FFN_I8_STREAM_TILE; ++lane) {
#pragma HLS UNROLL
                word.range(lane * 8 + 7, lane * 8) = gelu_tile[sequence][lane].range(7, 0);
            }
            gelu_i8_stream.write(word);
        }
    }
}

extern "C" {

void ffn_up_i8_v1(
    packed_f32_stream_t &attn_mid_stream,
    const packed_i8x64_t *w1,
    const int *b1,
    const float *w1_scale,
    const ffn_i8_requant_param_t *w1_to_gelu_requant,
    const packed_i8x64_t *gelu_lut_packed,
    float input_inverse_scale,
    float input_scale,
    float gelu_inverse_scale,
    packed_i8_stream_t &gelu_i8_stream) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_i8_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 depth=36864
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 depth=3072
#pragma HLS INTERFACE m_axi port=w1_scale offset=slave bundle=gmem_w1 depth=3072
#pragma HLS INTERFACE m_axi port=w1_to_gelu_requant offset=slave bundle=gmem_w1 depth=3072
#pragma HLS INTERFACE m_axi port=gelu_lut_packed offset=slave bundle=gmem_w1 depth=4
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w1_scale bundle=control
#pragma HLS INTERFACE s_axilite port=w1_to_gelu_requant bundle=control
#pragma HLS INTERFACE s_axilite port=gelu_lut_packed bundle=control
#pragma HLS INTERFACE s_axilite port=input_inverse_scale bundle=control
#pragma HLS INTERFACE s_axilite port=input_scale bundle=control
#pragma HLS INTERFACE s_axilite port=gelu_inverse_scale bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    static ffn_i8_t input_buffer[FFN_I8_SEQ_LEN][FFN_I8_HIDDEN_SIZE];
#ifdef FFN_I8_USE_BRAM
#pragma HLS BIND_STORAGE variable=input_buffer type=RAM_1P impl=BRAM latency=2
#else
#pragma HLS BIND_STORAGE variable=input_buffer type=RAM_1P impl=URAM latency=2
#endif
#if FFN_I8_K_PAR == 16
#pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=16 dim=2
#elif FFN_I8_K_PAR == 32
#pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=32 dim=2
#else
#pragma HLS ARRAY_PARTITION variable=input_buffer cyclic factor=64 dim=2
#endif

    ffn_up_i8_load_input(attn_mid_stream, input_inverse_scale, input_buffer);
    ffn_up_i8_compute(
        input_buffer,
        w1,
        b1,
        w1_scale,
        w1_to_gelu_requant,
        gelu_lut_packed,
        input_scale,
        gelu_inverse_scale,
        gelu_i8_stream);
}

}
