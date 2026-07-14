#include "ffn_int8_common.h"

#include <cmath>
#include <cstdio>
#include <vector>

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
    packed_i8_stream_t &gelu_i8_stream);

void ffn_down_i8_v1(
    packed_i8_stream_t &gelu_i8_stream,
    const packed_i8x64_t *w2,
    const int *b2,
    const float *w2_scale,
    float gelu_scale,
    packed_f32x16_t *output_hidden);
}

static ap_uint<32> test_float_bits(float value) {
    union {
        unsigned int u;
        float f;
    } conversion;
    conversion.f = value;
    return conversion.u;
}

static float test_bits_float(ap_uint<32> bits) {
    union {
        unsigned int u;
        float f;
    } conversion;
    conversion.u = (unsigned int)bits;
    return conversion.f;
}

static int test_round_even(float value) {
    int base = (int)value;
    float fraction = value - (float)base;
    if (fraction > 0.5f || (fraction == 0.5f && (base & 1))) {
        return base + 1;
    }
    if (fraction < -0.5f || (fraction == -0.5f && (base & 1))) {
        return base - 1;
    }
    return base;
}

static signed char test_quantize(float value) {
    int rounded = test_round_even(value);
    if (rounded > 127) rounded = 127;
    if (rounded < -127) rounded = -127;
    return (signed char)rounded;
}

static float test_gelu(float value) {
    if (value <= -3.0f) return 0.0f;
    if (value >= 3.0f) return value;
    return value * (0.5f + value * (1.0f / 6.0f));
}

static bool test_packed_dsp_multiply() {
    // Exhaust every signed INT8 product in each packed lane independently.
    for (int lhs = -128; lhs <= 127; ++lhs) {
        for (int rhs = -128; rhs <= 127; ++rhs) {
            ffn_i8_product_pair_t first = ffn_i8_multiply_pair_dsp(
                (ffn_i8_t)lhs, (ffn_i8_t)rhs, (ffn_i8_t)0);
            ffn_i8_product_pair_t second = ffn_i8_multiply_pair_dsp(
                (ffn_i8_t)lhs, (ffn_i8_t)0, (ffn_i8_t)rhs);
            int expected = lhs * rhs;
            if ((int)first.first != expected || (int)first.second != 0 ||
                (int)second.first != 0 || (int)second.second != expected) {
                std::fprintf(stderr,
                             "Packed DSP exhaustive mismatch lhs=%d rhs=%d\n",
                             lhs, rhs);
                return false;
            }
        }
    }

    // Exercise simultaneous signed lanes, including -128, with deterministic
    // random triples so cross-lane borrow/carry errors cannot hide behind zero.
    unsigned int state = 0x6d2b79f5u;
    for (int trial = 0; trial < 100000; ++trial) {
        state = state * 1664525u + 1013904223u;
        int lhs = (int)(signed char)(state >> 24);
        state = state * 1664525u + 1013904223u;
        int rhs_first = (int)(signed char)(state >> 24);
        state = state * 1664525u + 1013904223u;
        int rhs_second = (int)(signed char)(state >> 24);
        ffn_i8_product_pair_t actual = ffn_i8_multiply_pair_dsp(
            (ffn_i8_t)lhs, (ffn_i8_t)rhs_first, (ffn_i8_t)rhs_second);
        if ((int)actual.first != lhs * rhs_first ||
            (int)actual.second != lhs * rhs_second) {
            std::fprintf(stderr,
                         "Packed DSP random mismatch lhs=%d rhs=(%d,%d): (%d,%d)\n",
                         lhs, rhs_first, rhs_second,
                         (int)actual.first, (int)actual.second);
            return false;
        }
    }
    return true;
}

static void pack_weights(
    const std::vector<signed char> &logical,
    int outputs,
    int inputs,
    std::vector<packed_i8x64_t> &packed) {
    int output_tiles = outputs / FFN_I8_OUT_PAR;
    int k_tiles = inputs / FFN_I8_K_PAR;
    packed.assign(outputs * inputs / FFN_I8_I8_PER_WORD, 0);
    for (int output_tile = 0; output_tile < output_tiles; ++output_tile) {
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            int base_word =
                (output_tile * k_tiles + k_tile) * FFN_I8_WEIGHT_WORDS_PER_TILE;
            for (int k_lane = 0; k_lane < FFN_I8_K_PAR; ++k_lane) {
                for (int output_lane = 0; output_lane < FFN_I8_OUT_PAR; ++output_lane) {
                    int linear = k_lane * FFN_I8_OUT_PAR + output_lane;
                    int word_index = base_word + linear / FFN_I8_I8_PER_WORD;
                    int byte_lane = linear % FFN_I8_I8_PER_WORD;
                    int output = output_tile * FFN_I8_OUT_PAR + output_lane;
                    int input = k_tile * FFN_I8_K_PAR + k_lane;
                    ap_uint<8> bits = (unsigned char)logical[output * inputs + input];
                    packed[word_index].range(byte_lane * 8 + 7, byte_lane * 8) = bits;
                }
            }
        }
    }
}

int main() {
    if (!test_packed_dsp_multiply()) {
        return 5;
    }

    std::vector<signed char> input_q(FFN_I8_SEQ_LEN * FFN_I8_HIDDEN_SIZE);
    std::vector<signed char> w1_q(FFN_I8_INTERMEDIATE_SIZE * FFN_I8_HIDDEN_SIZE);
    std::vector<signed char> w2_q(FFN_I8_HIDDEN_SIZE * FFN_I8_INTERMEDIATE_SIZE);
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
        for (int hidden = 0; hidden < FFN_I8_HIDDEN_SIZE; ++hidden) {
            input_q[sequence * FFN_I8_HIDDEN_SIZE + hidden] =
                (signed char)(((sequence * 7 + hidden * 3) % 5) - 2);
        }
    }
    for (int output = 0; output < FFN_I8_INTERMEDIATE_SIZE; ++output) {
        for (int input = 0; input < FFN_I8_HIDDEN_SIZE; ++input) {
            w1_q[output * FFN_I8_HIDDEN_SIZE + input] =
                (signed char)(((output * 3 + input * 2 + 1) % 5) - 2);
        }
    }
    for (int output = 0; output < FFN_I8_HIDDEN_SIZE; ++output) {
        for (int input = 0; input < FFN_I8_INTERMEDIATE_SIZE; ++input) {
            w2_q[output * FFN_I8_INTERMEDIATE_SIZE + input] =
                (signed char)(((output * 11 + input * 7 + 2) % 5) - 2);
        }
    }

    std::vector<packed_i8x64_t> w1_packed;
    std::vector<packed_i8x64_t> w2_packed;
    pack_weights(w1_q, FFN_I8_INTERMEDIATE_SIZE, FFN_I8_HIDDEN_SIZE, w1_packed);
    pack_weights(w2_q, FFN_I8_HIDDEN_SIZE, FFN_I8_INTERMEDIATE_SIZE, w2_packed);
    std::vector<int> b1(FFN_I8_INTERMEDIATE_SIZE, 0);
    std::vector<int> b2(FFN_I8_HIDDEN_SIZE, 0);
    std::vector<float> sw1(FFN_I8_INTERMEDIATE_SIZE, 1.0f);
    std::vector<float> sw2(FFN_I8_HIDDEN_SIZE, 1.0f);
    ffn_i8_requant_param_t unit_requant =
        (ffn_i8_requant_param_t)65536 | ((ffn_i8_requant_param_t)16 << 18);
    std::vector<ffn_i8_requant_param_t> w1_to_gelu_requant(
        FFN_I8_INTERMEDIATE_SIZE, unit_requant);
    std::vector<packed_i8x64_t> gelu_lut_packed(4, 0);
    for (int input = -128; input <= 127; ++input) {
        int index = input + 128;
        signed char output = test_quantize(test_gelu((float)input));
        gelu_lut_packed[index / 64].range((index % 64) * 8 + 7, (index % 64) * 8) =
            (unsigned char)output;
    }

    packed_f32_stream_t input_stream;
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
        for (int pack = 0; pack < FFN_I8_HIDDEN_SIZE / FFN_I8_F32_PER_WORD; ++pack) {
            packed_f32x16_t word = 0;
            for (int lane = 0; lane < FFN_I8_F32_PER_WORD; ++lane) {
                float value = (float)input_q[
                    sequence * FFN_I8_HIDDEN_SIZE + pack * FFN_I8_F32_PER_WORD + lane];
                word.range(lane * 32 + 31, lane * 32) = test_float_bits(value);
            }
            input_stream.write(word);
        }
    }

    std::vector<signed char> golden_gelu(FFN_I8_SEQ_LEN * FFN_I8_INTERMEDIATE_SIZE);
    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
        for (int output = 0; output < FFN_I8_INTERMEDIATE_SIZE; ++output) {
            int accumulator = 0;
            for (int input = 0; input < FFN_I8_HIDDEN_SIZE; ++input) {
                accumulator +=
                    (int)input_q[sequence * FFN_I8_HIDDEN_SIZE + input] *
                    (int)w1_q[output * FFN_I8_HIDDEN_SIZE + input];
            }
            golden_gelu[sequence * FFN_I8_INTERMEDIATE_SIZE + output] =
                test_quantize(test_gelu((float)accumulator));
        }
    }

    packed_i8_stream_t up_output;
    ffn_up_i8_v1(
        input_stream,
        w1_packed.data(),
        b1.data(),
        sw1.data(),
        w1_to_gelu_requant.data(),
        gelu_lut_packed.data(),
        1.0f,
        1.0f,
        1.0f,
        up_output);

    packed_i8_stream_t down_input;
    int expected_words =
        FFN_I8_SEQ_LEN * FFN_I8_INTERMEDIATE_SIZE / FFN_I8_STREAM_TILE;
    for (int stream_tile = 0; stream_tile < FFN_I8_INTERMEDIATE_SIZE / FFN_I8_STREAM_TILE; ++stream_tile) {
        for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
            if (up_output.empty()) {
                std::fprintf(stderr, "UP stream ended early\n");
                return 1;
            }
            packed_i8x64_t word = up_output.read();
            --expected_words;
            for (int lane = 0; lane < FFN_I8_STREAM_TILE; ++lane) {
                ap_int<8> actual;
                actual.range(7, 0) = word.range(lane * 8 + 7, lane * 8);
                signed char expected = golden_gelu[
                    sequence * FFN_I8_INTERMEDIATE_SIZE +
                    stream_tile * FFN_I8_STREAM_TILE + lane];
                if ((int)actual != (int)expected) {
                    std::fprintf(stderr, "UP mismatch tile=%d sequence=%d lane=%d: %d != %d\n",
                                 stream_tile, sequence, lane, (int)actual, (int)expected);
                    return 2;
                }
            }
            down_input.write(word);
        }
    }
    if (expected_words != 0 || !up_output.empty()) {
        std::fprintf(stderr, "UP stream word count mismatch\n");
        return 3;
    }

    std::vector<packed_f32x16_t> output(
        FFN_I8_SEQ_LEN * FFN_I8_HIDDEN_SIZE / FFN_I8_F32_PER_WORD);
    ffn_down_i8_v1(
        down_input,
        w2_packed.data(),
        b2.data(),
        sw2.data(),
        1.0f,
        output.data());

    for (int sequence = 0; sequence < FFN_I8_SEQ_LEN; ++sequence) {
        for (int hidden = 0; hidden < FFN_I8_HIDDEN_SIZE; ++hidden) {
            int expected = 0;
            for (int input = 0; input < FFN_I8_INTERMEDIATE_SIZE; ++input) {
                expected +=
                    (int)golden_gelu[sequence * FFN_I8_INTERMEDIATE_SIZE + input] *
                    (int)w2_q[hidden * FFN_I8_INTERMEDIATE_SIZE + input];
            }
            int word_index =
                sequence * (FFN_I8_HIDDEN_SIZE / FFN_I8_F32_PER_WORD) +
                hidden / FFN_I8_F32_PER_WORD;
            int lane = hidden % FFN_I8_F32_PER_WORD;
            float actual = test_bits_float(output[word_index].range(lane * 32 + 31, lane * 32));
            if (actual != (float)expected) {
                std::fprintf(stderr, "DOWN mismatch sequence=%d hidden=%d: %.9g != %d\n",
                             sequence, hidden, actual, expected);
                return 4;
            }
        }
    }

    std::printf("PASS: bit-accurate UP stream and DOWN output (%d stream words)\n",
                FFN_I8_SEQ_LEN * FFN_I8_INTERMEDIATE_SIZE / FFN_I8_STREAM_TILE);
    return 0;
}
