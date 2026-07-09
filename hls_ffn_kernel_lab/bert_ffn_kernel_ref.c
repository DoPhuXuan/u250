#include <math.h>

#include "bert_ffn_kernel_lab.h"

float gelu_tanh_ref(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

float gelu_pwl_ref(float x) {
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

static void bert_ffn_kernel_ref_impl(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE],
    int use_pwl
) {
    static float intermediate[SEQ_LEN][INTERMEDIATE_SIZE];
    int t;
    int h;
    int i;
    int o;

    for (t = 0; t < SEQ_LEN; t++) {
        for (i = 0; i < INTERMEDIATE_SIZE; i++) {
            float acc = b1[i];
            for (h = 0; h < HIDDEN_SIZE; h++) {
                acc += input_hidden[t][h] * w1[h][i];
            }
            intermediate[t][i] = use_pwl ? gelu_pwl_ref(acc) : gelu_tanh_ref(acc);
        }
    }

    for (t = 0; t < SEQ_LEN; t++) {
        for (o = 0; o < HIDDEN_SIZE; o++) {
            float acc = b2[o];
            for (i = 0; i < INTERMEDIATE_SIZE; i++) {
                acc += intermediate[t][i] * w2[i][o];
            }
            output_hidden[t][o] = acc;
        }
    }
}

void bert_ffn_kernel_ref_tanh(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
    bert_ffn_kernel_ref_impl(input_hidden, w1, b1, w2, b2, output_hidden, 0);
}

void bert_ffn_kernel_ref_pwl(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
    bert_ffn_kernel_ref_impl(input_hidden, w1, b1, w2, b2, output_hidden, 1);
}
