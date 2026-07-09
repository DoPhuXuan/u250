#include <math.h>
#include <stdio.h>

#include "bert_ffn_kernel_lab.h"

static float input_hidden[SEQ_LEN][HIDDEN_SIZE];
static float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE];
static float b1[INTERMEDIATE_SIZE];
static float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE];
static float b2[HIDDEN_SIZE];
static float y_ref_tanh[SEQ_LEN][HIDDEN_SIZE];
static float y_ref_pwl[SEQ_LEN][HIDDEN_SIZE];
static float y_hls_v1[SEQ_LEN][HIDDEN_SIZE];
static float y_hls_v2[SEQ_LEN][HIDDEN_SIZE];
static float y_hls_v3[SEQ_LEN][HIDDEN_SIZE];
static float y_hls_v4[SEQ_LEN][HIDDEN_SIZE];
static float y_hls_v5[SEQ_LEN][HIDDEN_SIZE];
static float y_hls_v6[SEQ_LEN][HIDDEN_SIZE];

static void init_data(void) {
    int t;
    int h;
    int i;

    for (t = 0; t < SEQ_LEN; t++) {
        for (h = 0; h < HIDDEN_SIZE; h++) {
            input_hidden[t][h] = 0.02f * (float)(((t + 1) * (h + 3)) % 17 - 8);
        }
    }

    for (i = 0; i < INTERMEDIATE_SIZE; i++) {
        b1[i] = 0.001f * (float)((i % 7) - 3);
        for (h = 0; h < HIDDEN_SIZE; h++) {
            w2[i][h] = 0.0013f * (float)((((i + 1) * (h + 4)) % 17) - 8);
        }
    }

    for (h = 0; h < HIDDEN_SIZE; h++) {
        b2[h] = 0.001f * (float)((h % 5) - 2);
        for (i = 0; i < INTERMEDIATE_SIZE; i++) {
            w1[h][i] = 0.0017f * (float)((((h + 2) * (i + 3)) % 13) - 6);
        }
    }
}

static void compare_output(
    const char *name,
    float ref[SEQ_LEN][HIDDEN_SIZE],
    float got[SEQ_LEN][HIDDEN_SIZE]
) {
    int t;
    int h;
    float max_abs_error = 0.0f;
    float mean_abs_error = 0.0f;

    for (t = 0; t < SEQ_LEN; t++) {
        for (h = 0; h < HIDDEN_SIZE; h++) {
            float err = fabsf(ref[t][h] - got[t][h]);
            if (err > max_abs_error) {
                max_abs_error = err;
            }
            mean_abs_error += err;
        }
    }

    mean_abs_error /= (float)(SEQ_LEN * HIDDEN_SIZE);
    printf("%-34s max_abs_error=% .8f mean_abs_error=% .8f\n", name, max_abs_error, mean_abs_error);
}

int main(void) {
    init_data();

    printf("FFN kernel lab: SEQ_LEN=%d HIDDEN_SIZE=%d INTERMEDIATE_SIZE=%d TILE_T=%d TILE_T_V5=%d TILE_I=%d TILE_O=%d\n",
           SEQ_LEN,
           HIDDEN_SIZE,
           INTERMEDIATE_SIZE,
           TILE_T,
           TILE_T_V5,
           TILE_I,
           TILE_O);

    bert_ffn_kernel_ref_tanh(input_hidden, w1, b1, w2, b2, y_ref_tanh);
    bert_ffn_kernel_ref_pwl(input_hidden, w1, b1, w2, b2, y_ref_pwl);
    bert_ffn_kernel_v1_tiled_i128_h32_o16(input_hidden, w1, b1, w2, b2, y_hls_v1);
    bert_ffn_kernel_v2_row_local_i128_h32_o16(input_hidden, w1, b1, w2, b2, y_hls_v2);
    bert_ffn_kernel_v3_token_tile_t32_i128_h32_o16(input_hidden, w1, b1, w2, b2, y_hls_v3);
    bert_ffn_kernel_v4_token_tile_partial_accum(input_hidden, w1, b1, w2, b2, y_hls_v4);
    bert_ffn_kernel_v5_token_tile_t64_i128_h32_o16(input_hidden, w1, b1, w2, b2, y_hls_v5);
    bert_ffn_kernel_v6_flat_wide_t64_i128_h32_o16(
        (float *)input_hidden,
        (float *)w1,
        b1,
        (float *)w2,
        b2,
        (float *)y_hls_v6);

    compare_output("v1_hls_vs_ref_pwl", y_ref_pwl, y_hls_v1);
    compare_output("v1_hls_vs_ref_tanh", y_ref_tanh, y_hls_v1);
    compare_output("v2_hls_vs_ref_pwl", y_ref_pwl, y_hls_v2);
    compare_output("v2_hls_vs_ref_tanh", y_ref_tanh, y_hls_v2);
    compare_output("v3_hls_vs_ref_pwl", y_ref_pwl, y_hls_v3);
    compare_output("v3_hls_vs_ref_tanh", y_ref_tanh, y_hls_v3);
    compare_output("v4_hls_vs_ref_pwl", y_ref_pwl, y_hls_v4);
    compare_output("v4_hls_vs_ref_tanh", y_ref_tanh, y_hls_v4);
    compare_output("v5_hls_vs_ref_pwl", y_ref_pwl, y_hls_v5);
    compare_output("v5_hls_vs_ref_tanh", y_ref_tanh, y_hls_v5);
    compare_output("v6_hls_vs_ref_pwl", y_ref_pwl, y_hls_v6);
    compare_output("v6_hls_vs_ref_tanh", y_ref_tanh, y_hls_v6);
    compare_output("pwl_vs_tanh_reference", y_ref_tanh, y_ref_pwl);

    return 0;
}
