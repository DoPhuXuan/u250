#ifndef BERT_FFN_KERNEL_LAB_H
#define BERT_FFN_KERNEL_LAB_H

#ifndef FFN_KERNEL_TEST_SMALL
#define FFN_KERNEL_TEST_SMALL 0
#endif

#if FFN_KERNEL_TEST_SMALL
#define SEQ_LEN 4
#define HIDDEN_SIZE 8
#define INTERMEDIATE_SIZE 16
#define TILE_T 2
#define TILE_T_V5 4
#define TILE_I 8
#define TILE_O 8
#define WIDE_FACTOR 8
#else
#define SEQ_LEN 128
#define HIDDEN_SIZE 768
#define INTERMEDIATE_SIZE 3072
#define TILE_T 32
#define TILE_T_V5 64
#define TILE_I 128
#define TILE_O 128
#define WIDE_FACTOR 16
#endif

#if ((SEQ_LEN % TILE_T) != 0)
#error "SEQ_LEN must be divisible by TILE_T"
#endif

#if ((SEQ_LEN % TILE_T_V5) != 0)
#error "SEQ_LEN must be divisible by TILE_T_V5"
#endif

#if ((INTERMEDIATE_SIZE % TILE_I) != 0)
#error "INTERMEDIATE_SIZE must be divisible by TILE_I"
#endif

#if ((HIDDEN_SIZE % TILE_O) != 0)
#error "HIDDEN_SIZE must be divisible by TILE_O"
#endif

#if ((HIDDEN_SIZE % WIDE_FACTOR) != 0)
#error "HIDDEN_SIZE must be divisible by WIDE_FACTOR"
#endif

#if ((TILE_I % WIDE_FACTOR) != 0)
#error "TILE_I must be divisible by WIDE_FACTOR"
#endif

#if ((TILE_O % WIDE_FACTOR) != 0)
#error "TILE_O must be divisible by WIDE_FACTOR"
#endif

float gelu_tanh_ref(float x);
float gelu_pwl_ref(float x);

void bert_ffn_kernel_ref_tanh(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_ref_pwl(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_v1_tiled_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_v2_row_local_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_v3_token_tile_t32_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_v4_token_tile_partial_accum(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_v5_token_tile_t64_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
);

void bert_ffn_kernel_v6_flat_wide_t64_i128_h32_o16(
    float *input_hidden,
    float *w1,
    float *b1,
    float *w2,
    float *b2,
    float *output_hidden
);

#endif
