#include "bert_ffn_kernel_lab.h"

static float gelu_pwl_hls(float x) {
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

void bert_ffn_kernel_v1_tiled_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_buf[SEQ_LEN][HIDDEN_SIZE];
    static float out_buf[SEQ_LEN][HIDDEN_SIZE];
    static float w1_tile[TILE_I][HIDDEN_SIZE];
    static float gelu_buf[SEQ_LEN][TILE_I];
    static float w2_tile[TILE_I][TILE_O];
    float out_tile[TILE_O];

#pragma HLS BIND_STORAGE variable=x_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=out_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=gelu_buf type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w2_tile type=RAM_2P impl=BRAM latency=2

#pragma HLS ARRAY_PARTITION variable=x_buf cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=out_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w1_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=gelu_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=w2_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=out_tile cyclic factor=16 dim=1

    int t;
    int h;
    int i0;
    int ii;
    int o0;
    int oo;

load_input:
    for (t = 0; t < SEQ_LEN; t++) {
        for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
            x_buf[t][h] = input_hidden[t][h];
        }
    }

init_output:
    for (t = 0; t < SEQ_LEN; t++) {
        for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
            out_buf[t][h] = b2[h];
        }
    }

tile_intermediate:
    for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
    load_w1_tile_h:
        for (h = 0; h < HIDDEN_SIZE; h++) {
        load_w1_tile_i:
            for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                w1_tile[ii][h] = w1[h][i0 + ii];
            }
        }

    compute_w1_gelu_t:
        for (t = 0; t < SEQ_LEN; t++) {
        compute_w1_gelu_i:
            for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                float acc = b1[i0 + ii];
            compute_w1_reduce_h:
                for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                    acc += x_buf[t][h] * w1_tile[ii][h];
                }
                gelu_buf[t][ii] = gelu_pwl_hls(acc);
            }
        }

    tile_output:
        for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
        load_w2_tile_i:
            for (ii = 0; ii < TILE_I; ii++) {
            load_w2_tile_o:
                for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                    w2_tile[ii][oo] = w2[i0 + ii][o0 + oo];
                }
            }

        update_output_t:
            for (t = 0; t < SEQ_LEN; t++) {
            load_out_tile:
                for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                    out_tile[oo] = out_buf[t][o0 + oo];
                }

            update_output_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    float g = gelu_buf[t][ii];
                update_output_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                        out_tile[oo] += g * w2_tile[ii][oo];
                    }
                }

            store_out_tile:
                for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                    out_buf[t][o0 + oo] = out_tile[oo];
                }
            }
        }
    }

store_output:
    for (t = 0; t < SEQ_LEN; t++) {
        for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
            output_hidden[t][h] = out_buf[t][h];
        }
    }
}

void bert_ffn_kernel_v2_row_local_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_row[HIDDEN_SIZE];
    static float out_acc[HIDDEN_SIZE];
    static float w1_tile[TILE_I][HIDDEN_SIZE];
    static float gelu_tile[TILE_I];
    static float w2_tile[TILE_I][TILE_O];
    float out_tile[TILE_O];

#pragma HLS BIND_STORAGE variable=x_row type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=out_acc type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w1_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=gelu_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w2_tile type=RAM_2P impl=BRAM latency=2

#pragma HLS ARRAY_PARTITION variable=x_row cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=w1_tile cyclic factor=32 dim=2
#pragma HLS ARRAY_PARTITION variable=gelu_tile cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=w2_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=out_tile cyclic factor=16 dim=1

    int t;
    int h;
    int i0;
    int ii;
    int o0;
    int oo;

v2_token_loop:
    for (t = 0; t < SEQ_LEN; t++) {
    v2_load_input_row:
        for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
            x_row[h] = input_hidden[t][h];
        }

    v2_init_output_row:
        for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
            out_acc[h] = b2[h];
        }

    v2_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v2_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v2_load_w1_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    w1_tile[ii][h] = w1[h][i0 + ii];
                }
            }

        v2_compute_gelu_tile:
            for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                float acc = b1[i0 + ii];
            v2_compute_w1_reduce_h:
                for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                    acc += x_row[h] * w1_tile[ii][h];
                }
                gelu_tile[ii] = gelu_pwl_hls(acc);
            }

        v2_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v2_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v2_load_w2_tile_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        w2_tile[ii][oo] = w2[i0 + ii][o0 + oo];
                    }
                }

            v2_load_out_tile:
                for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                    out_tile[oo] = out_acc[o0 + oo];
                }

            v2_update_output_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    float g = gelu_tile[ii];
                v2_update_output_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                        out_tile[oo] += g * w2_tile[ii][oo];
                    }
                }

            v2_store_out_tile:
                for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                    out_acc[o0 + oo] = out_tile[oo];
                }
            }
        }

    v2_store_output_row:
        for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
            output_hidden[t][h] = out_acc[h];
        }
    }
}

void bert_ffn_kernel_v3_token_tile_t32_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_tile[TILE_T][HIDDEN_SIZE];
    static float out_acc[TILE_T][HIDDEN_SIZE];
    static float w1_tile[TILE_I][HIDDEN_SIZE];
    static float gelu_tile[TILE_T][TILE_I];
    static float w2_tile[TILE_I][TILE_O];
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
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1

    int t0;
    int tt;
    int h;
    int i0;
    int ii;
    int o0;
    int oo;

v3_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T) {
    v3_load_input_t:
        for (tt = 0; tt < TILE_T; tt++) {
        v3_load_input_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                x_tile[tt][h] = input_hidden[t0 + tt][h];
            }
        }

    v3_init_output_t:
        for (tt = 0; tt < TILE_T; tt++) {
        v3_init_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                out_acc[tt][h] = b2[h];
            }
        }

    v3_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v3_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v3_load_w1_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    w1_tile[ii][h] = w1[h][i0 + ii];
                }
            }

        v3_compute_gelu_t:
            for (tt = 0; tt < TILE_T; tt++) {
            v3_compute_gelu_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    float acc = b1[i0 + ii];
                v3_compute_w1_reduce_h:
                    for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                        acc += x_tile[tt][h] * w1_tile[ii][h];
                    }
                    gelu_tile[tt][ii] = gelu_pwl_hls(acc);
                }
            }

        v3_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v3_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v3_load_w2_tile_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        w2_tile[ii][oo] = w2[i0 + ii][o0 + oo];
                    }
                }

            v3_update_output_t:
                for (tt = 0; tt < TILE_T; tt++) {
                v3_load_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_vec[oo] = out_acc[tt][o0 + oo];
                    }

                v3_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                        float g = gelu_tile[tt][ii];
                    v3_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_vec[oo] += g * w2_tile[ii][oo];
                        }
                    }

                v3_store_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_acc[tt][o0 + oo] = out_vec[oo];
                    }
                }
            }
        }

    v3_store_output_t:
        for (tt = 0; tt < TILE_T; tt++) {
        v3_store_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                output_hidden[t0 + tt][h] = out_acc[tt][h];
            }
        }
    }
}

void bert_ffn_kernel_v4_token_tile_partial_accum(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float x_tile[TILE_T][HIDDEN_SIZE];
    static float out_acc[TILE_T][HIDDEN_SIZE];
    static float w1_tile[TILE_I][HIDDEN_SIZE];
    static float gelu_tile[TILE_T][TILE_I];
    static float w2_tile[TILE_I][TILE_O];
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
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1

    int t0;
    int tt;
    int h;
    int h0;
    int i0;
    int ii;
    int o0;
    int oo;
    int u;
    int p;
    int bank;

v4_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T) {
    v4_load_input_t:
        for (tt = 0; tt < TILE_T; tt++) {
        v4_load_input_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                x_tile[tt][h] = input_hidden[t0 + tt][h];
            }
        }

    v4_init_output_t:
        for (tt = 0; tt < TILE_T; tt++) {
        v4_init_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                out_acc[tt][h] = b2[h];
            }
        }

    v4_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v4_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v4_load_w1_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    w1_tile[ii][h] = w1[h][i0 + ii];
                }
            }

        v4_compute_gelu_t:
            for (tt = 0; tt < TILE_T; tt++) {
            v4_compute_gelu_i:
                for (ii = 0; ii < TILE_I; ii++) {
                    float acc_part[4][32];
#pragma HLS ARRAY_PARTITION variable=acc_part complete dim=0

                v4_acc_part_init_p:
                    for (p = 0; p < 4; p++) {
#pragma HLS UNROLL
                    v4_acc_part_init_u:
                        for (u = 0; u < 32; u++) {
#pragma HLS UNROLL
                            acc_part[p][u] = 0.0f;
                        }
                    }

                v4_compute_w1_reduce_h0:
                    for (h0 = 0; h0 < HIDDEN_SIZE; h0 += 32) {
#pragma HLS PIPELINE II=1
                        bank = (h0 >> 5) & 3;
                    v4_compute_w1_reduce_u:
                        for (u = 0; u < 32; u++) {
#pragma HLS UNROLL
                            if ((h0 + u) < HIDDEN_SIZE) {
                                acc_part[bank][u] += x_tile[tt][h0 + u] * w1_tile[ii][h0 + u];
                            }
                        }
                    }

                    float acc = b1[i0 + ii];
                v4_sum_part_p:
                    for (p = 0; p < 4; p++) {
#pragma HLS UNROLL
                    v4_sum_part_u:
                        for (u = 0; u < 32; u++) {
#pragma HLS UNROLL
                            acc += acc_part[p][u];
                        }
                    }
                    gelu_tile[tt][ii] = gelu_pwl_hls(acc);
                }
            }

        v4_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v4_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v4_load_w2_tile_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        w2_tile[ii][oo] = w2[i0 + ii][o0 + oo];
                    }
                }

            v4_update_output_t:
                for (tt = 0; tt < TILE_T; tt++) {
                    float out_part[8][TILE_O];
#pragma HLS ARRAY_PARTITION variable=out_part complete dim=0

                v4_load_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_vec[oo] = out_acc[tt][o0 + oo];
                    }

                v4_out_part_init_p:
                    for (p = 0; p < 8; p++) {
#pragma HLS UNROLL
                    v4_out_part_init_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_part[p][oo] = 0.0f;
                        }
                    }

                v4_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=out_part inter false
                        bank = ii & 7;
                        float g = gelu_tile[tt][ii];
                    v4_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_part[bank][oo] += g * w2_tile[ii][oo];
                        }
                    }

                v4_sum_out_vec_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        float acc = out_vec[oo];
                    v4_sum_out_vec_p:
                        for (p = 0; p < 8; p++) {
#pragma HLS UNROLL
                            acc += out_part[p][oo];
                        }
                        out_vec[oo] = acc;
                    }

                v4_store_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_acc[tt][o0 + oo] = out_vec[oo];
                    }
                }
            }
        }

    v4_store_output_t:
        for (tt = 0; tt < TILE_T; tt++) {
        v4_store_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                output_hidden[t0 + tt][h] = out_acc[tt][h];
            }
        }
    }
}

void bert_ffn_kernel_v5_token_tile_t64_i128_h32_o16(
    float input_hidden[SEQ_LEN][HIDDEN_SIZE],
    float w1[HIDDEN_SIZE][INTERMEDIATE_SIZE],
    float b1[INTERMEDIATE_SIZE],
    float w2[INTERMEDIATE_SIZE][HIDDEN_SIZE],
    float b2[HIDDEN_SIZE],
    float output_hidden[SEQ_LEN][HIDDEN_SIZE]
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out max_widen_bitwidth=512
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
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1

    int t0;
    int tt;
    int h;
    int i0;
    int ii;
    int o0;
    int oo;

v5_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T_V5) {
    v5_load_input_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v5_load_input_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                x_tile[tt][h] = input_hidden[t0 + tt][h];
            }
        }

    v5_init_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v5_init_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                out_acc[tt][h] = b2[h];
            }
        }

    v5_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v5_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v5_load_w1_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    w1_tile[ii][h] = w1[h][i0 + ii];
                }
            }

        v5_compute_gelu_t:
            for (tt = 0; tt < TILE_T_V5; tt++) {
            v5_compute_gelu_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    float acc = b1[i0 + ii];
                v5_compute_w1_reduce_h:
                    for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                        acc += x_tile[tt][h] * w1_tile[ii][h];
                    }
                    gelu_tile[tt][ii] = gelu_pwl_hls(acc);
                }
            }

        v5_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v5_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v5_load_w2_tile_o:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        w2_tile[ii][oo] = w2[i0 + ii][o0 + oo];
                    }
                }

            v5_update_output_t:
                for (tt = 0; tt < TILE_T_V5; tt++) {
                v5_load_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_vec[oo] = out_acc[tt][o0 + oo];
                    }

                v5_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                        float g = gelu_tile[tt][ii];
                    v5_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_vec[oo] += g * w2_tile[ii][oo];
                        }
                    }

                v5_store_out_vec:
                    for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS PIPELINE II=1
                        out_acc[tt][o0 + oo] = out_vec[oo];
                    }
                }
            }
        }

    v5_store_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v5_store_output_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS PIPELINE II=1
                output_hidden[t0 + tt][h] = out_acc[tt][h];
            }
        }
    }
}

void bert_ffn_kernel_v6_flat_wide_t64_i128_h32_o16(
    float *input_hidden,
    float *w1,
    float *b1,
    float *w2,
    float *b2,
    float *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out max_widen_bitwidth=512
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
#pragma HLS ARRAY_PARTITION variable=out_vec cyclic factor=16 dim=1

    int t0;
    int tt;
    int h;
    int h0;
    int i0;
    int ii;
    int ii0;
    int o0;
    int oo;
    int u;

v6_token_tile_loop:
    for (t0 = 0; t0 < SEQ_LEN; t0 += TILE_T_V5) {
    v6_load_input_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v6_load_input_h0:
            for (h0 = 0; h0 < HIDDEN_SIZE; h0 += WIDE_FACTOR) {
            v6_load_input_u:
                for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                    x_tile[tt][h0 + u] = input_hidden[(t0 + tt) * HIDDEN_SIZE + h0 + u];
                }
            }
        }

    v6_init_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v6_init_output_h0:
            for (h0 = 0; h0 < HIDDEN_SIZE; h0 += WIDE_FACTOR) {
            v6_init_output_u:
                for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                    out_acc[tt][h0 + u] = b2[h0 + u];
                }
            }
        }

    v6_tile_intermediate:
        for (i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += TILE_I) {
        v6_load_w1_tile_h:
            for (h = 0; h < HIDDEN_SIZE; h++) {
            v6_load_w1_tile_i0:
                for (ii0 = 0; ii0 < TILE_I; ii0 += WIDE_FACTOR) {
                v6_load_w1_tile_u:
                    for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                        w1_tile[ii0 + u][h] = w1[h * INTERMEDIATE_SIZE + i0 + ii0 + u];
                    }
                }
            }

        v6_compute_gelu_t:
            for (tt = 0; tt < TILE_T_V5; tt++) {
            v6_compute_gelu_i:
                for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                    float acc = b1[i0 + ii];
                v6_compute_w1_reduce_h:
                    for (h = 0; h < HIDDEN_SIZE; h++) {
#pragma HLS UNROLL factor=32
                        acc += x_tile[tt][h] * w1_tile[ii][h];
                    }
                    gelu_tile[tt][ii] = gelu_pwl_hls(acc);
                }
            }

        v6_tile_output:
            for (o0 = 0; o0 < HIDDEN_SIZE; o0 += TILE_O) {
            v6_load_w2_tile_i:
                for (ii = 0; ii < TILE_I; ii++) {
                v6_load_w2_tile_o0:
                    for (oo = 0; oo < TILE_O; oo += WIDE_FACTOR) {
                    v6_load_w2_tile_u:
                        for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                            w2_tile[ii][oo + u] = w2[(i0 + ii) * HIDDEN_SIZE + o0 + oo + u];
                        }
                    }
                }

            v6_update_output_t:
                for (tt = 0; tt < TILE_T_V5; tt++) {
                v6_load_out_vec_o0:
                    for (oo = 0; oo < TILE_O; oo += WIDE_FACTOR) {
                    v6_load_out_vec_u:
                        for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                            out_vec[oo + u] = out_acc[tt][o0 + oo + u];
                        }
                    }

                v6_update_output_i:
                    for (ii = 0; ii < TILE_I; ii++) {
#pragma HLS PIPELINE II=1
                        float g = gelu_tile[tt][ii];
                    v6_update_output_o:
                        for (oo = 0; oo < TILE_O; oo++) {
#pragma HLS UNROLL factor=16
                            out_vec[oo] += g * w2_tile[ii][oo];
                        }
                    }

                v6_store_out_vec_o0:
                    for (oo = 0; oo < TILE_O; oo += WIDE_FACTOR) {
                    v6_store_out_vec_u:
                        for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                            out_acc[tt][o0 + oo + u] = out_vec[oo + u];
                        }
                    }
                }
            }
        }

    v6_store_output_t:
        for (tt = 0; tt < TILE_T_V5; tt++) {
        v6_store_output_h0:
            for (h0 = 0; h0 < HIDDEN_SIZE; h0 += WIDE_FACTOR) {
            v6_store_output_u:
                for (u = 0; u < WIDE_FACTOR; u++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten off
                    output_hidden[(t0 + tt) * HIDDEN_SIZE + h0 + u] = out_acc[tt][h0 + u];
                }
            }
        }
    }
}
