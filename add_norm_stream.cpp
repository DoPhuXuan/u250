#include "bert_model.h"

void add_norm_stream(
    hidden_stream_t &residual_in,
    hidden_stream_t &sublayer_in,
    const bus_t     *gamma_in,
    const bus_t     *beta_in,
    hidden_stream_t &output_stream)
{
#pragma HLS INLINE off

    static const int RED_GROUPS = 4;
    static const int RED_BANKS = RED_GROUPS * PACK_SIZE;
    static const float INV_HIDDEN = 1.0f / HIDDEN;

    bus_t gamma_cache[PACKS];
    bus_t beta_cache[PACKS];
    bus_t local_x[PACKS];
#pragma HLS BIND_STORAGE variable=gamma_cache type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=beta_cache type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=local_x type=RAM_2P impl=BRAM

LOAD_GAMMA:
    for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
        gamma_cache[p] = gamma_in[p];
    }

LOAD_BETA:
    for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
        beta_cache[p] = beta_in[p];
    }

TOKEN_LOOP:
    for (int s = 0; s < SEQ_LEN; ++s) {
        float sum_banks[RED_BANKS];
        float sumsq_banks[RED_BANKS];
#pragma HLS ARRAY_PARTITION variable=sum_banks complete dim=1
#pragma HLS ARRAY_PARTITION variable=sumsq_banks complete dim=1

INIT_BANKS:
        for (int b = 0; b < RED_BANKS; ++b) {
#pragma HLS UNROLL
            sum_banks[b] = 0.0f;
            sumsq_banks[b] = 0.0f;
        }

COMBINE_AND_ACC:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float res_lane[PACK_SIZE];
            float sub_lane[PACK_SIZE];
            float x_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=res_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=sub_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1

            const int bank_group = p & (RED_GROUPS - 1);

            unpack_bus16(residual_in.read(), res_lane);
            unpack_bus16(sublayer_in.read(), sub_lane);

        ACCUM_LANES:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int bank = bank_group * PACK_SIZE + i;
                const float x = res_lane[i] + sub_lane[i];
                x_lane[i] = x;
                sum_banks[bank] += x;
                sumsq_banks[bank] += x * x;
            }

            local_x[p] = pack_bus16(x_lane);
        }

        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;

REDUCE_BANKS:
        for (int b = 0; b < RED_BANKS; ++b) {
#pragma HLS PIPELINE II=1
            total_sum += sum_banks[b];
            total_sum_sq += sumsq_banks[b];
        }

        const float mean = total_sum * INV_HIDDEN;
        const float variance = (total_sum_sq * INV_HIDDEN) - (mean * mean);
        const float inv_stddev = hls::rsqrtf(variance + 1e-12f);

APPLY_NORM:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float x_lane[PACK_SIZE];
            float g_lane[PACK_SIZE];
            float b_lane[PACK_SIZE];
            float out_lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=x_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=g_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_lane complete dim=1

            unpack_bus16(local_x[p], x_lane);
            unpack_bus16(gamma_cache[p], g_lane);
            unpack_bus16(beta_cache[p], b_lane);

        NORM_LANES:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const float norm_val = (x_lane[i] - mean) * inv_stddev;
                out_lane[i] = g_lane[i] * norm_val + b_lane[i];
            }

            output_stream.write(pack_bus16(out_lane));
        }
    }
}

