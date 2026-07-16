#include "bert_stage_kernels.h"

static void store_hidden_stream_and_signal(
    hidden_stream_t &src,
    bus_t           *dst,
    volatile unsigned int *ready,
    const stage_token_t token)
{
#pragma HLS INLINE off
STORE_EMBEDDING_HIDDEN:
    for (int i = 0; i < TOKEN_WORDS; ++i) {
#pragma HLS PIPELINE II=1
        dst[i] = src.read();
    }
    write_stage_token(ready, token);
}

static void prepare_embedding_hidden(
    const int   *input_ids,
    const int   *token_type_ids,
    const bus_t *token_emb,
    const bus_t *pos_emb,
    const bus_t *seg_emb,
    const bus_t *emb_gamma,
    const bus_t *emb_beta,
    bus_t       *embedding_hidden,
    volatile unsigned int *hidden_ready,
    const stage_token_t ready_token)
{
#pragma HLS INLINE off
    hidden_stream_t embedding_stream("embedding_stream");
#pragma HLS STREAM variable=embedding_stream depth=64

#pragma HLS DATAFLOW
    bert_embedding_stream(
        input_ids, token_type_ids, token_emb, pos_emb, seg_emb,
        emb_gamma, emb_beta, embedding_stream);
    store_hidden_stream_and_signal(
        embedding_stream, embedding_hidden, hidden_ready, ready_token);
}

extern "C" {
void bert_embedding_prep_kernel(
    const int   *input_ids,
    const int   *token_type_ids,
    const bus_t *token_emb,
    const bus_t *pos_emb,
    const bus_t *seg_emb,
    const bus_t *emb_gamma,
    const bus_t *emb_beta,
    bus_t       *embedding_hidden,
    volatile unsigned int *hidden_ready,
    int layer_id)
{
#pragma HLS INTERFACE m_axi port=input_ids offset=slave bundle=gmem_io depth=128
#pragma HLS INTERFACE m_axi port=token_type_ids offset=slave bundle=gmem_io depth=128
#pragma HLS INTERFACE m_axi port=token_emb offset=slave bundle=gmem_emb depth=1465056
#pragma HLS INTERFACE m_axi port=pos_emb offset=slave bundle=gmem_emb depth=6144
#pragma HLS INTERFACE m_axi port=seg_emb offset=slave bundle=gmem_emb depth=96
#pragma HLS INTERFACE m_axi port=emb_gamma offset=slave bundle=gmem_emb depth=48
#pragma HLS INTERFACE m_axi port=emb_beta offset=slave bundle=gmem_emb depth=48
#pragma HLS INTERFACE m_axi port=embedding_hidden offset=slave bundle=gmem_hidden depth=6144
#pragma HLS INTERFACE m_axi port=hidden_ready offset=slave bundle=gmem_hidden_ready depth=1
#pragma HLS INTERFACE s_axilite port=input_ids bundle=control
#pragma HLS INTERFACE s_axilite port=token_type_ids bundle=control
#pragma HLS INTERFACE s_axilite port=token_emb bundle=control
#pragma HLS INTERFACE s_axilite port=pos_emb bundle=control
#pragma HLS INTERFACE s_axilite port=seg_emb bundle=control
#pragma HLS INTERFACE s_axilite port=emb_gamma bundle=control
#pragma HLS INTERFACE s_axilite port=emb_beta bundle=control
#pragma HLS INTERFACE s_axilite port=embedding_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_ready bundle=control
#pragma HLS INTERFACE s_axilite port=layer_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const stage_token_t ready_token = (stage_token_t)(layer_id + 1);

    if (layer_id == 0) {
        prepare_embedding_hidden(
            input_ids, token_type_ids, token_emb, pos_emb, seg_emb,
            emb_gamma, emb_beta, embedding_hidden, hidden_ready, ready_token);
    } else {
        // Later layers already have their hidden input in DDR from the
        // previous FFN-down stage.  This prep kernel only publishes the token;
        // QKV projection lives exclusively in bert_qkv_kernel.
        write_stage_token(hidden_ready, ready_token);
    }
}
} // extern "C"
