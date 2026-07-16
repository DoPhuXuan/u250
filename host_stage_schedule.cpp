// Pseudocode for an out-of-order XRT/OpenCL host queue.
// It is intentionally API-neutral: retain the project's existing BO creation
// and set_arg/enqueue wrappers, but preserve this launch ordering.

// Required BO placement from system.cfg:
//   DDR[0]: hidden_ping, hidden_pong, residual_ready, inputs/mask
//   DDR[1]: embedding tables/norm params consumed by embedding_prep
// All model-weight BOs must be placed in the DDR bank of their consuming port.
//
// Required stream topology after the low-latency dataflow refactor:
//   qkv_stream:               qkv -> attn_core
//   context_stream:           attn_core -> attn_out
//   attn_mid_stream:          attn_out -> ffn_up
//   attn_mid_residual_stream: attn_out -> ffn_down
//   gelu_stream:              ffn_up -> ffn_down

void run_bert_12_layers(/* existing runtime handles and model BOs */)
{
    // Initialize the one remaining DDR token to zero once before layer 0.
    write_u32_to_device(residual_ready, 0);

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const bool first_layer = (layer == 0);
        const bool even_layer = ((layer & 1) == 0);

        // layer 0: embedding_prep writes embedding to ping; qkv and attention
        //          use ping; ffn down writes pong.
        // later:   previous output is input/residual; output ping-pongs.
        device_buffer &hidden_in = even_layer ? hidden_ping : hidden_pong;
        device_buffer &hidden_out = even_layer ? hidden_pong : hidden_ping;
        device_buffer &qkv_input = first_layer ? hidden_ping : hidden_in;
        device_buffer &attention_residual = first_layer ? hidden_ping : hidden_in;

        // Every invocation gets the complete model-base pointer; layer_id
        // selects the per-layer packed offset inside the kernel.
        set_embedding_prep_args(
            k_embedding_prep, input_ids, token_type_ids,
            token_emb, pos_emb, seg_emb, emb_gamma, emb_beta,
            hidden_ping, residual_ready, layer);
        set_qkv_args(
            k_qkv, qkv_input, residual_ready,
            attn_q_w_all, attn_q_b_all, attn_k_w_all, attn_k_b_all,
            attn_v_w_all, attn_v_b_all, layer);
        set_attn_core_args(k_attn_core, attention_mask);
        set_attn_out_args(
            k_attn_out, attention_residual, residual_ready,
            attn_o_w_all, attn_o_b_all,
            attn_norm_gamma_all, attn_norm_beta_all,
            layer);
        set_ffn_up_args(k_ffn_up, ffn_up_w_all, ffn_up_b_all, layer);
        set_ffn_down_args(
            k_ffn_down, ffn_down_w_all, ffn_down_b_all,
            ffn_norm_gamma_all, ffn_norm_beta_all,
            hidden_out, layer);

        // Critical: use one out-of-order command queue (or independent
        // queues). Do not put these tasks in a single in-order queue: the
        // producer can fill an AXIS FIFO before its consumer is started.
        event e_embedding_prep = enqueue_task_ooo(k_embedding_prep);
        event e_qkv = enqueue_task_ooo(k_qkv);
        event e_attn = enqueue_task_ooo(k_attn_core);
        event e_attn_out = enqueue_task_ooo(k_attn_out);
        event e_ffn_up = enqueue_task_ooo(k_ffn_up);
        event e_ffn_down = enqueue_task_ooo(k_ffn_down);

        // This also makes hidden_out a completed DDR buffer before it becomes
        // hidden_in on the next loop iteration. The in-device token waits let
        // qkv and attn_out observe when embedding_prep has materialized layer
        // 0 hidden state, or when later-layer hidden_in is ready to reuse.
        // FFN residual now travels over AXIS, so ffn_down can run in the same
        // layer dataflow wave instead of waiting for an attn_mid DDR token.
        wait(e_ffn_down);
    }

    // NUM_LAYERS is even, so layer 11 wrote hidden_ping.
    read_hidden_from_device(hidden_ping);
}
