# Clean Attention SLR baseline

The active Attention implementation was rebuilt from scratch for Vitis HLS
2022.1 after repeated silent LTO failures in the incremental source.

## Kernel split

| SLR | Kernel | Work |
|---|---|---|
| SLR0 | `bert_qkv_kernel` | Fused FP32 Q/K/V projection |
| SLR1 | `bert_attn_core_kernel` | score, mask, softmax, weighted sum |
| SLR2 | `bert_attn_out_norm_kernel` | output projection, residual, layer norm |

## Optimizations retained

- 512-bit DDR and AXIS words containing 16 FP32 values.
- Existing tile-major Q/K/V/O weight layout.
- Q/K/V fused so the hidden buffer and each input pack are reused.
- QKV output tile 64 and output-projection tile 32.
- Eight FP32 K lanes in QKV and eight-by-eight output-projection lanes.
- Pair-query Attention Core with both heads of a group unrolled.
- Key-major weighted sum with eight recurrence banks and 16 context lanes.
- QKV caches one complete 64x768 weight tile in URAM and uses row-local banked
  partial sums; the old full sequence accumulator is removed.
- Output projection processes eight outputs per flattened row/block iteration,
  replacing the unroutable 32-output v4 datapath.
- URAM for full hidden/context matrices and BRAM for local tiles.
- Head-group processing; no full attention probability tensor.
- Accurate `hls::expf` softmax and `hls::rsqrtf` layer norm.

## Stream contracts

Q/K/V streams contain six consecutive head groups. Within each group the order
is `[sequence][group_pack]`, 1,024 words per group and 6,144 words per stream.

`context_stream` contains six consecutive head-group chunks. Each chunk is
`[sequence][group_pack]` and has 1,024 words. This synchronized SLR1->SLR2
override lets Output projection accumulate each group while Core computes the
next group instead of serializing two full-tensor stages.

`attn_mid_stream` remains row-major `[sequence][hidden_pack]`, 6,144 words, so
the FFN input contract is unchanged.

## Stage tokens

Port names and argument positions remain compatible with the link config and
host schedule. The clean baseline performs only bounded single reads/writes;
there is no polling loop or wait stage in an Attention kernel.

## Build

`scripts/run_vitis_attention_csynth.tcl` adds exactly one source file to each
project and synthesizes all three kernels independently at 3.333 ns.
