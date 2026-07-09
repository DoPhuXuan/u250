# FFN V20 Compute-Margin Pass

## Change

V20 keeps the V15 split-kernel architecture, tile-major memory layout,
`K_PAR=16`, `MAC_GROUP=16`, URAM buffers, stream depth, DDR mapping, and SLR
placement. The only compute change is:

```text
fmul maxdsp latency: 4 -> 5
```

`bert_ffn_kernel_v20.cpp` enables this change and includes the V15
implementation. Building V15 directly from `bert_ffn_kernel_hls_packed.cpp`
still uses latency 4.

## Why

The V15 routed design misses 300 MHz by only 0.033 ns. Its top failing setup
paths are internal floating-point multiplier/DSP paths in both UP and DOWN.
AXI, AXIS, URAM, BRAM, DDR locality, and SLR capacity are not the top failing
paths.

## Acceptance Gate

Compare `csynth_up_v20.rpt` and `csynth_down_v20.rpt` against V15:

1. Both hot MAC loops must keep final `II=1`.
2. DSP parallelism must remain near 1,280 DSP per kernel.
3. Kernel latency increase should be small; reject if it resembles the
   10-13% V18 regression.
4. No new stride or clobber warnings.
5. Final acceptance requires routed `WNS >= 0` at 300 MHz, not only a positive
   HLS estimate.

If V20 UP passes route but DOWN still fails, retain V20 UP and make the next
DOWN-only adjustment. If V20 introduces a material II or latency regression,
restore V15 and use implementation-level physical optimization rather than
reducing `K_PAR` or `MAC_GROUP`.
