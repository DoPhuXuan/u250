# FFN W8A8 optimization branch

This directory is the separate INT8 development path derived from the accepted
FP32 V21 architecture. It does not modify
`../hls_ffn_kernel_lab/src/bert_ffn_kernel_v21.cpp`.

## Implemented contract

- W1/W2: signed symmetric INT8, zero point 0, per-output-channel scale.
- Activations: signed symmetric INT8, static per-tensor scale, range
  `[-127, 127]`.
- Bias and accumulation: signed INT32.
- Rounding: round-to-nearest-even; overflow: saturation.
- Input and output boundary: FP32.
- UP to DOWN stream: 512 bits containing 64 INT8 values. The complete
  `[128][3072]` tensor is exactly 6,144 stream beats.
- Weight order: `[out_tile][k_tile][k_lane][out_lane]`, with byte lane 0 in
  bits `[7:0]` of each 512-bit word.

The `v1` correctness profile `16x16`, I8-ROUTE profile `32x16`, and unpacked
I8-BALANCED profile `32x32` have passed U250 csynth with hot-loop `II=1`.
The packed `32x32` G0 result is 439,087 cycles / 1,024 DSP for UP and 409,240
cycles / 608 DSP for DOWN. Both hot loops remain at `II=1`; DOWN has reached the
DSP target. The synthesis Tcl now keeps the same `dsp2` MAC and defaults to G2,
which requantizes the UP accumulator once and uses a calibrated 256-entry GELU
LUT. Generated project metadata carries the `dsp2_g2` suffix.

The first G2 report reaches 437,441 cycles / 624 DSP for UP and 409,240 cycles /
608 DSP for DOWN. The next `ovl1` pass double-buffers UP weight tiles and overlaps
the next 16-word tile load with the current 128-sequence MAC; all arithmetic,
parallelism, stream layout, and G2 parameters remain unchanged.

`ovl1` reaches 373,219 cycles but HLS instantiates a second 512-DSP MAC engine
for the final K tile. `ovl2` runs that final tile through the same dataflow task
and issues a harmless dummy prefetch, retaining overlap without the duplicate
MAC engine.

`ovl2` reaches 373,315 cycles / 624 DSP, but the nested MAC function interface
raises UP to 232,342 FF and 192,006 LUT. `ovl3` inlines only that wrapper into
the dataflow region to remove interface duplication without changing the loop,
DSP mapping, or overlap schedule.

Vitis produces the same hardware for `ovl3` as `ovl2`, so the inline pragma is
neutral. Since DOWN now limits pipeline throughput at 409,240 cycles, `down1`
applies the corrected single-engine ping-pong schedule to its two K-subtiles per
stream/output tile. UP arithmetic and scheduling remain unchanged.

`down1` reaches 383,896 cycles / 608 DSP, but dataflow raises it to 284,409 FF
and 194,716 LUT. `fuse1` removes UP's dataflow region and prefetches one next-tile
AXI word during each of the first 16 sequence iterations of the existing MAC
pipeline. This targets the same overlap latency without dataflow PIPO overhead.

`fuse1` keeps `II=1` and 624 DSP while reducing UP from 232,342 to 153,418 FF,
but its non-inlined wrapper adds ten cycles per K tile and raises latency to
396,353 cycles. `fuse2` inlines only that fused wrapper to remove the repeated
call overhead while preserving the same prefetch/MAC loop.

`fuse2` causes the two explicit ping/pong branches to synthesize as two 512-DSP
MAC engines. `fuse3` replaces them with one completely partitioned two-bank
weight array and one MAC call-site selected by the K-tile parity bit, retaining
the inline fused loop without duplicating the engine.

`fuse3` flattens K-tile/sequence scheduling to 343,169 cycles, but dynamic bank
indexing at the dot-function boundary still creates two 512-DSP datapaths.
`fuse4` explicitly muxes each bank into local partitioned weight vectors before
the single dot call, targeting the same flattened schedule with one MAC engine.

`fuse4` reaches 343,265 cycles / 624 DSP with 172,601 FF and 186,568 LUT.
`down2` applies the same in-loop prefetch and explicit pre-mux structure to DOWN,
replacing `down1` dataflow while retaining its packed 512-DSP MAC contract.

## Calibration input

Prepare a weights NPZ:

```text
w1 [3072, 768] float32
b1 [3072]      float32
w2 [768, 3072] float32
b2 [768]       float32
```

Prepare a representative activation NPZ (512--2,048 task samples, normal model
preprocessing):

```text
ffn_input [..., 768]  float32
post_gelu [..., 3072] float32
```

Then run:

```powershell
python scripts/calibrate_ffn_int8.py `
  --weights data/ffn_weights.npz `
  --activations data/ffn_calibration.npz `
  --output-dir artifacts/calibration

python scripts/pack_ffn_int8_weights.py `
  --weights data/ffn_weights.npz `
  --scales artifacts/calibration/activation_scales.npz `
  --output-dir artifacts/packed_k16_o16 `
  --k-par 16 --out-par 16
```

The calibration report is a local FFN tensor comparison, not a substitute for
12-layer hidden-state and task accuracy/F1 evaluation.

## Functional tests

Run the bit-contract tests before any large synthesis:

```powershell
python -m unittest discover -s tests -v
```

They cover signed ties-to-even, saturation, INT32 accumulation, quantized bias,
tile-major packing, exhaustive per-lane signed INT8 DSP-product recovery, and
simultaneous two-lane randomized DSP packing cases.

Run the reduced-dimension HLS csim to exercise both real kernel tops and compare
every packed GELU byte plus every DOWN FP32 output against an independent INT32
golden implementation:

```powershell
vitis_hls -f scripts/run_vitis_ffn_int8_csim.tcl
```

## Csynth

With Vitis HLS available:

```powershell
$env:HLS_PART = 'xcu250-figd2104-2L-e'
$env:HLS_CLOCK_NS = '3.333'
$env:FFN_I8_K_PAR = '32'
$env:FFN_I8_OUT_PAR = '32'
$env:FFN_I8_GELU_PROFILE = '2'
vitis_hls -f scripts/run_vitis_ffn_int8.tcl
```

For source/scheduling checks on an UltraScale+ device without URAM, set
`FFN_I8_USE_BRAM=1`. This diagnostic mode must not be used for U250 resource or
timing claims.

The Tcl script exports only these two synthesis reports:

```text
reports/int8/csynth_up_int8.rpt
reports/int8/csynth_down_int8.rpt
```

No XML, per-module report tree, or RTL synthesis report is copied outside the
disposable HLS project under `build/`.

## Architecture notes

UP buffers the quantized `[128][768]` input once and computes subtiles that form
each 64-channel stream tile. G0 applies the V21 FP32 PWL; G2 uses per-output
calibrated 27x18 requantization followed by replicated BRAM LUT lookup. DOWN consumes the same
`[intermediate_tile][sequence]` order, accumulates all 3,072 products in INT32,
and dequantizes to packed FP32 at the output boundary.

SLR mapping remains unchanged: UP is on SLR2 and DOWN is on SLR3. MAC packing
uses 17-bit separation and signed-borrow correction. G2's LUT and normalized
requant parameters are generated from calibration scales by the packing script;
G0 remains selectable with `FFN_I8_GELU_PROFILE=0` for accuracy/resource A/B.
Deeper dataflow and full-model boundary quantization remain deferred.
