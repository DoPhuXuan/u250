# FFN Full Kernel Lab For U250

This lab starts the full FFN optimization path after the micro-kernel sweep.

Current default kernels:

```text
bert_ffn_up_gelu_v21_dotpipe_kernel
bert_ffn_down_v21_dotpipe_kernel
```

Goal:

- Standalone FFN kernel.
- Fused tiled `W1 -> GELU PWL -> W2`.
- No full intermediate tensor written to DDR.
- U250 target part: `xcu250-figd2104-2L-e`.
- Prepare for split-kernel SLR fit and later connectivity mapping.

Versions:

- `bert_ffn_kernel_v1_tiled_i128_h32_o16`: first fused tiled baseline; keeps full `x_buf`, `out_buf`, and `gelu_buf` on chip.
- `bert_ffn_kernel_v2_row_local_i128_h32_o16`: row-local version; keeps only one token row plus tile buffers to reduce BRAM pressure before increasing parallelism.
- `bert_ffn_kernel_v3_token_tile_t32_i128_h32_o16`: token-tile version; reuses W1/W2 over `TILE_T=32` tokens to recover speed while keeping BRAM below the full-sequence v1 design.
- `bert_ffn_kernel_v4_token_tile_partial_accum`: token-tile version with partial accumulators for W1 reduction and W2 accumulation to reduce loop-carried dependency II.
- `bert_ffn_kernel_v5_token_tile_t64_i128_h32_o16`: v3 schedule with `TILE_T_V5=64`; intended to reduce weight reload overhead while staying below v1 BRAM.
- `bert_ffn_kernel_v6_flat_wide_t64_i128_h32_o16`: v5 schedule with flat 1D DDR pointers and fixed `WIDE_FACTOR=16` load/store chunks to encourage 512-bit auto port widening.
- `bert_ffn_kernel_v7_packed512_t64_i128_h32_o16`: v5 schedule with explicit `ap_uint<512>` packed memory ports. Each DDR word contains 16 IEEE-754 float values, following row-major order.
- `bert_ffn_kernel_v8_packed512_w1word_t64_i128_h32_o16`: v7 packed interface, but keeps the W1 tile as packed 512-bit words on chip. This removes the 16-lane unpack write bottleneck in W1 load without adding a second partition dimension to the float W1 tile.
- `bert_ffn_kernel_v9_packed512_w1vec_outpart_t64_i128_h32_o16`: v8 packed W1 load plus a per-neuron W1 float vector to avoid dynamic 512-bit lane extraction inside compute. It also banks the W2 output accumulation into four partial accumulators to target the `update_output_i` II bottleneck.
- `bert_ffn_kernel_v10_stream_tile32_k64_p16_m16`: split-kernel-inspired FFN schedule. It uses 32-wide FFN/output tiles, 64-wide K tiles, `K_PAR=16`, `MAC_GROUP=16`, and an internal 512-bit GELU stream from W1/GELU producer to W2 consumer. This avoids full intermediate DDR and avoids the v7/v9 large `TILE_I=128` schedule.
- `bert_ffn_up_gelu_v11_kernel`: real split-kernel boundary for `W1 -> GELU`; input and output are 512-bit AXIS streams, weights/bias stay on `m_axi`.
- `bert_ffn_down_v11_kernel`: real split-kernel boundary for `W2`; input is the GELU AXIS stream and output is packed DDR.
- `bert_ffn_kernel_v11_split_estimate`: single-top latency estimate for the whole FFN path. It keeps the v10 internal dataflow so one `csynth.rpt` still reports end-to-end FFN latency while the link draft uses two real kernels for SLR placement.
- `bert_ffn_up_gelu_v12_uram_bindop_kernel`: v11 split up/GELU kernel with the full-sequence input buffer moved to URAM and the 16-lane dot-product float ops explicitly bound with extra pipeline latency. Parallelism stays `K_PAR=16`, `MAC_GROUP=16`.
- `bert_ffn_down_v12_uram_bindop_kernel`: v11 split down kernel with the full-sequence output accumulation buffer moved to URAM and the same float bind-op timing margin.
- `bert_ffn_kernel_v12_uram_bindop_estimate`: single-top total FFN latency estimate for v12.
- `bert_ffn_up_gelu_v13_tile16_uram_bindop_kernel`: v12 plus 16-wide intermediate tile. It keeps `K_PAR=16` and `MAC_GROUP=16`, so the per-cycle MAC parallelism stays the same while local `acc` and W tile routing pressure drop. The current version also removes avoidable packed-index arithmetic/switch-based unpack logic and gives the FP add tree one extra pipeline stage for timing margin.
- `bert_ffn_down_v13_tile16_uram_bindop_kernel`: v12 down kernel with 16-wide FFN activation tiles and 32-wide hidden output tiles. It keeps `K_PAR=16`, `MAC_GROUP=16`, uses the same v13 timing-margin dot tree, and keeps the URAM output accumulator.
- `bert_ffn_kernel_v13_tile16_uram_bindop_estimate`: single-top total FFN latency estimate for v13.
- `bert_ffn_up_gelu_v14_slr2_tile32_mac8_uram_kernel`: SLR2 route-fit FFN up/GELU profile. It returns to the v12 32-wide FFN tile to avoid v13 tile overhead, keeps URAM input buffering, and reduces the active MAC group from 16 to 8 outputs per cycle to lower DSP cluster/routing pressure inside one SLR.
- `bert_ffn_down_v14_slr3_tile32_mac8_uram_kernel`: SLR3 route-fit FFN down profile with the same 32-wide tile and 8-output MAC group. It keeps the URAM output accumulator and the packed GELU stream interface.
- `bert_ffn_kernel_v14_slr_split_tile32_mac8_estimate`: failed route-fit experiment. It reduced DSP but broke the hot MAC loop to II=6 and pushed latency to about 14.46M cycles, so it is kept only for reference and should not be the default path.
- `bert_ffn_up_gelu_v15_tilemajor_uram_bindop_kernel`: v13 compute schedule with tile-major W1 layout. It keeps `V13_TILE_I=16`, `V13_TILE_O=32`, `K_PAR=16`, `MAC_GROUP=16`, and URAM input buffering, but makes the W1 load stream contiguous per FFN tile to remove the remaining `Stride is incompatible` timing warning.
- `bert_ffn_down_v15_tilemajor_uram_bindop_kernel`: v13 down schedule with tile-major W2 layout. It keeps the same parallelism and URAM output accumulator while making W2 load contiguous per FFN/output tile for better timing slack without the v14 latency penalty.
- `bert_ffn_kernel_v15_tilemajor_uram_bindop_estimate`: single-top total FFN latency estimate for v15. Use it to check end-to-end latency after the split-kernel reports look healthy.
- `bert_ffn_up_gelu_v16_tilemajor_directpack_kernel`: v15 layout with direct bit-slice unpack/pack in memory loops. It is kept as an experiment to remove helper calls without changing tile size or MAC parallelism.
- `bert_ffn_down_v16_tilemajor_directpack_kernel`: v16 split down kernel with direct bit-slice bias/W2/output handling.
- `bert_ffn_kernel_v16_tilemajor_directpack_estimate`: single-top total FFN latency estimate for v16.
- `bert_ffn_up_gelu_v18_staged_tilemajor_kernel`: rolled-back staged-load experiment. It keeps v15 compute parallelism but adds a BRAM staging boundary for W1 load/unpack. The Jul 8 report shows higher latency without meaningful HLS timing improvement, so it is kept only as a reference.
- `bert_ffn_down_v18_staged_tilemajor_kernel`: rolled-back staged-load/store experiment. It keeps v15 compute parallelism but stages B2, W2, and output store. The Jul 8 report increases latency and BRAM while top timing remains `-0.00`, so it is not the default.
- `bert_ffn_up_gelu_v20_fmul5_kernel`: v15 UP/GELU schedule with the hot
  `fmul maxdsp` latency increased from 4 to 5. It preserves `K_PAR=16`,
  `MAC_GROUP=16`, tile-major storage, and the SLR2 stream interface.
- `bert_ffn_down_v20_fmul5_kernel`: matching v15 DOWN schedule with the same
  one-cycle fmul timing-margin experiment for SLR3.
- `bert_ffn_kernel_v20_fmul5_estimate`: optional single-top total FFN latency
  estimate using the V20 multiply latency.
- `bert_ffn_up_gelu_v21_dotpipe_kernel`: V15 UP/GELU with the 16-lane
  dot-product kept as a separate pipelined RTL function. This prevents
  automatic inlining from dissolving the child `BIND_OP` directives.
- `bert_ffn_down_v21_dotpipe_kernel`: matching DOWN experiment with the same
  separate `II=1` dot-product hierarchy.
- `bert_ffn_kernel_v21_dotpipe_estimate`: optional single-top V21 latency
  estimate.
- `bert_ffn_kernel_v18_staged_tilemajor_estimate`: single-top total FFN latency estimate for the rolled-back v18 experiment.

## Files

```text
bert_ffn_kernel_lab.h
bert_ffn_kernel_hls.c
bert_ffn_kernel_hls_packed.cpp
bert_ffn_kernel_v20.cpp
bert_ffn_kernel_v21.cpp
bert_ffn_kernel_ref.c
test_ffn_kernel.c
bert_ffn_input_feeder_kernel.cpp
run_vitis_ffn_kernel.tcl
collect_ffn_kernel_reports.py
ffn_kernel_u250.cfg
ffn_kernel_u250_v20.cfg
ffn_kernel_u250_v21.cfg
```

## C Smoke Test

Use small shape for C testing:

```powershell
gcc test_ffn_kernel.c bert_ffn_kernel_ref.c bert_ffn_kernel_hls.c -DFFN_KERNEL_TEST_SMALL=1 -std=c99 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label -o test_ffn_kernel -lm
.\test_ffn_kernel
```

Expected:

- `v*_hls_vs_ref_pwl` should be near zero.
- `v*_hls_vs_ref_tanh` reports GELU approximation error.

## Vitis HLS csynth

Default shape is full FFN:

```text
SEQ_LEN = 128
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
TILE_I = 128
TILE_O = 128
```

Run:

```bash
cd /home/minhphat/ffn
export HLS_PART=xcu250-figd2104-2L-e
export HLS_CLOCK_NS=5
vitis_hls -f run_vitis_ffn_kernel.tcl
python3 collect_ffn_kernel_reports.py
```

The default command synthesizes the two v21 dot-pipeline split kernels. The
TCL writes:

```text
proj_bert_ffn_up_gelu_v21_dotpipe_kernel/csynth.rpt
proj_bert_ffn_down_v21_dotpipe_kernel/csynth.rpt
csynth_up_v21.rpt
csynth_down_v21.rpt
csynth.rpt
```

For v21, make sure these files are present in the HLS working directory:

```text
bert_ffn_kernel_v21.cpp
bert_ffn_kernel_hls_packed.cpp
bert_ffn_kernel_lab.h
run_vitis_ffn_kernel.tcl
```

To rebuild the v1 baseline explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v1_tiled_i128_h32_o16
```

To rebuild the v2 row-local experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v2_row_local_i128_h32_o16
```

To rebuild the v3 token-tile experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v3_token_tile_t32_i128_h32_o16
```

To rebuild the v4 partial-accumulator experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v4_token_tile_partial_accum
```

To rebuild the v5 token-tile baseline explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v5_token_tile_t64_i128_h32_o16
```

To rebuild the v6 flat-pointer experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v6_flat_wide_t64_i128_h32_o16
```

To rebuild the v7 packed baseline explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v7_packed512_t64_i128_h32_o16
```

To rebuild the v8 W1 packed-word experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v8_packed512_w1word_t64_i128_h32_o16
```

To rebuild the v9 W1-vector plus output-partial experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v9_packed512_w1vec_outpart_t64_i128_h32_o16
```

To rebuild the v10 split-style stream-tile experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v10_stream_tile32_k64_p16_m16
```

To rebuild the v11 total-latency estimate explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v11_split_estimate
```

To inspect the two real split kernels separately:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_up_gelu_v11_kernel
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_down_v11_kernel
```

To rebuild the v12 timing/resource candidate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_up_gelu_v12_uram_bindop_kernel bert_ffn_down_v12_uram_bindop_kernel
python3 collect_ffn_kernel_reports.py
```

To rebuild the v13 timing/resource candidate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_up_gelu_v13_tile16_uram_bindop_kernel bert_ffn_down_v13_tile16_uram_bindop_kernel
python3 collect_ffn_kernel_reports.py
```

To rebuild the current v15 tile-major baseline:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_up_gelu_v15_tilemajor_uram_bindop_kernel bert_ffn_down_v15_tilemajor_uram_bindop_kernel
python3 collect_ffn_kernel_reports.py
```

To rebuild the v20 compute-margin candidate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl
python3 collect_ffn_kernel_reports.py
```

To rebuild the optional single-top V20 FFN latency estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v20_fmul5_estimate
python3 collect_ffn_kernel_reports.py
```

To rebuild the V21 dot-pipeline candidate at the implementation clock:

```bash
export HLS_CLOCK_NS=3.333
vitis_hls -f run_vitis_ffn_kernel.tcl
python3 collect_ffn_kernel_reports.py
```

To rebuild the optional single-top V21 estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v21_dotpipe_estimate
python3 collect_ffn_kernel_reports.py
```

To rebuild the rolled-back v18 staged experiment explicitly:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_up_gelu_v18_staged_tilemajor_kernel bert_ffn_down_v18_staged_tilemajor_kernel
python3 collect_ffn_kernel_reports.py
```

To rebuild only the v12 total-latency estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v12_uram_bindop_estimate
python3 collect_ffn_kernel_reports.py
```

To rebuild only the v13 total-latency estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v13_tile16_uram_bindop_estimate
python3 collect_ffn_kernel_reports.py
```

To rebuild only the v14 total-latency estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v14_slr_split_tile32_mac8_estimate
python3 collect_ffn_kernel_reports.py
```

To rebuild only the v15 total-latency estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v15_tilemajor_uram_bindop_estimate
python3 collect_ffn_kernel_reports.py
```

To rebuild only the v18 total-latency estimate:

```bash
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_kernel_v18_staged_tilemajor_estimate
python3 collect_ffn_kernel_reports.py
```

On Windows path:

```powershell
cd C:\Users\phudo\Documents\DOCUMENT\DOAN\KLTN\KLTN\hls_ffn_kernel_lab
vitis_hls -f run_vitis_ffn_kernel.tcl
python collect_ffn_kernel_reports.py
```

## Report Targets

Initial pass/fail:

```text
latency_max < old run_ffn_down_to_ddr latency 3,765,893 cycles
DSP <= 2200
BRAM_18K <= 900
URAM <= 240
estimated_clock_ns <= 5
```

For v7/v8, check the M_AXI table first. A useful report should show `512 -> 512` for packed ports. The packed layout is:

```text
input_hidden[(t * HIDDEN_SIZE + h) / 16].lane[h % 16]
w1[(h * INTERMEDIATE_SIZE + i) / 16].lane[i % 16]
w2[(i * HIDDEN_SIZE + h) / 16].lane[h % 16]
b1[i / 16].lane[i % 16]
b2[h / 16].lane[h % 16]
output_hidden[(t * HIDDEN_SIZE + h) / 16].lane[h % 16]
```

V15 keeps the same packed input, bias, stream, and output layout, but expects repacked tile-major weights:

```text
w1_tilemajor[((i_tile * HIDDEN_SIZE + h) * (V13_TILE_I / 16)) + i_pack].lane[lane]
  where i = i_tile * V13_TILE_I + i_pack * 16 + lane

w2_tilemajor[(((i_tile * (HIDDEN_SIZE / V13_TILE_O) + o_tile) * V13_TILE_I + tk) * (V13_TILE_O / 16)) + o_pack].lane[lane]
  where i = i_tile * V13_TILE_I + tk
  where h = o_tile * V13_TILE_O + o_pack * 16 + lane
```

After confirming `512 -> 512`, compare v10 against v7/v8/v9:

- It should no longer expose the old `v*_compute_gelu_t_v*_compute_gelu_i` and `v*_update_output_i` hot spots.
- The hot loops should become `v10_up_mac_group` and `v10_down_mac_group`, both targeting `II=1`.
- Expected DSP is higher than v7 because producer and consumer can run as a dataflow pair, but the design should remain far below the U250 budget if `MAC_GROUP=16`.
- For v11, use `bert_ffn_kernel_v11_split_estimate` to track total latency, then synthesize `bert_ffn_up_gelu_v11_kernel` and `bert_ffn_down_v11_kernel` only when checking per-SLR resource fit.
- For v12, check that the MAC loops keep `II=1`, slack moves clearly positive, and BRAM per split kernel drops because `input_buf`/`output_buf` are mapped to URAM.
- For v13, compare against v12. Expected result: similar latency, lower local BRAM pressure in the producer/consumer core, and better timing margin without reducing `K_PAR=16` or `MAC_GROUP=16`.
- For v14, keep it as a negative control only. Its `MAC_GROUP=8` profile broke the MAC loop to II=6 in the single-top estimate and should not be used for the main FFN path.
- For v15, compare against v13. Expected result: same hot-loop II=1, DSP near 1280, BRAM/URAM near v13, latency close to v13, and fewer or no `Stride is incompatible` warnings in the W1/W2 load loops.
- For v18, first verify the report header names `bert_ffn_up_gelu_v18_staged_tilemajor_kernel` and `bert_ffn_down_v18_staged_tilemajor_kernel`. The Jul 8 report kept hot MAC loops at II=1 and kept stride warnings gone, but latency increased by about 10.59% for UP and 12.72% for DOWN while top timing stayed `-0.00`. Keep it as a reference only unless post-route evidence proves the extra staging fixes a real implementation path.

## Vitis Link Connectivity Draft

Use `ffn_kernel_u250_v20.cfg` for the v20 implementation:

```bash
v++ -c -t hw --platform "$PLATFORM" \
  -k bert_ffn_input_v15_feeder_kernel \
  -o bert_ffn_input_v15.xo bert_ffn_input_feeder_kernel.cpp

v++ -c -t hw --platform "$PLATFORM" \
  -k bert_ffn_up_gelu_v20_fmul5_kernel \
  -o bert_ffn_up_v20.xo bert_ffn_kernel_v20.cpp

v++ -c -t hw --platform "$PLATFORM" \
  -k bert_ffn_down_v20_fmul5_kernel \
  -o bert_ffn_down_v20.xo bert_ffn_kernel_v20.cpp

v++ -l -t hw --platform "$PLATFORM" \
  --config ffn_kernel_u250_v20.cfg \
  -o bert_ffn_v20_u250.xclbin \
  bert_ffn_input_v15.xo bert_ffn_up_v20.xo bert_ffn_down_v20.xo
```

For V21, compile `bert_ffn_kernel_v21.cpp` and link with
`ffn_kernel_u250_v21.cfg` after both split reports pass the V21 acceptance
gate.

The draft maps:

```text
bert_ffn_input_v15_feeder_kernel -> SLR2, DDR[2] input stream source
bert_ffn_up_gelu_1 -> SLR2, DDR[2] weights
bert_ffn_down_1    -> SLR3, DDR[3] weights, DDR[0] output
gelu_stream        -> direct AXIS link, depth 512
```

The U250 connectivity files use short compute-unit aliases `in1`, `up1`, and `down1`
because Vitis limits `kernel_name:compute_unit_name` to 64 characters during
`v++ --link`.

For FFN-only implementation, compile the small input feeder too. It reads packed
input rows from DDR and drives `up1.attn_mid_stream`; the full attention system
will later replace this feeder with the real attention output stream.

## Post-Route Reports

`run_vivado_postroute_reports.tcl` is for Vivado after full implementation, not
for HLS csynth:

```tcl
source run_vivado_postroute_reports.tcl
```

Do not run it as `./run_vivado_postroute_reports.tcl` and do not run it with
`vitis_hls -f`. The report commands in that file are Vivado implementation
commands, so they are only valid after the routed design is open.
