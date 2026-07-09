# V19 Decision Report

Pass: `V19_postroute_interface_dataplacement`
Date: 2026-07-08

## Baseline Decision

Baseline restored to v15:

```text
bert_ffn_up_gelu_v15_tilemajor_uram_bindop_kernel
bert_ffn_down_v15_tilemajor_uram_bindop_kernel
```

Default `run_vitis_ffn_kernel.tcl` now builds v15. `ffn_kernel_u250.cfg` also
maps the v15 UP kernel to SLR2 and the v15 DOWN kernel to SLR3.

V18 is rolled back as default. It remains in the source only as a reference
experiment because staged load/store increased latency and did not remove the
borderline HLS timing issue.

## V15 vs V18 HLS Evidence

Reports used:

```text
csynth_up_v15.rpt    Wed Jul  8 13:39:44 2026
csynth_down_v15.rpt  Wed Jul  8 13:40:24 2026
csynth_up_v18.rpt    Wed Jul  8 13:54:54 2026
csynth_down_v18.rpt  Wed Jul  8 13:55:32 2026
```

| Version | Kernel | Latency | Interval | Hot MAC II | LUT | FF | BRAM | URAM | DSP | HLS timing | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| v15 | UP/GELU | 1,501,064 | 1,501,065 | 1 | 95,238 | 115,221 | 302 | 32 | 1,280 | -0.00 | Baseline; tile-major W1; no stride/clobber warning found |
| v18 | UP/GELU | 1,660,040 | 1,660,041 | 1 | 95,341 | 115,249 | 310 | 32 | 1,280 | -0.00 | +158,976 cycles, +10.59%; staged W1 did not improve top timing |
| v15 | DOWN | 1,586,139 | 1,586,140 | 1 | 98,451 | 145,884 | 76 | 32 | 1,280 | -0.00 | Baseline; tile-major W2; no stride/clobber warning found |
| v18 | DOWN | 1,787,910 | 1,787,911 | 1 | 98,663 | 145,855 | 100 | 32 | 1,280 | -0.00 | +201,771 cycles, +12.72%; BRAM +24; staged output store is much slower |

Key observations:

- V15 already removes the old tile-major stride/clobber warnings while keeping
  hot MAC loops at II=1.
- V18 keeps MAC II=1, but the staged load/unpack/store work increases the loop
  body latency. UP hidden-tile latency rises from 7,500 to 8,328 cycles.
- V18 DOWN store latency rises from the v15 `v10_store_output_matrix` 6,155
  cycles to `v18_store_output_matrix_staged` 14,337 cycles.
- HLS top timing remains `-0.00` in both v15 and v18, so V18 is not a valid
  replacement for the baseline.

## Post-Route Diagnosis Status

Post-route reports are now available under `_x/reports/link/imp/`:

```text
impl_1_hw_bb_locked_timing_summary_routed.rpt
impl_1_full_util_routed.rpt
impl_1_kernel_util_routed.rpt
impl_1_slr_util_routed.rpt
```

Current status:

| Item | Status |
|---|---|
| WNS/TNS | WNS `-0.033 ns`, TNS `-5.784 ns`, 336 setup failing endpoints |
| Top 10 timing paths | mostly `up1` FP multiply in `v15_up_k_chunk_v15_up_seq`; secondary paths in `down1` FP multiply |
| AXI wrapper involvement | not in top failing paths |
| AXIS crossing involvement | not in top failing paths |
| URAM/BRAM path involvement | not in top failing paths |
| DSP/MAC involvement | confirmed |
| SLR crossing involvement | SLL utilization is modest; clock path crosses SLR to SLR2/SLR3, but data path is local DSP/fmul |
| DDR bank locality issue | not in top failing paths |

## Next-Pass Decision

Start `V20_compute_margin`.

The next experiment should target the FP multiply/MAC path, especially UP:

1. Keep V15 tile-major layout, `K_PAR=16`, SLR mapping, feeder, and stream
   topology unchanged.
2. First try increasing/binding floating-point multiply latency for the UP hot
   loop only, so the fmul DSP result path gets one more register boundary.
3. Rebuild UP/DOWN/link and check whether WNS crosses positive with small
   latency impact.
4. Apply the same idea to DOWN only if the next routed report still shows DOWN
   fmul paths in the failing set.
5. Do not run V19a/V19b now; AXI and bias ports are not the routed bottleneck.
