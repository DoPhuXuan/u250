# V19 Post-Route Diagnosis

Pass: `V19_postroute_interface_dataplacement`
Date: 2026-07-08
Status: routed reports analyzed.

This file is intentionally post-route driven. Do not use HLS `Timing -0.00`
alone to justify another loop-schedule change.

## Baseline Under Test

Use the restored v15 split kernels:

```text
bert_ffn_up_gelu_v15_tilemajor_uram_bindop_kernel   -> SLR2
bert_ffn_down_v15_tilemajor_uram_bindop_kernel      -> SLR3
```

Keep the current compute profile:

```text
K_PAR = 16
MAC_GROUP = 16
V13_TILE_I = 16
V13_TILE_O = 32
tile-major W1/W2 layout
URAM buffering
```

## Required Reports

The current analysis uses the Vitis-generated routed reports under
`_x/reports/link/imp/`:

```text
impl_1_hw_bb_locked_timing_summary_routed.rpt
impl_1_full_util_routed.rpt
impl_1_kernel_util_routed.rpt
impl_1_slr_util_routed.rpt
```

If extra Vivado reports are needed later, generate them after full
link/implementation, after the routed design is open inside Vivado:

```tcl
source run_vivado_postroute_reports.tcl
```

Do not run this script with bash or `vitis_hls -f`. Bash will treat Tcl syntax
as shell syntax, and Vitis HLS csynth does not provide Vivado implementation
commands such as `report_timing_summary`.

or directly:

```tcl
report_timing_summary -max_paths 50 -file timing_summary_v15.rpt
report_utilization -hierarchical -file utilization_hier_v15.rpt
report_qor_suggestions -file qor_suggestions_v15.rpt
report_design_analysis -congestion -file congestion_v15.rpt
report_clock_utilization -file clock_util_v15.rpt
```

## Timing Summary

| Metric | Value | Evidence |
|---|---:|---|
| WNS | -0.033 ns | `impl_1_hw_bb_locked_timing_summary_routed.rpt`, design timing summary |
| TNS | -5.784 ns | `impl_1_hw_bb_locked_timing_summary_routed.rpt`, design timing summary |
| TNS failing endpoints | 336 | all on `clk_kernel_00_unbuffered_net` |
| WHS | 0.005 ns | no hold violations |
| THS | 0.000 ns | no hold violations |
| Setup status | not met | very small setup miss at 300 MHz |

## Top Critical Paths

| # | Slack | From | To | Clock group | SLR/pblock | Classification | Evidence |
|---:|---:|---|---|---|---|---|---|
| 1 | -0.033 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[17]/R` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | DSP_ALU + DSP_OUTPUT + LUT6, route 65.2% |
| 2 | -0.033 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[21]/R` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | DSP_ALU + DSP_OUTPUT + LUT3/LUT6, route 62.1% |
| 3 | -0.033 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[2]/R` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | DSP_ALU + DSP_OUTPUT + LUT3/LUT6, route 62.3% |
| 4 | -0.030 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[18]/R` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | DSP_ALU + DSP_OUTPUT + LUT6, route 65.2% |
| 5 | -0.030 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[17]/D` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | two DSP_ALU/DSP_OUTPUT stages plus LUT6 |
| 6 | -0.030 ns | `down1/.../v15_down_seq_v15_down_mac_group/.../fmul...DSP_M_DATA_INST/CLK` | `down1/.../fmul...mant_op_reg[22]/D` | `clk_kernel_00_unbuffered_net` | DOWN in SLR3 | DSP/MAC cluster | route 63.7%, secondary bottleneck |
| 7 | -0.028 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[20]/R` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | route 62.7% |
| 8 | -0.025 ns | `down1/.../v15_down_seq_v15_down_mac_group/.../fmul...DSP_M_DATA_INST/CLK` | `down1/.../fmul...exp_op_reg[2]/R` | `clk_kernel_00_unbuffered_net` | DOWN in SLR3 | DSP/MAC cluster | route 65.2%, secondary bottleneck |
| 9 | -0.025 ns | `down1/.../v15_down_seq_v15_down_mac_group/.../fmul...DSP_M_DATA_INST/CLK` | `down1/.../fmul...exp_op_reg[6]/R` | `clk_kernel_00_unbuffered_net` | DOWN in SLR3 | DSP/MAC cluster | route 65.2%, secondary bottleneck |
| 10 | -0.024 ns | `up1/.../v15_up_k_chunk_v15_up_seq/.../fmul...DSP_M_DATA_INST/CLK` | `up1/.../fmul...mant_op_reg[10]/R` | `clk_kernel_00_unbuffered_net` | UP in SLR2 | DSP/MAC cluster | route 65.0% |

## Classification Checklist

| Path class | Current evidence | Next action if confirmed |
|---|---|---|
| AXI or m_axi adapter wrapper | not in top failing paths | Do not run V19a as the next step |
| AXIS stream crossing | not in top failing paths; SLR2->SLR3 SLL use is 1,474 / 23,040 | Do not change stream topology yet |
| URAM/BRAM access path | not in top failing paths | Keep URAM buffering and tile-major layout |
| DSP/MAC cluster | confirmed in UP hot MAC first, DOWN hot MAC second | Start `V20_compute_margin`, prioritize UP |
| Top-level control/fanout | not in top failing paths | No control-interface change needed yet |
| SLR crossing or pblock boundary | clock path shows SLR crossing to SLR2 clock root, but data endpoints are local UP/DOWN DSP paths | Preserve SLR2/SLR3 placement; use compute margin before remap |
| DDR bank/interface locality | not in top failing paths | Keep DDR mapping for now |

## Resource/SLR Notes

Kernel routed utilization:

| CU | LUT | REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| `in1` feeder | 1,716 | 4,432 | 7 | 0 | 0 |
| `up1` V15 | 71,610 | 134,107 | 143 | 32 | 1,280 |
| `down1` V15 | 79,150 | 163,887 | 23 | 32 | 1,280 |

SLR utilization remains reasonable:

| SLR | CLB LUTs | CLB Registers | BRAM Tile | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| SLR2 | 113,545 (26.28%) | 194,519 (22.51%) | 247 (36.76%) | 32 (10.00%) | 1,283 (41.76%) |
| SLR3 | 108,725 (25.17%) | 206,822 (23.94%) | 87 (12.95%) | 32 (10.00%) | 1,283 (41.76%) |

SLL use is not near capacity: SLR2<->SLR3 uses 2,789 / 23,040 (12.11%).

## Decision Gate

Routed evidence now points to the MAC/DSP floating-point multiply path, not the
AXI wrapper. The next pass should be a narrow `V20_compute_margin` experiment:
add timing margin around the UP hot MAC/fmul path first while preserving the
V15 tile-major layout, split-kernel SLR placement, and as much parallelism as
possible. Do not repeat v14 `MAC_GROUP=8` directly.
