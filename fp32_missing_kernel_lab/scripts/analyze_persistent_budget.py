#!/usr/bin/env python3
"""Gate persistent-kernel CSYNTH resources and a conservative 12-layer budget."""

from __future__ import annotations

import sys
from pathlib import Path

from analyze_csynth_budget import (
    BASELINE,
    CURRENT,
    ROOT,
    gate,
    print_metrics,
    read_iteration_latency,
    read_metrics,
    slr_row,
)


FULL_REPORTS = [
    ("QKV12 / SLR0", "csynth_full_qkv_12layer.rpt", "bert_qkv_12layer_kernel"),
    ("Core12 / SLR1", "csynth_full_attn_core_12layer.rpt", "bert_attn_core_12layer_kernel"),
    ("Out12 / SLR2", "csynth_full_attn_out_12layer.rpt", "bert_attn_out_norm_12layer_kernel"),
    ("UP12 / SLR2", "csynth_full_ffn_up_12layer.rpt", "bert_ffn_up_gelu_12layer_kernel"),
    ("DOWN12 / SLR3", "csynth_full_ffn_down_12layer.rpt", "bert_ffn_down_norm_feedback_12layer_kernel"),
]


def main() -> int:
    missing = [CURRENT / name for _, name, _ in FULL_REPORTS if not (CURRENT / name).exists()]
    if missing:
        print("Missing persistent CSYNTH reports:")
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}")
        print("Run: FP32_KERNELS=full vitis_hls -f scripts/run_vitis_fp32_csynth.tcl")
        return 2

    full = []
    print("PERSISTENT 12-LAYER CSYNTH")
    print("==========================")
    for label, filename, top in FULL_REPORTS:
        path = CURRENT / filename
        metric = read_metrics(path, top)
        full.append(metric)
        print_metrics(label, metric, path)

    qkv, core, out, up, down = full
    checks = []
    print("\nCLOCK/RESOURCE SANITY")
    print("=====================")
    for label, metric in zip(("QKV12", "Core12", "Out12", "UP12", "DOWN12"), full):
        checks.append(gate(f"{label} HLS slack", metric.slack, -0.05, ">="))

    capacities = {
        "SLR0": {"LUT": 420_000, "FF": 840_000, "BRAM": 668, "URAM": 312, "DSP": 3_032},
        "SLR1": {"LUT": 205_000, "FF": 411_000, "BRAM": 384, "URAM": 128, "DSP": 1_536},
        "SLR2": {"LUT": 407_000, "FF": 815_000, "BRAM": 660, "URAM": 308, "DSP": 2_994},
        "SLR3": {"LUT": 424_000, "FF": 849_000, "BRAM": 672, "URAM": 320, "DSP": 3_072},
    }
    old_qkv = read_metrics(BASELINE / "csynth_qkv.rpt", "bert_qkv_kernel")
    old_core = read_metrics(BASELINE / "csynth_attn_core.rpt", "bert_attn_core_kernel")
    # Preserve the locked RTL evidence for SLR0/1 and add only any resource
    # growth introduced by the persistent wrapper relative to old CSYNTH.
    slr0 = {
        "LUT": 131_396 + max(0, qkv.lut - old_qkv.lut),
        "FF": 230_591 + max(0, qkv.ff - old_qkv.ff),
        "BRAM": 54.5 + max(0, qkv.bram18 - old_qkv.bram18) / 2,
        "URAM": 216 + max(0, qkv.uram - old_qkv.uram),
        "DSP": 960 + max(0, qkv.dsp - old_qkv.dsp),
    }
    slr1 = {
        "LUT": 70_459 + max(0, core.lut - old_core.lut),
        "FF": 126_926 + max(0, core.ff - old_core.ff),
        "BRAM": 1.5 + max(0, core.bram18 - old_core.bram18) / 2,
        "URAM": 24 + max(0, core.uram - old_core.uram),
        "DSP": 185 + max(0, core.dsp - old_core.dsp),
    }
    slr2 = {"LUT": 35_401 + out.lut + up.lut, "FF": 56_483 + out.ff + up.ff,
            "BRAM": 78 + out.bram18 / 2 + up.bram18 / 2,
            "URAM": out.uram + up.uram, "DSP": 3 + out.dsp + up.dsp}
    slr3 = {"LUT": 28_227 + down.lut, "FF": 41_481 + down.ff,
            "BRAM": 53.5 + down.bram18 / 2, "URAM": down.uram, "DSP": 3 + down.dsp}
    print("\nPROJECTED PER-SLR UTILIZATION")
    print("=============================")
    for name, values in (("SLR0", slr0), ("SLR1", slr1), ("SLR2", slr2), ("SLR3", slr3)):
        checks.append(slr_row(name, values, capacities[name]))

    # Use the accepted one-layer schedule, then add an intentionally
    # conservative full transfer for every new residual and feedback boundary.
    q_path = BASELINE / "csynth_qkv.rpt"
    c_path = BASELINE / "csynth_attn_core.rpt"
    o_path = CURRENT / "csynth_attn_out_norm_residual.rpt"
    up_path = BASELINE / "csynth_up_v21.rpt"
    down_path = CURRENT / "csynth_ffn_down_residual_norm_fp32.rpt"
    old_out = read_metrics(o_path, "bert_attn_out_norm_residual_kernel")
    old_up = read_metrics(up_path, "bert_ffn_up_gelu_v21_dotpipe_kernel")
    old_down = read_metrics(down_path, "bert_ffn_down_residual_norm_fp32_kernel")
    q_group = read_iteration_latency(q_path, "HEAD_GROUP")
    c_group = read_iteration_latency(c_path, "ATTENTION_GROUP")
    o_group = read_iteration_latency(o_path, "PROJECT_CONTEXT_GROUP")
    norm_tail = old_out.latency - 6 * o_group
    attention = q_group + c_group + o_group + 5 * max(q_group, c_group, o_group) + norm_tail
    layer = attention + max(old_up.latency, old_down.latency)
    base_cycles = 12 * layer
    residual_guard = 12 * 6144
    feedback_guard = 11 * 6144
    transition_guard = 12 * 128
    projected_cycles = base_cycles + residual_guard + feedback_guard + transition_guard
    projected_ms = projected_cycles / 300_000.0

    print("\nPERSISTENT LATENCY BUDGET")
    print("=========================")
    print(f"Accepted one-layer schedule: {layer:,} cycles")
    print(f"Base 12-layer cycles:        {base_cycles:,}")
    print(f"Residual serialization guard:{residual_guard:9,d}")
    print(f"Feedback serialization guard:{feedback_guard:9,d}")
    print(f"Transition guard:            {transition_guard:9,d}")
    print(f"Conservative total:          {projected_cycles:,} cycles = {projected_ms:.3f} ms")
    checks.append(gate("Persistent model latency at 300 MHz (ms)", projected_ms, 200.0))
    minimum_mhz = projected_cycles / 200_000.0
    print(f"Minimum post-route clock for 200 ms: {minimum_mhz:.3f} MHz")
    print("Final acceptance still requires linked hardware timing and device profiling.")

    print("\nOVERALL:", "PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
