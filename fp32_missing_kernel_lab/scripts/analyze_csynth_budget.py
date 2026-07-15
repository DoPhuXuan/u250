#!/usr/bin/env python3
"""Summarize HLS reports and enforce the FP32 one-layer/200 ms gates."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "reports" / "current"
BASELINE = ROOT / "reports" / "baseline"


@dataclass(frozen=True)
class Metrics:
    top: str
    latency: int
    slack: float
    bram18: int
    dsp: int
    ff: int
    lut: int
    uram: int


def integer(cell: str) -> int:
    match = re.search(r"-?\d+", cell.replace(",", ""))
    if not match:
        return 0
    return int(match.group(0))


def read_metrics(path: Path, top: str) -> Metrics:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 15 or cells[1].lstrip("+").strip() != top:
            continue
        return Metrics(
            top=top,
            slack=float(cells[3]),
            latency=integer(cells[4]),
            bram18=integer(cells[10]),
            dsp=integer(cells[11]),
            ff=integer(cells[12]),
            lut=integer(cells[13]),
            uram=integer(cells[14]),
        )
    raise RuntimeError(f"top row for {top} was not found in {path}")


def read_iteration_latency(path: Path, loop_name: str) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) >= 7 and cells[1] == f"o {loop_name}":
            value = integer(cells[6])
            if value > 0:
                return value
    raise RuntimeError(f"iteration latency for {loop_name} was not found in {path}")


def read_loop_interval(path: Path, loop_name: str) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 8 or not cells[1].startswith("o "):
            continue
        reported_name = cells[1][2:].strip()
        if reported_name == loop_name or reported_name.endswith(loop_name):
            value = integer(cells[7])
            if value > 0:
                return value
    raise RuntimeError(f"interval for {loop_name} was not found in {path}")


def choose(current_name: str, baseline_name: str) -> tuple[Path, bool]:
    current = CURRENT / current_name
    if current.exists():
        return current, True
    return BASELINE / baseline_name, False


def percent(value: float, capacity: float) -> float:
    return 100.0 * value / capacity


def print_metrics(label: str, metrics: Metrics, source: Path) -> None:
    print(
        f"{label:21s} latency={metrics.latency:9,d}  slack={metrics.slack:+.2f}  "
        f"BRAM18={metrics.bram18:4d} DSP={metrics.dsp:4d} "
        f"FF={metrics.ff:7,d} LUT={metrics.lut:7,d} URAM={metrics.uram:3d}"
    )
    print(f"  report: {source.relative_to(ROOT)}")


def gate(name: str, actual: float, limit: float, relation: str = "<=") -> bool:
    passed = actual <= limit if relation == "<=" else actual >= limit
    result = "PASS" if passed else "FAIL"
    print(f"  {result:4s} {name}: {actual:g} {relation} {limit:g}")
    return passed


def slr_row(name: str, values: dict[str, float], capacities: dict[str, float]) -> bool:
    limits = {"LUT": 65.0, "FF": 70.0, "BRAM": 65.0, "URAM": 70.0, "DSP": 70.0}
    ratios = {key: percent(values[key], capacities[key]) for key in values}
    rendered = "  ".join(f"{key}={ratios[key]:5.1f}%" for key in ("LUT", "FF", "BRAM", "URAM", "DSP"))
    passed = all(ratios[key] <= limits[key] for key in ratios)
    print(f"{name}: {rendered}  {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    qkv_path, _ = choose("csynth_qkv.rpt", "csynth_qkv.rpt")
    core_path, _ = choose("csynth_attn_core.rpt", "csynth_attn_core.rpt")
    out_path, out_is_current = choose(
        "csynth_attn_out_norm_residual.rpt", "csynth_attn_out_norm.rpt"
    )
    up_path, _ = choose("csynth_ffn_up_fp32.rpt", "csynth_up_v21.rpt")
    down_path, down_is_current = choose(
        "csynth_ffn_down_residual_norm_fp32.rpt", "csynth_down_v21.rpt"
    )

    qkv = read_metrics(qkv_path, "bert_qkv_kernel")
    core = read_metrics(core_path, "bert_attn_core_kernel")
    out_top = (
        "bert_attn_out_norm_residual_kernel"
        if out_is_current
        else "bert_attn_out_norm_kernel"
    )
    out = read_metrics(out_path, out_top)
    up = read_metrics(up_path, "bert_ffn_up_gelu_v21_dotpipe_kernel")
    down_top = (
        "bert_ffn_down_residual_norm_fp32_kernel"
        if down_is_current
        else "bert_ffn_down_v21_dotpipe_kernel"
    )
    down = read_metrics(down_path, down_top)

    print("FP32 CSYNTH SUMMARY")
    print("====================")
    print_metrics("QKV / SLR0", qkv, qkv_path)
    print_metrics("Attention Core / SLR1", core, core_path)
    print_metrics("Out+Norm+fork / SLR2", out, out_path)
    print_metrics("FFN UP / SLR2", up, up_path)
    print_metrics("FFN DOWN+LN / SLR3", down, down_path)

    if not out_is_current or not down_is_current:
        print("\nINCOMPLETE: changed-kernel reports are missing.")
        if not out_is_current:
            print("  - reports/current/csynth_attn_out_norm_residual.rpt")
        if not down_is_current:
            print("  - reports/current/csynth_ffn_down_residual_norm_fp32.rpt")
        print("Run vitis_hls -f scripts/run_vitis_fp32_csynth.tcl first.")
        return 2

    print("\nCHANGED-KERNEL GATES")
    print("====================")
    checks = []
    checks += [
        gate("Out/Norm/fork latency", out.latency, 1_370_000),
        gate("Out/Norm/fork HLS slack", out.slack, -0.05, ">="),
        gate("Out/Norm/fork DSP", out.dsp, 410),
        gate("Out/Norm/fork FF", out.ff, 115_000),
        gate("Out/Norm/fork LUT", out.lut, 105_000),
        gate("Out/Norm/fork URAM", out.uram, 32),
    ]
    checks += [
        gate("Fused DOWN latency", down.latency, 1_725_260),
        gate("Fused DOWN HLS slack", down.slack, -0.05, ">="),
        gate("Fused DOWN DSP", down.dsp, 1_040),
        gate("Fused DOWN FF", down.ff, 235_000),
        gate("Fused DOWN LUT", down.lut, 121_000),
        gate("Fused DOWN BRAM18", down.bram18, 48),
        gate("Fused DOWN URAM", down.uram, 32),
    ]
    checks += [
        gate(
            "Out projection ACCUMULATE_ROW_BLOCK II",
            read_loop_interval(out_path, "ACCUMULATE_ROW_BLOCK"),
            2,
        ),
        gate(
            "Out NORM_WRITE_PACK II",
            read_loop_interval(out_path, "NORM_WRITE_PACK"),
            1,
        ),
        gate(
            "DOWN hot MAC II",
            read_loop_interval(down_path, "v15_down_seq_v15_down_mac_group"),
            1,
        ),
        gate(
            "DOWN residual drain II",
            read_loop_interval(down_path, "fp32_down_residual_pack"),
            1,
        ),
        gate(
            "DOWN LayerNorm reduce II",
            read_loop_interval(down_path, "fp32_norm_reduce_pack"),
            4,
        ),
        gate(
            "DOWN LayerNorm write II",
            read_loop_interval(down_path, "fp32_norm_write_pack"),
            1,
        ),
    ]

    q_group = read_iteration_latency(qkv_path, "HEAD_GROUP")
    c_group = read_iteration_latency(core_path, "ATTENTION_GROUP")
    o_group = read_iteration_latency(out_path, "PROJECT_CONTEXT_GROUP")
    norm_tail = out.latency - 6 * o_group
    attention_cycles = q_group + c_group + o_group + 5 * max(q_group, c_group, o_group) + norm_tail
    ffn_cycles = max(up.latency, down.latency)
    layer_cycles = attention_cycles + ffn_cycles
    model_cycles = 12 * layer_cycles
    model_ms = model_cycles / 300_000.0

    print("\nLATENCY BUDGET")
    print("==============")
    print(f"Q/C/O group cycles: {q_group:,} / {c_group:,} / {o_group:,}")
    print(f"Attention/layer:     {attention_cycles:,} cycles")
    print(f"FFN critical stage:  {ffn_cycles:,} cycles")
    print(f"Encoder layer:       {layer_cycles:,} cycles")
    print(f"12 layers @ 300 MHz: {model_cycles:,} cycles = {model_ms:.3f} ms")
    checks += [
        gate("Per-layer cycle budget", layer_cycles, 5_000_000),
        gate("Twelve-layer latency budget (ms)", model_ms, 200.0),
    ]

    # Conservative pre-route projections. SLR0/1 use the locked RTL synthesis
    # evidence. SLR2/3 add standalone HLS resources to platform/routing proxies
    # extracted from the accepted FFN-only routed design. BRAM is in 36K tiles.
    capacities = {
        "SLR0": {"LUT": 420_000, "FF": 840_000, "BRAM": 668, "URAM": 312, "DSP": 3_032},
        "SLR1": {"LUT": 205_000, "FF": 411_000, "BRAM": 384, "URAM": 128, "DSP": 1_536},
        "SLR2": {"LUT": 407_000, "FF": 815_000, "BRAM": 660, "URAM": 308, "DSP": 2_994},
        "SLR3": {"LUT": 424_000, "FF": 849_000, "BRAM": 672, "URAM": 320, "DSP": 3_072},
    }
    slr0 = {"LUT": 131_396, "FF": 230_591, "BRAM": 54.5, "URAM": 216, "DSP": 960}
    slr1 = {"LUT": 70_459, "FF": 126_926, "BRAM": 1.5, "URAM": 24, "DSP": 185}
    slr2 = {
        "LUT": 35_401 + up.lut + out.lut,
        "FF": 56_483 + up.ff + out.ff,
        "BRAM": 78.0 + up.bram18 / 2.0 + max(out.bram18 / 2.0, 53.5),
        "URAM": up.uram + out.uram,
        "DSP": 3 + up.dsp + out.dsp,
    }
    slr3 = {
        "LUT": 28_227 + down.lut,
        "FF": 41_481 + down.ff,
        "BRAM": 53.5 + down.bram18 / 2.0,
        "URAM": down.uram,
        "DSP": 3 + down.dsp,
    }

    print("\nPROJECTED PER-SLR UTILIZATION")
    print("=============================")
    print("Pre-route estimates; final acceptance still requires full-link/post-route.")
    checks += [
        slr_row("SLR0", slr0, capacities["SLR0"]),
        slr_row("SLR1", slr1, capacities["SLR1"]),
        slr_row("SLR2", slr2, capacities["SLR2"]),
        slr_row("SLR3", slr3, capacities["SLR3"]),
    ]

    print("\nOVERALL:", "PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
