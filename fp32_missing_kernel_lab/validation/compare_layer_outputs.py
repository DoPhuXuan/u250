#!/usr/bin/env python3
"""Compare 12-layer HLS C-sim tensors with the pinned Hugging Face golden data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


LAYERS = 12
SEQ_LEN = 128
HIDDEN = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--csim-output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--max-rmse", type=float, default=0.15)
    parser.add_argument("--max-abs", type=float, default=1.0)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument(
        "--gate-scope", choices=("valid", "all", "both"), default="both"
    )
    return parser.parse_args()


def load_tensor(path: Path) -> np.ndarray:
    expected = SEQ_LEN * HIDDEN
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected:
        raise RuntimeError(f"{path}: {values.size} floats, expected {expected}")
    return values.reshape(SEQ_LEN, HIDDEN)


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    got = actual.astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(got)
    nonfinite = int((~finite).sum())
    if nonfinite:
        return {
            "count": int(got.size), "nonfinite": nonfinite,
            "mae": math.inf, "rmse": math.inf, "max_abs": math.inf,
            "p95_abs": math.inf, "p99_abs": math.inf,
            "relative_l2": math.inf, "cosine": -1.0,
        }
    difference = got - ref
    absolute = np.abs(difference)
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    denominator = ref_norm * got_norm
    return {
        "count": int(got.size),
        "nonfinite": 0,
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(absolute.max()),
        "p95_abs": float(np.percentile(absolute, 95)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "relative_l2": float(np.linalg.norm(difference) / max(ref_norm, 1e-30)),
        "cosine": float(np.dot(ref, got) / denominator) if denominator else 1.0,
    }


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.csim_output_dir.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    mask = np.fromfile(data_dir / "input/attention_mask.i32", dtype="<i4")
    if mask.size != SEQ_LEN:
        raise RuntimeError("attention_mask.i32 has the wrong length")
    valid = mask != 0

    rows: list[dict[str, object]] = []
    overall_pass = True
    gated_scopes = {args.gate_scope} if args.gate_scope != "both" else {"valid", "all"}
    for layer in range(LAYERS):
        for stage in ("attention", "encoder"):
            reference = load_tensor(
                data_dir / "golden" / f"{stage}_layer_{layer:02d}.f32"
            )
            actual = load_tensor(
                output_dir / f"{stage}_layer_{layer:02d}.f32"
            )
            for scope, selector in (("valid", valid), ("all", np.ones(SEQ_LEN, bool))):
                result = metrics(reference[selector], actual[selector])
                passed = (
                    result["nonfinite"] == 0
                    and result["rmse"] <= args.max_rmse
                    and result["max_abs"] <= args.max_abs
                    and result["cosine"] >= args.min_cosine
                )
                gated = scope in gated_scopes
                if gated:
                    overall_pass = overall_pass and passed
                rows.append({
                    "layer": layer,
                    "stage": stage,
                    "scope": scope,
                    **result,
                    "gated": gated,
                    "pass": passed,
                })

    csv_path = report_dir / "accuracy_by_layer.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "model_id": metadata["model_id"],
        "resolved_commit": metadata["resolved_commit"],
        "active_tokens": metadata["input"]["active_tokens"],
        "thresholds": {
            "max_rmse": args.max_rmse,
            "max_abs": args.max_abs,
            "min_cosine": args.min_cosine,
            "gate_scope": args.gate_scope,
        },
        "rows": rows,
    }
    (report_dir / "accuracy_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    markdown = [
        "# BERT FP32 12-layer C-sim accuracy",
        "",
        f"**Overall: {report['status']}**",
        "",
        f"- Model: `{metadata['model_id']}`",
        f"- Checkpoint commit: `{metadata['resolved_commit']}`",
        f"- Active tokens: {metadata['input']['active_tokens']}/{SEQ_LEN}",
        f"- Gates: RMSE <= {args.max_rmse}, max abs <= {args.max_abs}, "
        f"cosine >= {args.min_cosine}, scope={args.gate_scope}",
        "",
        "| Layer | Stage | Scope | RMSE | Max abs | P99 abs | Cosine | Result |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['layer']} | {row['stage']} | {row['scope']} | "
            f"{row['rmse']:.6g} | {row['max_abs']:.6g} | "
            f"{row['p99_abs']:.6g} | {row['cosine']:.8f} | "
            f"{'PASS' if row['pass'] else 'FAIL'}{'*' if row['gated'] else ''} |"
        )
    markdown += [
        "",
        "`*` indicates a row included in the overall gate.",
        "",
        "The Hugging Face reference uses exact erf-based GELU. The HLS kernel "
        "uses its documented eight-segment quadratic FP32 GELU approximation, "
        "so bit-exact equality is not expected.",
        "",
    ]
    (report_dir / "accuracy_report.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )

    print(f"Accuracy report: {report_dir / 'accuracy_report.md'}")
    print(f"OVERALL: {report['status']}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

