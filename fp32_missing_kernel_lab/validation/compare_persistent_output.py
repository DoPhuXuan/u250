#!/usr/bin/env python3
"""Compare the single-launch persistent encoder output with Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from compare_layer_outputs import HIDDEN, SEQ_LEN, load_tensor, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--csim-output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--max-rmse", type=float, default=0.03)
    parser.add_argument("--max-abs", type=float, default=0.25)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = args.data_dir.resolve()
    output = args.csim_output_dir.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    mask = np.fromfile(data / "input/attention_mask.i32", dtype="<i4")
    if mask.size != SEQ_LEN:
        raise RuntimeError("attention mask length is not 128")
    reference = load_tensor(data / "golden/encoder_layer_11.f32")
    actual = load_tensor(output / "final_encoder_output.f32")

    valid_result = metrics(reference[mask != 0], actual[mask != 0])
    all_result = metrics(reference, actual)
    passed = (
        valid_result["nonfinite"] == 0
        and valid_result["rmse"] <= args.max_rmse
        and valid_result["max_abs"] <= args.max_abs
        and valid_result["cosine"] >= args.min_cosine
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "model_id": metadata["model_id"],
        "resolved_commit": metadata["resolved_commit"],
        "active_tokens": metadata["input"]["active_tokens"],
        "execution": "five persistent CUs, one start each, no host layer loop",
        "thresholds_valid_tokens": {
            "max_rmse": args.max_rmse,
            "max_abs": args.max_abs,
            "min_cosine": args.min_cosine,
        },
        "valid_tokens": valid_result,
        "all_tokens_diagnostic": all_result,
    }
    (report_dir / "persistent_accuracy_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Persistent BERT FP32 12-layer accuracy",
        "",
        f"**Overall: {report['status']}**",
        "",
        f"- Model: `{metadata['model_id']}`",
        f"- Commit: `{metadata['resolved_commit']}`",
        f"- Active tokens: {metadata['input']['active_tokens']}/{SEQ_LEN}",
        "- Execution: five CUs started once; no host-managed layer loop",
        "",
        "| Scope | RMSE | Max abs | P99 abs | Cosine | Gate |",
        "|---|---:|---:|---:|---:|---|",
        f"| valid | {valid_result['rmse']:.7g} | {valid_result['max_abs']:.7g} | "
        f"{valid_result['p99_abs']:.7g} | {valid_result['cosine']:.9f} | "
        f"{'PASS' if passed else 'FAIL'} |",
        f"| all (diagnostic) | {all_result['rmse']:.7g} | "
        f"{all_result['max_abs']:.7g} | {all_result['p99_abs']:.7g} | "
        f"{all_result['cosine']:.9f} | not gated |",
        "",
    ]
    (report_dir / "persistent_accuracy_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Persistent accuracy report: {report_dir / 'persistent_accuracy_report.md'}")
    print(f"OVERALL: {report['status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)

