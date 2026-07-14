#!/usr/bin/env python3
"""Calibrate static activation scales and validate the W8A8 FFN golden path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ffn_int8_quant import (
    error_metrics,
    gelu_pwl,
    quantized_ffn_golden,
    saturation_rate,
    symmetric_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="NPZ containing w1, b1, w2, b2")
    parser.add_argument("--activations", type=Path, required=True, help="NPZ containing ffn_input and post_gelu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99.99, help="activation clipping percentile")
    parser.add_argument("--max-golden-rows", type=int, default=128, help="rows used for tensor-level golden comparison; 0 disables")
    return parser.parse_args()


def require_array(archive: np.lib.npyio.NpzFile, key: str, ndim: int | None = None) -> np.ndarray:
    if key not in archive:
        raise KeyError(f"missing required NPZ array: {key}")
    value = np.asarray(archive[key], dtype=np.float32)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{key} must have {ndim} dimensions, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains NaN or infinity")
    return value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.weights, allow_pickle=False) as weights, np.load(args.activations, allow_pickle=False) as activations:
        w1 = require_array(weights, "w1", 2)
        b1 = require_array(weights, "b1", 1)
        w2 = require_array(weights, "w2", 2)
        b2 = require_array(weights, "b2", 1)
        ffn_input = require_array(activations, "ffn_input")
        post_gelu = require_array(activations, "post_gelu")

    if w1.shape[0] != b1.size or w2.shape[0] != b2.size or w1.shape[0] != w2.shape[1]:
        raise ValueError("weight/bias dimensions do not form W1 -> GELU -> W2")
    if ffn_input.shape[-1] != w1.shape[1] or post_gelu.shape[-1] != w1.shape[0]:
        raise ValueError("activation dimensions do not match W1")

    input_scale = symmetric_scale(ffn_input, args.percentile)
    gelu_scale = symmetric_scale(post_gelu, args.percentile)
    np.savez(
        args.output_dir / "activation_scales.npz",
        input_scale=np.float32(input_scale),
        gelu_scale=np.float32(gelu_scale),
    )

    summary_rows = [
        {
            "tensor": "ffn_input",
            "scale": input_scale,
            "min": float(np.min(ffn_input)),
            "max": float(np.max(ffn_input)),
            "saturation_rate": saturation_rate(ffn_input, input_scale),
            "samples": int(ffn_input.size),
        },
        {
            "tensor": "post_gelu",
            "scale": gelu_scale,
            "min": float(np.min(post_gelu)),
            "max": float(np.max(post_gelu)),
            "saturation_rate": saturation_rate(post_gelu, gelu_scale),
            "samples": int(post_gelu.size),
        },
    ]
    with (args.output_dir / "quantization_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    golden_metrics: dict[str, float | int] | None = None
    g2_vs_g0_metrics: dict[str, float | int] | None = None
    if args.max_golden_rows > 0:
        flat_input = ffn_input.reshape(-1, ffn_input.shape[-1])[: args.max_golden_rows]
        candidate, details = quantized_ffn_golden(flat_input, w1, b1, w2, b2, input_scale, gelu_scale)
        candidate_g2, details_g2 = quantized_ffn_golden(
            flat_input, w1, b1, w2, b2, input_scale, gelu_scale, gelu_profile=2
        )
        reference = gelu_pwl(flat_input @ w1.T + b1) @ w2.T + b2
        golden_metrics = {**error_metrics(reference, candidate), **details, "rows": int(flat_input.shape[0])}
        g2_vs_g0_metrics = {
            **error_metrics(candidate, candidate_g2),
            **details_g2,
            "rows": int(flat_input.shape[0]),
        }

    metadata = {
        "quantization": "W8A8 symmetric, per-output weight, per-tensor activation",
        "rounding": "round-to-nearest-even",
        "range": [127 * -1, 127],
        "percentile": args.percentile,
        "input_scale": input_scale,
        "gelu_scale": gelu_scale,
        "input_saturation": summary_rows[0]["saturation_rate"],
        "gelu_saturation": summary_rows[1]["saturation_rate"],
        "golden_tensor_metrics": golden_metrics,
        "g2_vs_g0_tensor_metrics": g2_vs_g0_metrics,
        "task_accuracy_measured": False,
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# INT8 calibration summary",
        "",
        f"- Activation clipping percentile: {args.percentile}",
        f"- FFN input scale: {input_scale:.9g}",
        f"- GELU output scale: {gelu_scale:.9g}",
        f"- FFN input saturation: {summary_rows[0]['saturation_rate']:.6%}",
        f"- GELU output saturation: {summary_rows[1]['saturation_rate']:.6%}",
        "- Full-model task accuracy: not measured by this tensor calibration script",
    ]
    if golden_metrics is not None:
        lines.extend(
            [
                "",
                "## Local FFN golden comparison",
                "",
                f"- Rows: {golden_metrics['rows']}",
                f"- Cosine similarity: {golden_metrics['cosine_similarity']:.9f}",
                f"- MAE: {golden_metrics['mae']:.9g}",
                f"- RMSE: {golden_metrics['rmse']:.9g}",
                f"- Normalized RMSE: {golden_metrics['normalized_rmse']:.9g}",
                f"- Max absolute error: {golden_metrics['max_absolute_error']:.9g}",
                f"- Max |UP accumulator|: {golden_metrics['max_abs_acc1']}",
                f"- Max |DOWN accumulator|: {golden_metrics['max_abs_acc2']}",
            ]
        )
    if g2_vs_g0_metrics is not None:
        lines.extend(
            [
                "",
                "## G2 LUT versus G0 tensor comparison",
                "",
                f"- Rows: {g2_vs_g0_metrics['rows']}",
                f"- Cosine similarity: {g2_vs_g0_metrics['cosine_similarity']:.9f}",
                f"- MAE: {g2_vs_g0_metrics['mae']:.9g}",
                f"- RMSE: {g2_vs_g0_metrics['rmse']:.9g}",
                f"- Normalized RMSE: {g2_vs_g0_metrics['normalized_rmse']:.9g}",
                f"- Max absolute error: {g2_vs_g0_metrics['max_absolute_error']:.9g}",
            ]
        )
    (args.output_dir / "accuracy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
