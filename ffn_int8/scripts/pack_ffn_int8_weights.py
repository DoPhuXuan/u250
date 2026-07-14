#!/usr/bin/env python3
"""Quantize and pack FFN weights for the HLS tile-major 512-bit interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ffn_int8_quant import (
    make_gelu_i8_lut,
    make_requant_params,
    pack_weights_tile_major,
    quantize_bias,
    quantize_weights_per_output,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="NPZ containing w1, b1, w2, b2")
    parser.add_argument("--scales", type=Path, required=True, help="NPZ from calibrate_ffn_int8.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-par", type=int, default=16)
    parser.add_argument("--out-par", type=int, default=16)
    return parser.parse_args()


def write_array(path: Path, array: np.ndarray, dtype: str) -> None:
    normalized = np.ascontiguousarray(array).astype(np.dtype(dtype).newbyteorder("<"), copy=False)
    path.write_bytes(normalized.tobytes(order="C"))


def main() -> None:
    args = parse_args()
    if (args.k_par, args.out_par) not in {(16, 16), (32, 16), (32, 32), (64, 16)}:
        raise ValueError("profile must be one of 16x16, 32x16, 32x32, or 64x16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.weights, allow_pickle=False) as weights, np.load(args.scales, allow_pickle=False) as scales:
        required_weights = {name: np.asarray(weights[name], dtype=np.float32) for name in ("w1", "b1", "w2", "b2")}
        input_scale = float(np.asarray(scales["input_scale"]).item())
        gelu_scale = float(np.asarray(scales["gelu_scale"]).item())

    expected_shapes = {
        "w1": (3072, 768),
        "b1": (3072,),
        "w2": (768, 3072),
        "b2": (768,),
    }
    for name, expected in expected_shapes.items():
        if required_weights[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {required_weights[name].shape}")

    w1q, sw1 = quantize_weights_per_output(required_weights["w1"])
    w2q, sw2 = quantize_weights_per_output(required_weights["w2"])
    b1q = quantize_bias(required_weights["b1"], input_scale, sw1)
    b2q = quantize_bias(required_weights["b2"], gelu_scale, sw2)
    w1_to_gelu = make_requant_params(
        np.asarray(input_scale * sw1 / gelu_scale, dtype=np.float64)
    )
    gelu_lut = make_gelu_i8_lut(gelu_scale)
    payloads = {
        "w1_i8_tilemajor.bin": (pack_weights_tile_major(w1q, args.k_par, args.out_par), "i1"),
        "b1_i32.bin": (b1q, "i4"),
        "w1_scale_f32.bin": (sw1, "f4"),
        "w1_to_gelu_requant_u32.bin": (w1_to_gelu, "u4"),
        "gelu_i8_lut.bin": (gelu_lut, "i1"),
        "w2_i8_tilemajor.bin": (pack_weights_tile_major(w2q, args.k_par, args.out_par), "i1"),
        "b2_i32.bin": (b2q, "i4"),
        "w2_scale_f32.bin": (sw2, "f4"),
    }
    for name, (array, dtype) in payloads.items():
        write_array(args.output_dir / name, array, dtype)

    theoretical_up = int(required_weights["w1"].shape[1] * 127 * 127)
    theoretical_down = int(required_weights["w2"].shape[1] * 127 * 127)
    max_b1 = int(np.max(np.abs(b1q.astype(np.int64)), initial=0))
    max_b2 = int(np.max(np.abs(b2q.astype(np.int64)), initial=0))
    i32_max = np.iinfo(np.int32).max
    if theoretical_up + max_b1 > i32_max or theoretical_down + max_b2 > i32_max:
        raise OverflowError("worst-case accumulator plus quantized bias exceeds INT32")
    if theoretical_up + max_b1 >= (1 << 26):
        raise OverflowError("UP accumulator does not fit G2's signed 27-bit DSP operand")

    metadata = {
        "layout": "[out_tile][k_tile][k_lane][out_lane]",
        "axi_word_bits": 512,
        "signed_byte_order_in_word": "lane 0 occupies bits [7:0]",
        "k_par": args.k_par,
        "out_par": args.out_par,
        "input_scale": input_scale,
        "input_inv_scale": 1.0 / input_scale,
        "gelu_scale": gelu_scale,
        "gelu_inv_scale": 1.0 / gelu_scale,
        "w1_shape": list(w1q.shape),
        "w2_shape": list(w2q.shape),
        "max_abs_b1q": max_b1,
        "max_abs_b2q": max_b2,
        "worst_case_up_accumulator_bound": theoretical_up + max_b1,
        "worst_case_down_accumulator_bound": theoretical_down + max_b2,
        "files": {},
    }
    for name in payloads:
        path = args.output_dir / name
        metadata["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    (args.output_dir / "packing_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
