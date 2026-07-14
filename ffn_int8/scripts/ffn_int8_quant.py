"""Shared, bit-oriented quantization helpers for the FFN INT8 flow.

The HLS contract is W8A8 symmetric quantization with zero point 0,
round-to-nearest-even, saturation to [-127, 127], and INT32 accumulation.
Weights are represented logically as [output_channel, input_channel].
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


QMIN = -127
QMAX = 127


def round_to_nearest_even(values: np.ndarray | float) -> np.ndarray:
    """Round using IEEE ties-to-even semantics."""

    return np.rint(np.asarray(values, dtype=np.float64))


def symmetric_scale(values: np.ndarray, percentile: float = 100.0) -> float:
    """Return a static per-tensor symmetric scale for signed INT8."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("cannot calibrate an empty tensor")
    if not np.all(np.isfinite(array)):
        raise ValueError("calibration tensor contains NaN or infinity")
    bound = float(np.percentile(np.abs(array).astype(np.float64), percentile))
    return bound / QMAX if bound > 0.0 else 1.0


def quantize_symmetric(values: np.ndarray, scale: float) -> np.ndarray:
    """Quantize a tensor to the hardware's signed INT8 domain."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    rounded = round_to_nearest_even(np.asarray(values, dtype=np.float64) / scale)
    return np.clip(rounded, QMIN, QMAX).astype(np.int8)


def quantize_weights_per_output(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize [output, input] weights with one symmetric scale per output."""

    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must have shape [output, input]")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contain NaN or infinity")
    bounds = np.max(np.abs(weights), axis=1, initial=0.0).astype(np.float64)
    scales = np.where(bounds > 0.0, bounds / QMAX, 1.0).astype(np.float32)
    quantized = quantize_symmetric(weights / scales[:, None], 1.0)
    return quantized, scales


def quantize_bias(bias: np.ndarray, activation_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    """Quantize FP32 bias into each output channel's accumulator domain."""

    bias = np.asarray(bias, dtype=np.float64)
    weight_scales = np.asarray(weight_scales, dtype=np.float64)
    if bias.ndim != 1 or weight_scales.ndim != 1 or bias.shape != weight_scales.shape:
        raise ValueError("bias and weight_scales must be equal-length vectors")
    denominator = activation_scale * weight_scales
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("bias scale must be finite and positive")
    rounded = round_to_nearest_even(bias / denominator)
    info = np.iinfo(np.int32)
    if np.any(rounded < info.min) or np.any(rounded > info.max):
        raise OverflowError("quantized bias does not fit in INT32")
    return rounded.astype(np.int32)


def gelu_pwl(values: np.ndarray) -> np.ndarray:
    """FP32 reference for the exact three-segment V21 GELU approximation."""

    values = np.asarray(values, dtype=np.float32)
    gate = np.float32(0.5) + values * np.float32(1.0 / 6.0)
    middle = values * gate
    return np.where(values <= -3.0, 0.0, np.where(values >= 3.0, values, middle)).astype(np.float32)


def make_gelu_i8_lut(gelu_scale: float) -> np.ndarray:
    """Return G2's 256-entry LUT indexed by a signed INT8 pre-GELU value."""
    if not np.isfinite(gelu_scale) or gelu_scale <= 0.0:
        raise ValueError("gelu_scale must be finite and positive")
    inputs = np.arange(-128, 128, dtype=np.float64) * float(gelu_scale)
    return quantize_symmetric(gelu_pwl(inputs), gelu_scale)


def make_requant_params(real_multipliers: np.ndarray) -> np.ndarray:
    """Encode a positive multiplier as signed-18-bit mantissa plus right shift."""
    values = np.asarray(real_multipliers, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("requant multipliers must be finite and positive")
    mantissa, exponent = np.frexp(values)
    quantized = np.rint(mantissa * (1 << 17)).astype(np.int64)
    carry = quantized == (1 << 17)
    quantized[carry] >>= 1
    exponent = exponent.astype(np.int64) + carry.astype(np.int64)
    right_shift = 17 - exponent
    if np.any(quantized < (1 << 16)) or np.any(quantized >= (1 << 17)):
        raise OverflowError("normalized requant mantissa does not fit signed 18 bits")
    if np.any(right_shift < 0) or np.any(right_shift >= 45):
        raise OverflowError("requant right shift does not fit the 27x18 DSP implementation")
    return (quantized.astype(np.uint32) | (right_shift.astype(np.uint32) << 18)).astype(np.uint32)


def apply_requant_params(accumulators: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Software model of the G2 DSP requantizer, including signed RNE."""
    accumulators = np.asarray(accumulators, dtype=np.int64)
    parameters = np.asarray(parameters, dtype=np.uint32)
    multiplier = (parameters & np.uint32((1 << 18) - 1)).astype(np.int64)
    right_shift = ((parameters >> np.uint32(18)) & np.uint32(0x3F)).astype(np.int64)
    product = accumulators * multiplier
    magnitude = np.abs(product)
    quotient = np.right_shift(magnitude, right_shift)
    mask = np.left_shift(np.int64(1), right_shift) - 1
    remainder = magnitude & mask
    half = np.where(right_shift == 0, 0, np.left_shift(np.int64(1), right_shift - 1))
    increment = (right_shift != 0) & (
        (remainder > half) | ((remainder == half) & ((quotient & 1) != 0))
    )
    rounded = quotient + increment.astype(np.int64)
    rounded = np.where(product < 0, -rounded, rounded)
    return np.clip(rounded, -127, 127).astype(np.int8)


def saturation_rate(values: np.ndarray, scale: float) -> float:
    """Fraction that reaches either INT8 endpoint after quantization."""

    q = quantize_symmetric(values, scale)
    return float(np.mean((q == QMIN) | (q == QMAX)))


def matmul_i8_i32(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output_tile: int = 64,
) -> np.ndarray:
    """INT8 x INT8 -> INT32 golden matmul, tiled to bound host memory."""

    inputs = np.asarray(inputs, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.int8)
    bias = np.asarray(bias, dtype=np.int32)
    if inputs.ndim != 2 or weights.ndim != 2:
        raise ValueError("inputs and weights must be matrices")
    if inputs.shape[1] != weights.shape[1] or weights.shape[0] != bias.shape[0]:
        raise ValueError("incompatible matmul or bias shapes")
    result = np.empty((inputs.shape[0], weights.shape[0]), dtype=np.int32)
    inputs_i32 = inputs.astype(np.int32)
    for start in range(0, weights.shape[0], output_tile):
        stop = min(start + output_tile, weights.shape[0])
        partial = inputs_i32 @ weights[start:stop].astype(np.int32).T
        partial += bias[start:stop]
        result[:, start:stop] = partial
    return result


def quantized_ffn_golden(
    inputs: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    input_scale: float,
    gelu_scale: float,
    gelu_profile: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the mixed-precision G0 or calibrated 256-entry G2 FFN reference."""

    if gelu_profile not in (0, 2):
        raise ValueError("gelu_profile must be 0 or 2")

    inputs = np.asarray(inputs, dtype=np.float32)
    original_shape = inputs.shape[:-1]
    flat = inputs.reshape(-1, inputs.shape[-1])
    w1q, sw1 = quantize_weights_per_output(w1)
    w2q, sw2 = quantize_weights_per_output(w2)
    b1q = quantize_bias(b1, input_scale, sw1)
    b2q = quantize_bias(b2, gelu_scale, sw2)
    xq = quantize_symmetric(flat, input_scale)
    acc1 = matmul_i8_i32(xq, w1q, b1q)
    pre_gelu = acc1.astype(np.float32) * (np.float32(input_scale) * sw1[None, :])
    gelu = gelu_pwl(pre_gelu)
    if gelu_profile == 2:
        parameters = make_requant_params(
            np.asarray(input_scale * sw1 / gelu_scale, dtype=np.float64)
        )
        pre_gelu_q = apply_requant_params(acc1, parameters[None, :])
        lut = make_gelu_i8_lut(gelu_scale)
        gq = lut[pre_gelu_q.astype(np.int16) + 128]
    else:
        gq = quantize_symmetric(gelu, gelu_scale)
    acc2 = matmul_i8_i32(gq, w2q, b2q)
    output = acc2.astype(np.float32) * (np.float32(gelu_scale) * sw2[None, :])
    details = {
        "input_saturation": saturation_rate(flat, input_scale),
        "gelu_saturation": saturation_rate(gelu, gelu_scale),
        "max_abs_acc1": int(np.max(np.abs(acc1.astype(np.int64)), initial=0)),
        "max_abs_acc2": int(np.max(np.abs(acc2.astype(np.int64)), initial=0)),
        "gelu_profile": gelu_profile,
    }
    return output.reshape(*original_shape, output.shape[-1]), details


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return tensor-level metrics without claiming full-model accuracy."""

    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("reference and candidate must be equal, non-empty shapes")
    error = candidate - reference
    denom = np.linalg.norm(reference) * np.linalg.norm(candidate)
    cosine = float(np.dot(reference, candidate) / denom) if denom else float(reference.size == 1 and reference[0] == candidate[0])
    rms_reference = float(np.sqrt(np.mean(reference * reference)))
    return {
        "cosine_similarity": cosine,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "normalized_rmse": float(np.sqrt(np.mean(error * error)) / max(rms_reference, np.finfo(np.float64).tiny)),
        "max_absolute_error": float(np.max(np.abs(error))),
    }


def pack_weights_tile_major(weights_q: np.ndarray, k_par: int, out_par: int) -> np.ndarray:
    """Pack [out, k] as [out_tile][k_tile][k_lane][out_lane]."""

    weights_q = np.asarray(weights_q, dtype=np.int8)
    if weights_q.ndim != 2:
        raise ValueError("weights_q must have shape [output, input]")
    outputs, inputs = weights_q.shape
    if k_par <= 0 or out_par <= 0 or (k_par * out_par) % 64:
        raise ValueError("each compute tile must occupy an integer number of 512-bit words")
    if outputs % out_par or inputs % k_par:
        raise ValueError("weight dimensions must be divisible by out_par and k_par")
    tiled = weights_q.reshape(outputs // out_par, out_par, inputs // k_par, k_par)
    packed = tiled.transpose(0, 2, 3, 1).copy().reshape(-1)
    if packed.nbytes % 64:
        raise AssertionError("tile-major payload is not aligned to a 512-bit AXI word")
    return packed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
