from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ffn_int8_quant import (  # noqa: E402
    apply_requant_params,
    error_metrics,
    make_gelu_i8_lut,
    make_requant_params,
    matmul_i8_i32,
    pack_weights_tile_major,
    quantize_bias,
    quantize_symmetric,
    quantized_ffn_golden,
    round_to_nearest_even,
)


class QuantizationContractTests(unittest.TestCase):
    def test_round_to_nearest_even_signed_ties(self) -> None:
        values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
        expected = np.array([-2, -2, 0, 0, 2, 2])
        np.testing.assert_array_equal(round_to_nearest_even(values), expected)

    def test_symmetric_quantization_saturates_without_minus_128(self) -> None:
        values = np.array([-1000.0, -127.0, -1.5, 1.5, 127.0, 1000.0])
        expected = np.array([-127, -127, -2, 2, 127, 127], dtype=np.int8)
        np.testing.assert_array_equal(quantize_symmetric(values, 1.0), expected)

    def test_bias_uses_accumulator_scale_and_ties_to_even(self) -> None:
        bias = np.array([0.5, 1.5, 2.5, -1.5], dtype=np.float32)
        scales = np.ones(4, dtype=np.float32)
        expected = np.array([0, 2, 2, -2], dtype=np.int32)
        np.testing.assert_array_equal(quantize_bias(bias, 1.0, scales), expected)

    def test_tile_major_layout_matches_contract(self) -> None:
        weights = np.arange(16 * 16, dtype=np.int16).astype(np.int8).reshape(16, 16)
        packed = pack_weights_tile_major(weights, k_par=8, out_par=8)
        expected_first_tile = np.array(
            [weights[out_lane, k_lane] for k_lane in range(8) for out_lane in range(8)],
            dtype=np.int8,
        )
        np.testing.assert_array_equal(packed[:64], expected_first_tile)

    def test_tile_major_rejects_partial_axi_word_tiles(self) -> None:
        with self.assertRaises(ValueError):
            pack_weights_tile_major(np.zeros((8, 8), dtype=np.int8), k_par=4, out_par=4)

    def test_int32_matmul_does_not_wrap_like_int8(self) -> None:
        x = np.full((1, 768), 127, dtype=np.int8)
        w = np.full((1, 768), 127, dtype=np.int8)
        result = matmul_i8_i32(x, w, np.array([0], dtype=np.int32))
        self.assertEqual(int(result[0, 0]), 768 * 127 * 127)

    def test_error_metrics_identity(self) -> None:
        metrics = error_metrics(np.array([1.0, -2.0]), np.array([1.0, -2.0]))
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertEqual(metrics["max_absolute_error"], 0.0)

    def test_g2_lut_uses_calibrated_gelu_scale(self) -> None:
        lut = make_gelu_i8_lut(0.25)
        self.assertEqual(lut.shape, (256,))
        self.assertEqual(lut.dtype, np.int8)
        expected = quantize_symmetric(
            np.array([0.0, 0.25, 31.75], dtype=np.float32), 0.25
        )
        np.testing.assert_array_equal(lut[[128, 129, 255]], expected)

    def test_g2_requant_params_preserve_rne_and_scale(self) -> None:
        scales = np.array([1.0, 0.5, 0.125], dtype=np.float64)
        params = make_requant_params(scales)
        accumulators = np.array(
            [[-127, -5, -4], [-2, -1, 0], [1, 2, 5], [17, 31, 127]],
            dtype=np.int64,
        )
        actual = apply_requant_params(accumulators, params)
        expected = np.clip(np.rint(accumulators * scales), -127, 127).astype(np.int8)
        np.testing.assert_array_equal(actual, expected)

    def test_small_quantized_ffn_has_expected_shape_and_int32_bounds(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 0.2, size=(3, 4)).astype(np.float32)
        w1 = rng.normal(0.0, 0.2, size=(8, 4)).astype(np.float32)
        b1 = rng.normal(0.0, 0.01, size=8).astype(np.float32)
        w2 = rng.normal(0.0, 0.2, size=(4, 8)).astype(np.float32)
        b2 = rng.normal(0.0, 0.01, size=4).astype(np.float32)
        output, details = quantized_ffn_golden(x, w1, b1, w2, b2, 0.01, 0.01)
        self.assertEqual(output.shape, (3, 4))
        self.assertLess(details["max_abs_acc1"], np.iinfo(np.int32).max)
        self.assertLess(details["max_abs_acc2"], np.iinfo(np.int32).max)

        output_g2, details_g2 = quantized_ffn_golden(
            x, w1, b1, w2, b2, 0.01, 0.01, gelu_profile=2
        )
        self.assertEqual(output_g2.shape, output.shape)
        self.assertEqual(details_g2["gelu_profile"], 2)


if __name__ == "__main__":
    unittest.main()
