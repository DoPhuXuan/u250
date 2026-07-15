#!/usr/bin/env python3
"""Compare text output produced by HLS CSim with original BERT FFN output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("build/fp32_csim_data"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/accuracy/fp32_v21_real_bert_csim.json"),
    )
    parser.add_argument("--minimum-cosine", type=float, default=0.99)
    return parser.parse_args()


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.shape != candidate.shape or reference.size != 768:
        raise ValueError(f"expected equal 768-element outputs, got {reference.shape} and {candidate.shape}")
    error = candidate - reference
    cosine = float(np.dot(reference, candidate) / (np.linalg.norm(reference) * np.linalg.norm(candidate)))
    return {
        "cosine_similarity": cosine,
        "cosine_loss": 1.0 - cosine,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def main() -> None:
    args = parse_args()
    exact_path = args.data_dir / "exact_output.txt"
    csim_path = args.data_dir / "csim_output.txt"
    exact = np.loadtxt(exact_path, dtype=np.float64)
    csim = np.loadtxt(csim_path, dtype=np.float64)
    result = {
        "reference": str(exact_path),
        "candidate": str(csim_path),
        **metrics(exact, csim),
    }
    result["minimum_cosine"] = args.minimum_cosine
    result["pass"] = bool(result["cosine_similarity"] >= args.minimum_cosine)
    result["bit_exact"] = bool(np.array_equal(exact.astype(np.float32), csim.astype(np.float32)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
