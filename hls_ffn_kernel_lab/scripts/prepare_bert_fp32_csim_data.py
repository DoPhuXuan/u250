#!/usr/bin/env python3
"""Export one real BERT layer-0 FFN row in V21 tile-major binary layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("build/fp32_csim_data"))
    return parser.parse_args()


def gelu_pq8zero(values: np.ndarray) -> np.ndarray:
    coefficients = (
        (-4.0, 0.001065739883, 0.004102702461),
        (-3.0, 0.01860162772, 0.05455128354),
        (-2.0, 0.1245729792, 0.260966163),
        (-1.0, 0.3164173761, 0.4679412082),
        (0.0, 0.3164173761, 0.5320587918),
        (1.0, 0.1245729792, 0.739033837),
        (2.0, 0.01860162772, 0.9454487165),
        (3.0, 0.001065739883, 0.9958972975),
    )
    result = np.where(values < 0.0, 0.0, values).astype(np.float32)
    for lower, a, b in coefficients:
        selected = (values >= lower) & (values < lower + 1.0)
        result[selected] = values[selected] * (np.float32(a) * values[selected] + np.float32(b))
    return result


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(values, dtype="<f4").tobytes())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = BertModel.from_pretrained(args.model, local_files_only=True).eval().cpu()
    batch = tokenizer("Hardware acceleration can reduce neural network inference latency.", return_tensors="pt")
    captured = {}

    def capture(module, module_input):
        captured["input"] = module_input[0].detach()[0, 0].cpu().numpy().copy()

    hook = model.encoder.layer[0].intermediate.register_forward_pre_hook(capture)
    with torch.no_grad():
        model(**batch)
    hook.remove()

    layer = model.encoder.layer[0]
    x = captured["input"].astype(np.float32)
    w1 = layer.intermediate.dense.weight.detach().cpu().numpy().astype(np.float32)
    b1 = layer.intermediate.dense.bias.detach().cpu().numpy().astype(np.float32)
    w2 = layer.output.dense.weight.detach().cpu().numpy().astype(np.float32)
    b2 = layer.output.dense.bias.detach().cpu().numpy().astype(np.float32)

    # V21 W1: [output_tile16][input][output_lane].
    w1_tilemajor = w1.reshape(3072 // 16, 16, 768).transpose(0, 2, 1).copy()
    # V21 W2: [input_tile16][output_tile32][input_lane][output_lane].
    w2_tilemajor = w2.reshape(768 // 32, 32, 3072 // 16, 16).transpose(2, 0, 3, 1).copy()
    pre_gelu = x @ w1.T + b1
    exact_gelu = torch.nn.functional.gelu(torch.from_numpy(pre_gelu), approximate="none").numpy()
    exact_output = exact_gelu @ w2.T + b2
    pq8_output = gelu_pq8zero(pre_gelu) @ w2.T + b2

    payloads = {
        "input.bin": x,
        "w1_tilemajor.bin": w1_tilemajor,
        "b1.bin": b1,
        "w2_tilemajor.bin": w2_tilemajor,
        "b2.bin": b2,
        "exact_output.bin": exact_output,
        "pq8_output.bin": pq8_output,
    }
    for name, values in payloads.items():
        write_f32(args.output_dir / name, values)
    np.savetxt(args.output_dir / "exact_output.txt", exact_output, fmt="%.9g")
    np.savetxt(args.output_dir / "pq8_output.txt", pq8_output, fmt="%.9g")
    manifest = {name: {"floats": int(np.asarray(values).size), "bytes": int(np.asarray(values).size * 4)} for name, values in payloads.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
