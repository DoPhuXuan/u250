#!/usr/bin/env python3
"""Evaluate the FP32 V21 three-segment GELU through all 12 BERT layers."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, BertModel


SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hardware acceleration can reduce neural network inference latency.",
    "Quantization maps floating point tensors into compact integer domains.",
    "A balanced pipeline must preserve both throughput and numerical accuracy.",
    "Transformers use attention layers followed by feed forward networks.",
    "The engineer measured timing, resource utilization, and output error.",
    "Reliable experiments require deterministic inputs and reproducible reports.",
    "Integer matrix multiplication is efficient on modern FPGA DSP blocks.",
    "Calibration estimates activation ranges from representative model inputs.",
    "Cosine similarity measures whether two output vectors point in similar directions.",
    "The final implementation keeps residual connections and normalization in floating point.",
    "Routing congestion can limit frequency even when synthesis estimates look acceptable.",
    "Per channel weight scales usually preserve more accuracy than one global scale.",
    "The sequence contains words of different lengths and semantic contexts.",
    "Testing corner cases prevents signed arithmetic errors from reaching the bitstream.",
    "This evaluation compares the optimized kernel with the original feed forward output.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--sentences", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--gelu-mode", choices=("v21", "pq8", "pq8zero", "tanh"), default="v21")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/accuracy/bert_full_model_fp32_v21_pwl.json"),
    )
    return parser.parse_args()


def gelu_v21(values: torch.Tensor) -> torch.Tensor:
    middle = values * (torch.tensor(0.5, dtype=values.dtype) + values * torch.tensor(1.0 / 6.0, dtype=values.dtype))
    return torch.where(values <= -3.0, torch.zeros_like(values), torch.where(values >= 3.0, values, middle))


PQ8_COEFFICIENTS = (
    (-4.0, -0.005006682249, -0.0385207216, -0.07429150066),
    (-3.0, -0.03705479525, -0.2252352114, -0.3470252458),
    (-2.0, -0.008670650083, -0.1449150694, -0.2982606566),
    (-1.0, 0.3009219056, 0.4493461787, -0.004648873598),
    (0.0, 0.3009219056, 0.5506538213, -0.004648873598),
    (1.0, -0.008670650083, 1.144915069, -0.2982606566),
    (2.0, -0.03705479525, 1.225235211, -0.3470252458),
    (3.0, -0.005006682249, 1.038520722, -0.07429150066),
)

PQ8_ZERO_COEFFICIENTS = (
    (-4.0, 0.001065739883, 0.004102702461),
    (-3.0, 0.01860162772, 0.05455128354),
    (-2.0, 0.1245729792, 0.260966163),
    (-1.0, 0.3164173761, 0.4679412082),
    (0.0, 0.3164173761, 0.5320587918),
    (1.0, 0.1245729792, 0.739033837),
    (2.0, 0.01860162772, 0.9454487165),
    (3.0, 0.001065739883, 0.9958972975),
)


def gelu_pq8(values: torch.Tensor) -> torch.Tensor:
    result = torch.where(values < 0.0, torch.zeros_like(values), values)
    for lower, a, b, c in PQ8_COEFFICIENTS:
        polynomial = (values * a + b) * values + c
        result = torch.where((values >= lower) & (values < lower + 1.0), polynomial, result)
    return result


def gelu_pq8zero(values: torch.Tensor) -> torch.Tensor:
    result = torch.where(values < 0.0, torch.zeros_like(values), values)
    for lower, a, b in PQ8_ZERO_COEFFICIENTS:
        polynomial = values * (values * a + b)
        result = torch.where((values >= lower) & (values < lower + 1.0), polynomial, result)
    return result


def candidate_gelu(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "v21":
        return gelu_v21(values)
    if mode == "pq8":
        return gelu_pq8(values)
    if mode == "pq8zero":
        return gelu_pq8zero(values)
    return torch.nn.functional.gelu(values, approximate="tanh")


def tensor_cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    return float(np.dot(reference, candidate) / (np.linalg.norm(reference) * np.linalg.norm(candidate)))


def row_cosine(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    numerator = np.sum(reference * candidate, axis=-1)
    denominator = np.linalg.norm(reference, axis=-1) * np.linalg.norm(candidate, axis=-1)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return {
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "p05": float(np.percentile(values, 5.0)),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.sentences <= len(SENTENCES):
        raise ValueError(f"sentences must be in [1, {len(SENTENCES)}]")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = BertModel.from_pretrained(args.model, local_files_only=True).eval().cpu()
    batch = tokenizer(
        SENTENCES[: args.sentences], return_tensors="pt", padding=True,
        truncation=True, max_length=args.max_length,
    )
    valid = batch["attention_mask"].bool().reshape(-1)
    captured: list[np.ndarray | None] = [None] * len(model.encoder.layer)
    hooks = []
    for layer_index, layer in enumerate(model.encoder.layer):
        def capture(module, module_input, module_output, index=layer_index):
            flat = module_input[0].detach().reshape(-1, module_input[0].shape[-1])
            captured[index] = flat[valid].cpu().numpy().copy()
        hooks.append(layer.intermediate.register_forward_hook(capture))

    with torch.no_grad():
        original = model(**batch, output_hidden_states=True)
    for hook in hooks:
        hook.remove()

    independent_ffn = []
    for index, layer in enumerate(model.encoder.layer):
        inputs = torch.from_numpy(captured[index])
        pre_gelu = layer.intermediate.dense(inputs)
        exact_output = layer.output.dense(layer.intermediate.intermediate_act_fn(pre_gelu))
        v21_output = layer.output.dense(candidate_gelu(pre_gelu, args.gelu_mode))
        exact_np = exact_output.numpy()
        v21_np = v21_output.numpy()
        cosine = tensor_cosine(exact_np, v21_np)
        independent_ffn.append({
            "layer": index,
            "global_cosine": cosine,
            "cosine_loss": 1.0 - cosine,
            "row_cosine": row_cosine(exact_np, v21_np),
            "maximum_absolute_error": float(np.max(np.abs(v21_np - exact_np))),
        })

    for layer in model.encoder.layer:
        def feed_forward_chunk(this, attention_output):
            pre_gelu = this.intermediate.dense(attention_output)
            v21_gelu = candidate_gelu(pre_gelu, args.gelu_mode)
            ffn_output = this.output.dense(v21_gelu)
            ffn_output = this.output.dropout(ffn_output)
            return this.output.LayerNorm(ffn_output + attention_output)
        layer.feed_forward_chunk = types.MethodType(feed_forward_chunk, layer)

    with torch.no_grad():
        candidate = model(**batch, output_hidden_states=True)

    reference_hidden = original.last_hidden_state.reshape(-1, 768)[valid].numpy()
    candidate_hidden = candidate.last_hidden_state.reshape(-1, 768)[valid].numpy()
    reference_cls = original.last_hidden_state[:, 0, :].numpy()
    candidate_cls = candidate.last_hidden_state[:, 0, :].numpy()
    final_cosine = tensor_cosine(reference_hidden, candidate_hidden)
    layer_propagation = []
    for layer_index in range(1, len(original.hidden_states)):
        reference_layer = original.hidden_states[layer_index].reshape(-1, 768)[valid].numpy()
        candidate_layer = candidate.hidden_states[layer_index].reshape(-1, 768)[valid].numpy()
        cosine = tensor_cosine(reference_layer, candidate_layer)
        layer_propagation.append({
            "layer": layer_index - 1,
            "global_cosine": cosine,
            "cosine_loss": 1.0 - cosine,
            "token_cosine": row_cosine(reference_layer, candidate_layer),
        })

    result = {
        "model": args.model,
        "source_contract": "FP32 weights/activations with selectable GELU approximation",
        "gelu_mode": args.gelu_mode,
        "sentences": SENTENCES[: args.sentences],
        "valid_tokens": int(valid.sum().item()),
        "criterion": "1 - final_hidden_global_cosine <= 0.01",
        "independent_ffn_cosine": independent_ffn,
        "encoder_layer_cosine": layer_propagation,
        "final_hidden_global_cosine": final_cosine,
        "final_hidden_cosine_loss": 1.0 - final_cosine,
        "final_hidden_token_cosine": row_cosine(reference_hidden, candidate_hidden),
        "cls_global_cosine": tensor_cosine(reference_cls, candidate_cls),
        "cls_row_cosine": row_cosine(reference_cls, candidate_cls),
        "passes_one_percent_cosine_loss": bool(1.0 - final_cosine <= 0.01),
        "maximum_absolute_error": float(np.max(np.abs(candidate_hidden - reference_hidden))),
        "scope": "BERT hidden-state comparison; not a downstream task accuracy/F1 guarantee",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
