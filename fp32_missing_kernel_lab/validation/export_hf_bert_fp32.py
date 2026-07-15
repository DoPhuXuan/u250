#!/usr/bin/env python3
"""Export a real Hugging Face BERT-base checkpoint for the HLS C-sim.

The binary files contain little-endian FP32 scalars. Packed files are ordered
exactly as the 16 lanes of each ap_uint<512> word consumed by the kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoTokenizer, BertModel


LAYERS = 12
SEQ_LEN = 128
HIDDEN = 768
INTERMEDIATE = 3072
PACK = 16

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. BERT is a bidirectional "
    "Transformer encoder trained from unlabeled text. This validation vector "
    "checks attention masking, residual connections, layer normalization, and "
    "the feed-forward network with authentic pretrained parameters. The same "
    "tokens and weights are used by the Hugging Face reference and the FPGA "
    "C simulation so numerical differences can be measured layer by layer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default="google-bert/bert-base-uncased",
        help="Hugging Face model repository",
    )
    parser.add_argument(
        "--revision", default="main",
        help="Hub branch, tag, or commit; resolved commit is recorded",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/data/bert_base_uncased_seq128",
    )
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--text", default=None)
    text.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--force", action="store_true",
        help="Replace an existing export directory",
    )
    return parser.parse_args()


def as_f32(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().cpu().float().numpy(), dtype="<f4")


def write_f32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(values, dtype="<f4").tofile(path)


def write_i32(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(values, dtype="<i4").tofile(path)


def pack_qkv(weight: np.ndarray) -> np.ndarray:
    # PyTorch [out, in] -> [out_tile64, input_pack16, local_out, lane].
    return weight.reshape(12, 64, 48, 16).transpose(0, 2, 1, 3).reshape(-1)


def pack_attention_output(weight: np.ndarray) -> np.ndarray:
    # [out, in] -> [head_group128, out_tile32, input_pack16, local_out, lane].
    return (
        weight.reshape(24, 32, 6, 8, 16)
        .transpose(2, 0, 3, 1, 4)
        .reshape(-1)
    )


def pack_ffn_up(weight: np.ndarray) -> np.ndarray:
    # [3072 out, 768 in] -> [out_tile16, input, output_lane].
    return weight.reshape(192, 16, 768).transpose(0, 2, 1).reshape(-1)


def pack_ffn_down(weight: np.ndarray) -> np.ndarray:
    # [768 out, 3072 in] ->
    # [input_tile16, output_tile32, input_lane, output_pack16, output_lane].
    return (
        weight.reshape(24, 2, 16, 192, 16)
        .transpose(3, 0, 4, 1, 2)
        .reshape(-1)
    )


def verify_packers() -> None:
    rng = np.random.default_rng(20260715)

    qkv = rng.standard_normal((HIDDEN, HIDDEN), dtype=np.float32)
    qkv_p = pack_qkv(qkv)
    out = rng.integers(0, HIDDEN, 128)
    ins = rng.integers(0, HIDDEN, 128)
    for o, i in zip(out, ins):
        word = ((o // 64) * 48 + i // 16) * 64 + o % 64
        assert qkv_p[word * 16 + i % 16] == qkv[o, i]

    attn = rng.standard_normal((HIDDEN, HIDDEN), dtype=np.float32)
    attn_p = pack_attention_output(attn)
    for o, i in zip(out, ins):
        word = ((((i // 128) * 24 + o // 32) * 8 + (i % 128) // 16) * 32
                + o % 32)
        assert attn_p[word * 16 + i % 16] == attn[o, i]

    up = rng.standard_normal((INTERMEDIATE, HIDDEN), dtype=np.float32)
    up_p = pack_ffn_up(up)
    outs_up = rng.integers(0, INTERMEDIATE, 128)
    for o, i in zip(outs_up, ins):
        word = (o // 16) * HIDDEN + i
        assert up_p[word * 16 + o % 16] == up[o, i]

    down = rng.standard_normal((HIDDEN, INTERMEDIATE), dtype=np.float32)
    down_p = pack_ffn_down(down)
    ins_down = rng.integers(0, INTERMEDIATE, 128)
    for o, i in zip(out, ins_down):
        word = ((((i // 16) * 24 + o // 32) * 16 + i % 16) * 2
                + (o % 32) // 16)
        assert down_p[word * 16 + o % 16] == down[o, i]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(model: BertModel) -> None:
    cfg = model.config
    expected = {
        "num_hidden_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 12,
    }
    for name, value in expected.items():
        actual = getattr(cfg, name)
        if actual != value:
            raise RuntimeError(f"{name}={actual}, expected {value}")
    if abs(float(cfg.layer_norm_eps) - 1.0e-12) > 1.0e-20:
        raise RuntimeError(
            f"layer_norm_eps={cfg.layer_norm_eps}; kernels require 1e-12"
        )
    if cfg.hidden_act != "gelu":
        raise RuntimeError(f"hidden_act={cfg.hidden_act!r}; expected 'gelu'")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.force:
            raise RuntimeError(
                f"{output_dir} already exists; pass --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    text = DEFAULT_TEXT
    if args.text is not None:
        text = args.text
    elif args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")

    print(f"Resolving {args.model_id}@{args.revision} ...", flush=True)
    model_info = HfApi().model_info(args.model_id, revision=args.revision)
    commit = model_info.sha
    if not commit:
        raise RuntimeError("Hugging Face Hub did not return a commit SHA")
    print(f"Pinned checkpoint commit: {commit}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=commit)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = BertModel.from_pretrained(
        args.model_id,
        revision=commit,
        torch_dtype=torch.float32,
        add_pooling_layer=False,
        attn_implementation="eager",
    )
    model.eval()
    validate_config(model)
    verify_packers()

    encoded = tokenizer(
        text,
        max_length=SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

    attention_outputs: dict[int, torch.Tensor] = {}
    handles = []
    for layer_id, layer in enumerate(model.encoder.layer):
        def capture(_module, _inputs, output, index=layer_id):
            attention_outputs[index] = output.detach().cpu()
        handles.append(layer.attention.output.register_forward_hook(capture))

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()

    if outputs.hidden_states is None or len(outputs.hidden_states) != LAYERS + 1:
        raise RuntimeError("BertModel did not return embedding + 12 hidden states")
    if len(attention_outputs) != LAYERS:
        raise RuntimeError("Failed to capture all 12 Attention outputs")

    input_dir = output_dir / "input"
    golden_dir = output_dir / "golden"
    weights_dir = output_dir / "weights"
    embedding = as_f32(outputs.hidden_states[0][0])
    write_f32(input_dir / "embedding_output.f32", embedding)
    write_i32(input_dir / "attention_mask.i32", encoded["attention_mask"][0].numpy())
    write_i32(input_dir / "input_ids.i32", encoded["input_ids"][0].numpy())
    write_i32(input_dir / "token_type_ids.i32", encoded["token_type_ids"][0].numpy())
    write_f32(golden_dir / "embedding_output.f32", embedding)

    for layer_id in range(LAYERS):
        write_f32(
            golden_dir / f"attention_layer_{layer_id:02d}.f32",
            as_f32(attention_outputs[layer_id][0]),
        )
        write_f32(
            golden_dir / f"encoder_layer_{layer_id:02d}.f32",
            as_f32(outputs.hidden_states[layer_id + 1][0]),
        )

    attention_files = {
        "q_weight": (weights_dir / "attention_q_weight.packed.f32", pack_qkv),
        "k_weight": (weights_dir / "attention_k_weight.packed.f32", pack_qkv),
        "v_weight": (weights_dir / "attention_v_weight.packed.f32", pack_qkv),
        "o_weight": (
            weights_dir / "attention_o_weight.packed.f32",
            pack_attention_output,
        ),
        "q_bias": (weights_dir / "attention_q_bias.packed.f32", None),
        "k_bias": (weights_dir / "attention_k_bias.packed.f32", None),
        "v_bias": (weights_dir / "attention_v_bias.packed.f32", None),
        "o_bias": (weights_dir / "attention_o_bias.packed.f32", None),
        "attn_gamma": (weights_dir / "attention_norm_gamma.packed.f32", None),
        "attn_beta": (weights_dir / "attention_norm_beta.packed.f32", None),
    }
    for path, _packer in attention_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    print("Exporting 12 layers of packed FP32 weights ...", flush=True)
    for layer_id, layer in enumerate(model.encoder.layer):
        tensors = {
            "q_weight": as_f32(layer.attention.self.query.weight),
            "k_weight": as_f32(layer.attention.self.key.weight),
            "v_weight": as_f32(layer.attention.self.value.weight),
            "o_weight": as_f32(layer.attention.output.dense.weight),
            "q_bias": as_f32(layer.attention.self.query.bias),
            "k_bias": as_f32(layer.attention.self.key.bias),
            "v_bias": as_f32(layer.attention.self.value.bias),
            "o_bias": as_f32(layer.attention.output.dense.bias),
            "attn_gamma": as_f32(layer.attention.output.LayerNorm.weight),
            "attn_beta": as_f32(layer.attention.output.LayerNorm.bias),
        }
        for name, tensor in tensors.items():
            path, packer = attention_files[name]
            packed = packer(tensor) if packer is not None else tensor.reshape(-1)
            with path.open("ab") as handle:
                np.ascontiguousarray(packed, dtype="<f4").tofile(handle)

        layer_dir = weights_dir / f"layer_{layer_id:02d}"
        write_f32(
            layer_dir / "ffn_w1.packed.f32",
            pack_ffn_up(as_f32(layer.intermediate.dense.weight)),
        )
        write_f32(layer_dir / "ffn_b1.packed.f32", as_f32(layer.intermediate.dense.bias))
        write_f32(
            layer_dir / "ffn_w2.packed.f32",
            pack_ffn_down(as_f32(layer.output.dense.weight)),
        )
        write_f32(layer_dir / "ffn_b2.packed.f32", as_f32(layer.output.dense.bias))
        write_f32(layer_dir / "ffn_norm_gamma.packed.f32", as_f32(layer.output.LayerNorm.weight))
        write_f32(layer_dir / "ffn_norm_beta.packed.f32", as_f32(layer.output.LayerNorm.bias))
        print(f"  layer {layer_id:02d}: done", flush=True)

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    checksums = {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    input_ids = encoded["input_ids"][0].tolist()
    metadata = {
        "format_version": 1,
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_commit": commit,
        "geometry": {
            "layers": LAYERS,
            "sequence_length": SEQ_LEN,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "attention_heads": 12,
            "pack_float32": PACK,
        },
        "model_config": {
            "hidden_act": model.config.hidden_act,
            "layer_norm_eps": model.config.layer_norm_eps,
            "attention_implementation": "eager",
            "eval_mode": True,
        },
        "input": {
            "text": text,
            "active_tokens": int(encoded["attention_mask"].sum().item()),
            "input_ids": input_ids,
            "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "packing": {
            "attention_qkv": "[out_tile64,input_pack16,local_out,lane]",
            "attention_output": "[head_group128,out_tile32,input_pack16,local_out,lane]",
            "ffn_up": "[out_tile16,input,output_lane]",
            "ffn_down": "[input_tile16,out_tile32,input_lane,out_pack16,out_lane]",
        },
        "files": checksums,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Export complete: {output_dir}")
    print(f"Active tokens: {metadata['input']['active_tokens']}/{SEQ_LEN}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
