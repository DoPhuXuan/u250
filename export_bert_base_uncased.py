#!/usr/bin/env python3
"""Export google-bert/bert-base-uncased tensors in the layout used by the U250 xclbin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL_ID = "google-bert/bert-base-uncased"
SEQ_LEN = 128
HIDDEN = 768
FFN_DIM = 3072
NUM_LAYERS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BERT base uncased and export FP32 files for host_bert_u250"
    )
    parser.add_argument("--output-dir", default="bert_base_uncased_u250")
    parser.add_argument("--text", default="Hello from the Alveo U250.")
    parser.add_argument("--text-pair", default=None)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use an already-populated Hugging Face cache without network access",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoTokenizer, BertModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python3 -m pip install numpy torch transformers safetensors"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = BertModel.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model.eval()

    encoded = tokenizer(
        args.text,
        args.text_pair,
        max_length=SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    state = model.state_dict()
    prefix = "embeddings"
    encoder_prefix = "encoder.layer"
    tensors: dict[str, "torch.Tensor"] = {}

    # Embedding tables are already row-major [vocabulary/position/type, hidden].
    tensors["token_emb.bin"] = state[f"{prefix}.word_embeddings.weight"]
    tensors["pos_emb.bin"] = state[f"{prefix}.position_embeddings.weight"][:SEQ_LEN]
    tensors["seg_emb.bin"] = state[f"{prefix}.token_type_embeddings.weight"]
    tensors["emb_gamma.bin"] = state[f"{prefix}.LayerNorm.weight"]
    tensors["emb_beta.bin"] = state[f"{prefix}.LayerNorm.bias"]

    q_w, q_b, k_w, k_b, v_w, v_b = [], [], [], [], [], []
    o_w, o_b, attn_gamma, attn_beta = [], [], [], []
    up_w, up_b, down_w, down_b, ffn_gamma, ffn_beta = [], [], [], [], [], []

    for layer in range(NUM_LAYERS):
        base = f"{encoder_prefix}.{layer}"
        self_attn = f"{base}.attention.self"
        attn_out = f"{base}.attention.output"
        ffn_intermediate = f"{base}.intermediate.dense"
        ffn_output = f"{base}.output"

        # HLS attention loaders index W[out][in], identical to torch Linear.
        q_w.append(state[f"{self_attn}.query.weight"])
        q_b.append(state[f"{self_attn}.query.bias"])
        k_w.append(state[f"{self_attn}.key.weight"])
        k_b.append(state[f"{self_attn}.key.bias"])
        v_w.append(state[f"{self_attn}.value.weight"])
        v_b.append(state[f"{self_attn}.value.bias"])
        o_w.append(state[f"{attn_out}.dense.weight"])
        o_b.append(state[f"{attn_out}.dense.bias"])
        attn_gamma.append(state[f"{attn_out}.LayerNorm.weight"])
        attn_beta.append(state[f"{attn_out}.LayerNorm.bias"])

        # HLS FFN loaders index W[input][output], so transpose torch [out, in].
        up_w.append(state[f"{ffn_intermediate}.weight"].transpose(0, 1).contiguous())
        up_b.append(state[f"{ffn_intermediate}.bias"])
        down_w.append(state[f"{ffn_output}.dense.weight"].transpose(0, 1).contiguous())
        down_b.append(state[f"{ffn_output}.dense.bias"])
        ffn_gamma.append(state[f"{ffn_output}.LayerNorm.weight"])
        ffn_beta.append(state[f"{ffn_output}.LayerNorm.bias"])

    layer_groups = {
        "attn_q_w.bin": q_w,
        "attn_q_b.bin": q_b,
        "attn_k_w.bin": k_w,
        "attn_k_b.bin": k_b,
        "attn_v_w.bin": v_w,
        "attn_v_b.bin": v_b,
        "attn_o_w.bin": o_w,
        "attn_o_b.bin": o_b,
        "attn_norm_gamma.bin": attn_gamma,
        "attn_norm_beta.bin": attn_beta,
        "ffn_up_w.bin": up_w,
        "ffn_up_b.bin": up_b,
        "ffn_down_w.bin": down_w,
        "ffn_down_b.bin": down_b,
        "ffn_norm_gamma.bin": ffn_gamma,
        "ffn_norm_beta.bin": ffn_beta,
    }
    for filename, values in layer_groups.items():
        tensors[filename] = torch.stack(values, dim=0)

    # The kernels consume signed int32 IDs/masks. Nonzero mask entries are active.
    input_arrays = {
        "input_ids.bin": encoded["input_ids"][0].to(torch.int32).cpu().numpy(),
        "token_type_ids.bin": encoded.get(
            "token_type_ids", torch.zeros((1, SEQ_LEN), dtype=torch.int64)
        )[0].to(torch.int32).cpu().numpy(),
        "attention_mask.bin": encoded["attention_mask"][0].to(torch.int32).cpu().numpy(),
    }

    files: dict[str, dict[str, object]] = {}
    for filename, tensor in tensors.items():
        array = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        if array.dtype.byteorder == ">":
            array = array.byteswap().newbyteorder("<")
        path = output_dir / filename
        with path.open("wb") as handle:
            array.astype("<f4", copy=False).tofile(handle)
            handle.flush()
        files[filename] = {
            "dtype": "float32-le",
            "shape": list(array.shape),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    for filename, array in input_arrays.items():
        path = output_dir / filename
        with path.open("wb") as handle:
            np.asarray(array, dtype="<i4").tofile(handle)
            handle.flush()
        files[filename] = {
            "dtype": "int32-le",
            "shape": [SEQ_LEN],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "model": args.model,
        "sequence_length": SEQ_LEN,
        "hidden_size": HIDDEN,
        "intermediate_size": FFN_DIM,
        "num_layers": NUM_LAYERS,
        "text": args.text,
        "text_pair": args.text_pair,
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(files)} binary files to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
