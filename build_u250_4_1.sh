#!/usr/bin/env bash
set -euo pipefail

PLATFORM_DIR="${PLATFORM_DIR:-$HOME/MinhPhat/xilinx_u250_gen3x16_xdma_4_1_202210_1}"
PLATFORM="${PLATFORM:-$(find "$PLATFORM_DIR" -name '*.xpfm' -print -quit)}"
BUILD_DIR="build_4_1"
XCLBIN="bert_pipeline_u250_4_1.xclbin"

if [[ -z "$PLATFORM" || ! -f "$PLATFORM" ]]; then
  echo "Cannot find the U250 4.1 .xpfm under: $PLATFORM_DIR" >&2
  exit 1
fi
command -v v++ >/dev/null || { echo "v++ is not in PATH" >&2; exit 1; }

mkdir -p "$BUILD_DIR"
echo "Platform: $PLATFORM"

v++ -t hw -c --platform "$PLATFORM" -I. -k bert_embedding_prep_kernel \
  bert_emb_qkv_kernel.cpp embedding_stream.cpp \
  -o "$BUILD_DIR/bert_embedding_prep_kernel.xo"

v++ -t hw -c --platform "$PLATFORM" -I. -k bert_qkv_kernel \
  bert_qkv_kernel.cpp attention_stream.cpp \
  -o "$BUILD_DIR/bert_qkv_kernel.xo"

v++ -t hw -c --platform "$PLATFORM" -I. -k bert_attn_core_kernel \
  bert_attn_core_kernel.cpp attention_stream.cpp \
  -o "$BUILD_DIR/bert_attn_core_kernel.xo"

v++ -t hw -c --platform "$PLATFORM" -I. -k bert_attn_out_norm_kernel \
  bert_attn_out_norm_kernel.cpp attention_stream.cpp \
  -o "$BUILD_DIR/bert_attn_out_norm_kernel.xo"

v++ -t hw -c --platform "$PLATFORM" -I. -k bert_ffn_up_gelu_kernel \
  bert_ffn_up_gelu_kernel.cpp ffn_stream.cpp \
  -o "$BUILD_DIR/bert_ffn_up_gelu_kernel.xo"

v++ -t hw -c --platform "$PLATFORM" -I. -k bert_ffn_down_norm_kernel \
  bert_ffn_down_norm_kernel.cpp ffn_stream.cpp \
  -o "$BUILD_DIR/bert_ffn_down_norm_kernel.xo"

v++ -t hw -l --platform "$PLATFORM" --config system_optimized.cfg \
  "$BUILD_DIR/bert_embedding_prep_kernel.xo" \
  "$BUILD_DIR/bert_qkv_kernel.xo" \
  "$BUILD_DIR/bert_attn_core_kernel.xo" \
  "$BUILD_DIR/bert_attn_out_norm_kernel.xo" \
  "$BUILD_DIR/bert_ffn_up_gelu_kernel.xo" \
  "$BUILD_DIR/bert_ffn_down_norm_kernel.xo" \
  -o "$XCLBIN"

echo "Built: $XCLBIN"
