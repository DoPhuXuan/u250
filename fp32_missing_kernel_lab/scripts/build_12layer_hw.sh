#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/build_12layer_hw.sh /path/to/u250_platform.xpfm" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM="$(readlink -f "$1")"
if [[ ! -f "$PLATFORM" ]]; then
  echo "Platform does not exist: $PLATFORM" >&2
  exit 2
fi
command -v v++ >/dev/null || { echo "v++ is not on PATH" >&2; exit 2; }

BUILD_DIR="$ROOT_DIR/build/full_12layer_hw"
XO_DIR="$BUILD_DIR/xo"
REPORT_DIR="$ROOT_DIR/reports/full_12layer_hw"
LOG_DIR="$REPORT_DIR/logs"
mkdir -p "$XO_DIR" "$REPORT_DIR" "$LOG_DIR"

compile_kernel() {
  local top="$1"
  local source="$2"
  echo "============================================================"
  echo "Compiling $top"
  echo "============================================================"
  v++ -c -t hw \
    --platform "$PLATFORM" \
    -k "$top" \
    --hls.clock "300000000:$top" \
    --include "$ROOT_DIR/include" \
    --optimize 3 \
    --save-temps \
    --temp_dir "$BUILD_DIR/tmp_compile_$top" \
    --report_dir "$REPORT_DIR/compile_$top" \
    --log_dir "$LOG_DIR/compile_$top" \
    "$ROOT_DIR/kernels/$source" \
    -o "$XO_DIR/$top.xo"
}

compile_kernel bert_qkv_12layer_kernel bert_qkv_kernel.cpp
compile_kernel bert_attn_core_12layer_kernel bert_attn_core_kernel.cpp
compile_kernel bert_attn_out_norm_12layer_kernel bert_attn_out_norm_residual_kernel.cpp
compile_kernel bert_ffn_up_gelu_12layer_kernel bert_ffn_fp32_kernels.cpp
compile_kernel bert_ffn_down_norm_feedback_12layer_kernel bert_ffn_fp32_kernels.cpp

echo "============================================================"
echo "Linking persistent 12-layer encoder at 300 MHz"
echo "============================================================"
v++ -l -t hw \
  --platform "$PLATFORM" \
  --kernel_frequency 300 \
  --config "$ROOT_DIR/config/system_12layer.cfg" \
  --save-temps \
  --temp_dir "$BUILD_DIR/tmp_link" \
  --report_dir "$REPORT_DIR/link" \
  --log_dir "$LOG_DIR/link" \
  "$XO_DIR/bert_qkv_12layer_kernel.xo" \
  "$XO_DIR/bert_attn_core_12layer_kernel.xo" \
  "$XO_DIR/bert_attn_out_norm_12layer_kernel.xo" \
  "$XO_DIR/bert_ffn_up_gelu_12layer_kernel.xo" \
  "$XO_DIR/bert_ffn_down_norm_feedback_12layer_kernel.xo" \
  -o "$BUILD_DIR/bert_base_12layer_u250.xclbin"

echo "XCLBIN: $BUILD_DIR/bert_base_12layer_u250.xclbin"
echo "Reports: $REPORT_DIR"

