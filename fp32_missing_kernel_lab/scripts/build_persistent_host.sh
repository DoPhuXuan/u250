#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="$ROOT_DIR/build/full_12layer_hw/run_persistent_12layer"
mkdir -p "$(dirname "$OUTPUT")"

if pkg-config --exists xrt 2>/dev/null; then
  read -r -a XRT_CFLAGS <<<"$(pkg-config --cflags xrt)"
  read -r -a XRT_LIBS <<<"$(pkg-config --libs xrt)"
else
  : "${XILINX_XRT:?Set XILINX_XRT or install the xrt pkg-config file}"
  XRT_CFLAGS=("-I$XILINX_XRT/include")
  XRT_LIBS=("-L$XILINX_XRT/lib" -lxrt_coreutil)
fi

g++ -std=c++17 -O2 -pthread \
  "${XRT_CFLAGS[@]}" \
  "$ROOT_DIR/host/run_persistent_12layer.cpp" \
  "${XRT_LIBS[@]}" \
  -o "$OUTPUT"
echo "Host runner: $OUTPUT"

