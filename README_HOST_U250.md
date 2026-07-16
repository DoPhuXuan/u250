# BERT base uncased host for Alveo U250

This host runs the supplied `bert_pipeline_hw_200.xclbin` with the FP32
weights from `google-bert/bert-base-uncased`. The bitstream targets
`xilinx_u250_gen3x16_xdma_3_1_202020_1`, has a fixed sequence length of 128,
and produces the final encoder tensor with shape `[128, 768]`.

## Important compatibility note

The supplied xclbin and the newest C++ kernel files in this directory are from
different revisions. The xclbin metadata contains `attn_mid_ddr` and
`attn_mid_done`; the newest sources replaced that DDR residual hand-off with an
AXI stream. `host_bert_u250.cpp` intentionally follows the interface embedded
in **the supplied xclbin**, including its original argument indices and DDR
placement.

## 1. Prepare the U250 machine

Use Linux with the XRT/runtime and deployment platform matching the xclbin.
Confirm that the card is visible before running:

```bash
xbutil examine
```

The card shell should be compatible with:

```text
xilinx_u250_gen3x16_xdma_3_1_202020_1
```

## 2. Export Google BERT weights and tokenize an input

The exporter writes raw little-endian FP32 tensors and three int32 input files.
Attention matrices remain in PyTorch `[out, in]` order. FFN matrices are
transposed to the `[in, out]` order used by the HLS address calculations.

```bash
python3 -m pip install numpy torch transformers safetensors
python3 export_bert_base_uncased.py \
  --output-dir bert_base_uncased_u250 \
  --text "The quick brown fox jumps over the lazy dog."
```

For sentence-pair input, add `--text-pair "second sentence"`. The tokenizer
always pads/truncates to 128 tokens. `manifest.json` records every shape, byte
count, and SHA-256 digest.

If the model is already cached and the machine is offline, add
`--local-files-only`.

## 3. Build and run the XRT host

```bash
source /opt/xilinx/xrt/setup.sh
make
./host_bert_u250 \
  --device 0 \
  --xclbin bert_pipeline_hw_200.xclbin \
  --data-dir bert_base_uncased_u250 \
  --output last_hidden_state.bin
```

The host allocates model tensors in the banks fixed by the xclbin:

- DDR0: inputs, masks, hidden ping/pong, Q weights and synchronization token.
- DDR1: embeddings, K weights and the attention residual buffer/token.
- DDR2: V weights and FFN-up weights.
- DDR3: attention-output, FFN-down and layer-normalization parameters.

For each of the 12 layers, all six kernels are started sink-to-source so that
the internally connected AXI streams have active consumers. The host waits for
FFN-down before reusing the ping-pong hidden buffer in the next layer.

## Output

`last_hidden_state.bin` contains 98,304 little-endian FP32 values in row-major
`[sequence, hidden]` order. It is the encoder output only; the supplied xclbin
does not implement BERT's pooler or a task-specific classification head.

Load it in Python with:

```python
import numpy as np

hidden = np.fromfile("last_hidden_state.bin", dtype="<f4").reshape(128, 768)
print(hidden[0, :16])
```
