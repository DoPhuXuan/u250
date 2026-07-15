# BERT-base FP32 12-layer C-simulation validation

This validation uses the same pinned Hugging Face checkpoint for both sides:

- Hugging Face `BertModel` produces the embedding input and golden tensors.
- The exporter repacks that model's tensors into the exact 512-bit layouts
  consumed by the HLS kernels.
- The C++ testbench executes QKV, Attention Core, Attention Out/Norm, FFN UP,
  and fused FFN DOWN/Norm sequentially for all 12 encoder layers.
- The comparison script measures the Attention midpoint and encoder output of
  every layer against the original model.

Embedding is not synthesized. Its real Hugging Face output is the common
input tensor for layer 0.

## 1. Install Python dependencies

Run from the `fp32_missing_kernel_lab` directory:

```bash
python3 -m venv .venv-validation
source .venv-validation/bin/activate
python3 -m pip install -r validation/requirements.txt
```

The dependency range supports the Python 3.8 environment commonly shipped
with Vitis 2022.1; it selects Transformers 4.46.3 on that Python version.

## 2. Download, pin, and export the real checkpoint

```bash
python3 validation/export_hf_bert_fp32.py \
  --model-id google-bert/bert-base-uncased \
  --output-dir validation/data/bert_base_uncased_seq128
```

The exporter resolves `main` to an immutable Hub commit SHA and records it in
`metadata.json`. To reproduce an earlier run exactly, pass that SHA through
`--revision`.

To replace an existing export, add `--force`. A custom input can be supplied
with `--text` or `--text-file`.

## 3. Run all 12 layers in Vitis HLS C-simulation

```bash
vitis_hls -f scripts/run_bert_12layer_csim.tcl
```

This is a full software execution of the optimized HLS C++ and can take a long
time because BERT-base performs billions of FP32 operations. It does not run
RTL simulation and does not estimate FPGA latency.

Outputs:

```text
validation/outputs/bert_base_uncased_seq128/
  attention_layer_00.f32 ... attention_layer_11.f32
  encoder_layer_00.f32   ... encoder_layer_11.f32
  csim_layer_outputs.csv

reports/csim/bert_base_uncased_seq128/
  accuracy_by_layer.csv
  accuracy_report.json
  accuracy_report.md
```

The default accuracy gates are RMSE <= 0.15, maximum absolute error <= 1.0,
and cosine similarity >= 0.99 for valid tokens and the full 128-token tensor.
They are numerical acceptance gates, not bit-exact gates, because the HLS FFN
uses an eight-segment quadratic GELU approximation.

Override paths and gates with environment variables:

```bash
BERT_CSIM_DATA=/data/bert_validation \
BERT_CSIM_OUTPUT=/scratch/bert_csim_outputs \
BERT_CSIM_REPORT=/scratch/bert_csim_reports \
CSIM_MAX_RMSE=0.15 \
CSIM_MAX_ABS=1.0 \
CSIM_MIN_COSINE=0.99 \
CSIM_GATE_SCOPE=both \
vitis_hls -f scripts/run_bert_12layer_csim.tcl
```

Set `RUN_CSIM_COMPARE=0` to run C-sim without invoking the Python comparison.

## Persistent single-launch graph

After the sequential numerical baseline passes, validate the actual feedback
architecture with bounded, thread-safe software FIFOs:

```bash
vitis_hls -f scripts/run_bert_12layer_persistent_csim.tcl
```

This starts each of the five persistent top functions exactly once in the
testbench and compares only the final encoder output with Hugging Face. The
expected report is:

```text
reports/csim/persistent_12layer/persistent_accuracy_report.md
```

The test-only `BERT_CSIM_THREAD_SAFE_STREAM` macro does not participate in
synthesis. Hardware builds use the ordinary `hls::stream` implementation.
