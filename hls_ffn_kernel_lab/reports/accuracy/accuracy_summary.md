# FP32 V21 numerical accuracy check

Evaluation date: 2026-07-14

## Contract

The evaluated HLS source keeps weights, activations, bias, and accumulation in
FP32. Its GELU is the V21 three-segment approximation:

- `0` for `x <= -3`;
- `x` for `x >= 3`;
- `x * (0.5 + x / 6)` otherwise.

This differs from the erf-based GELU used by the original pretrained
`bert-base-uncased` model.

## Independent FFN comparison

The original framework input to each layer was held fixed and only that FFN's
GELU was replaced by the V21 PWL.

| Layer | Global cosine |
|---:|---:|
| 0 | 0.628637174 |
| 1 | 0.628082132 |
| 2 | 0.966003518 |
| 3 | 0.812927978 |
| 4 | 0.870875761 |
| 5 | 0.820171243 |
| 6 | 0.695433072 |
| 7 | 0.574848587 |
| 8 | 0.661599609 |
| 9 | 0.998091641 |
| 10 | 0.999093207 |
| 11 | 0.916656211 |

## Twelve-layer propagation

All 12 FFNs were replaced by the FP32 V21 PWL contract. The comparison used
16 deterministic text sentences with 228 valid tokens. Attention, residual
connections, LayerNorm, and all other operations stayed in framework FP32.

- Final hidden-state global cosine: 0.245780177.
- Final hidden-state cosine loss: 75.421982%.
- Mean token cosine: 0.248874502.
- Minimum token cosine: -0.094412163.
- CLS global cosine: 0.049364984.
- Result for `1 - cosine <= 1%`: **FAIL**.

The first encoder layer already falls to cosine 0.933696109 and accumulated
error grows rapidly. FP32 precision therefore does not make V21 numerically
equivalent to the original model; the dominant discrepancy is the GELU
formula. See `bert_full_model_fp32_v21_pwl.json` for complete per-layer data.

This is a hidden-state comparison, not a downstream task accuracy/F1 test.

## Updated resource-neutral PWL

The source was updated to eight constrained quadratic intervals over `[-4,4)`.
Every interval uses `x * (a*x + b)`, so each lane retains V21's arithmetic
count of two FP32 multiplications and one FP32 addition. Values below `-4`
map to zero and values at or above `4` pass through unchanged.

Full-model results on the same 16 sentences and 228 valid tokens:

- Final hidden-state global cosine: 0.999246224.
- Final hidden-state cosine loss: 0.075378%.
- Mean token cosine: 0.999258044.
- Minimum token cosine: 0.971253816.
- CLS global cosine: 0.999644604.
- Result for `1 - cosine <= 1%`: **PASS**.

This is not bit-exact equality with framework GELU. Exact equality for every
FP32 input would require an erf implementation and matching the framework's
matrix-accumulation order, which cannot preserve identical hardware cost.
The unchanged DSP/II/latency hypothesis must still be checked in the new
csynth report; coefficient-selection comparators can affect LUT and timing.
