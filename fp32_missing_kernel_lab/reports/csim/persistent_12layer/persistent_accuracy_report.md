# Persistent BERT FP32 12-layer accuracy

**Overall: PASS**

- Model: `google-bert/bert-base-uncased`
- Commit: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Active tokens: 90/128
- Execution: five CUs started once; no host-managed layer loop

| Scope | RMSE | Max abs | P99 abs | Cosine | Gate |
|---|---:|---:|---:|---:|---|
| valid | 0.01851688 | 0.1594163 | 0.05096477 | 0.999389279 | PASS |
| all (diagnostic) | 0.02036995 | 0.3935428 | 0.06388558 | 0.999140894 | not gated |
