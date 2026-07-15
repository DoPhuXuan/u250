# BERT FP32 12-layer C-sim accuracy

**Overall: PASS**

- Model: `google-bert/bert-base-uncased`
- Checkpoint commit: `86b5e0934494bd15c9632b12f734a8a67f723594`
- Active tokens: 90/128
- Gates: RMSE <= 0.03, max abs <= 1.1, cosine >= 0.999, scope=valid

| Layer | Stage | Scope | RMSE | Max abs | P99 abs | Cosine | Result |
|---:|---|---|---:|---:|---:|---:|---|
| 0 | attention | valid | 1.9569e-07 | 7.62939e-06 | 5.96046e-07 | 1.00000000 | PASS* |
| 0 | attention | all | 1.85736e-07 | 7.62939e-06 | 5.36442e-07 | 1.00000000 | PASS |
| 0 | encoder | valid | 0.00620857 | 0.0318304 | 0.0169341 | 0.99995814 | PASS* |
| 0 | encoder | all | 0.00541748 | 0.0318304 | 0.0160393 | 0.99996284 | PASS |
| 1 | attention | valid | 0.00819508 | 0.0958424 | 0.0216106 | 0.99997850 | PASS* |
| 1 | attention | all | 0.00761174 | 0.0958424 | 0.020886 | 0.99998085 | PASS |
| 1 | encoder | valid | 0.0105067 | 0.0560477 | 0.0295674 | 0.99990293 | PASS* |
| 1 | encoder | all | 0.00973736 | 0.0560477 | 0.0280339 | 0.99990639 | PASS |
| 2 | attention | valid | 0.0115864 | 0.138256 | 0.0306905 | 0.99995328 | PASS* |
| 2 | attention | all | 0.0117434 | 0.173929 | 0.0313536 | 0.99995603 | PASS |
| 2 | encoder | valid | 0.012496 | 0.0671569 | 0.0348491 | 0.99985342 | PASS* |
| 2 | encoder | all | 0.0120951 | 0.0686463 | 0.0338832 | 0.99984816 | PASS |
| 3 | attention | valid | 0.0142293 | 0.10705 | 0.0381946 | 0.99991982 | PASS* |
| 3 | attention | all | 0.0149426 | 0.283798 | 0.0407237 | 0.99991638 | PASS |
| 3 | encoder | valid | 0.0141778 | 0.077953 | 0.038996 | 0.99981825 | PASS* |
| 3 | encoder | all | 0.0144312 | 0.140379 | 0.0407642 | 0.99980208 | PASS |
| 4 | attention | valid | 0.015554 | 0.140816 | 0.0424281 | 0.99990241 | PASS* |
| 4 | attention | all | 0.0166213 | 0.406424 | 0.0466587 | 0.99989595 | PASS |
| 4 | encoder | valid | 0.0162724 | 0.121745 | 0.0452715 | 0.99977544 | PASS* |
| 4 | encoder | all | 0.0169511 | 0.175653 | 0.0495246 | 0.99974369 | PASS |
| 5 | attention | valid | 0.0177951 | 0.17785 | 0.0485729 | 0.99985718 | PASS* |
| 5 | attention | all | 0.0197972 | 0.437519 | 0.0578628 | 0.99982978 | PASS |
| 5 | encoder | valid | 0.0182876 | 0.123835 | 0.049976 | 0.99972601 | PASS* |
| 5 | encoder | all | 0.0202483 | 0.2279 | 0.0598835 | 0.99965408 | PASS |
| 6 | attention | valid | 0.0191864 | 0.187958 | 0.0517836 | 0.99982995 | PASS* |
| 6 | attention | all | 0.0225445 | 0.597769 | 0.0684808 | 0.99976847 | PASS |
| 6 | encoder | valid | 0.019343 | 0.126572 | 0.0522169 | 0.99968766 | PASS* |
| 6 | encoder | all | 0.0231234 | 0.57321 | 0.0716098 | 0.99954817 | PASS |
| 7 | attention | valid | 0.0200452 | 0.257705 | 0.0540881 | 0.99982307 | PASS* |
| 7 | attention | all | 0.02565 | 0.742373 | 0.0854866 | 0.99969624 | PASS |
| 7 | encoder | valid | 0.0189216 | 0.153424 | 0.0504705 | 0.99967283 | PASS* |
| 7 | encoder | all | 0.0265933 | 1.49844 | 0.0922674 | 0.99940799 | FAIL |
| 8 | attention | valid | 0.0216617 | 0.366391 | 0.0574291 | 0.99981499 | PASS* |
| 8 | attention | all | 0.0285749 | 1.41962 | 0.0961422 | 0.99964485 | FAIL |
| 8 | encoder | valid | 0.0197721 | 0.27629 | 0.053716 | 0.99961850 | PASS* |
| 8 | encoder | all | 0.030152 | 2.17439 | 0.111427 | 0.99925847 | FAIL |
| 9 | attention | valid | 0.0226316 | 0.424755 | 0.0603463 | 0.99972258 | PASS* |
| 9 | attention | all | 0.029186 | 1.31644 | 0.1024 | 0.99950112 | FAIL |
| 9 | encoder | valid | 0.0219173 | 0.294059 | 0.0592972 | 0.99957227 | PASS* |
| 9 | encoder | all | 0.0320577 | 2.5754 | 0.112293 | 0.99923062 | FAIL |
| 10 | attention | valid | 0.0254353 | 1.05772 | 0.0680111 | 0.99969825 | PASS* |
| 10 | attention | all | 0.0317053 | 1.26445 | 0.107902 | 0.99947506 | FAIL |
| 10 | encoder | valid | 0.0233681 | 0.307019 | 0.0632766 | 0.99953019 | PASS* |
| 10 | encoder | all | 0.0343146 | 2.6999 | 0.123728 | 0.99919180 | FAIL |
| 11 | attention | valid | 0.0258471 | 0.593171 | 0.069021 | 0.99973536 | PASS* |
| 11 | attention | all | 0.0336906 | 3.16542 | 0.106994 | 0.99961650 | FAIL |
| 11 | encoder | valid | 0.0185169 | 0.159416 | 0.0509648 | 0.99938928 | PASS* |
| 11 | encoder | all | 0.0203699 | 0.393543 | 0.0638856 | 0.99914089 | PASS |

`*` indicates a row included in the overall gate.

The Hugging Face reference uses exact erf-based GELU. The HLS kernel uses its documented eight-segment quadratic FP32 GELU approximation, so bit-exact equality is not expected.
