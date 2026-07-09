# FFN V18 Staged Comparison

Status: completed. V18 is a rolled-back reference experiment, not the default.

Required reports:

```text
csynth_up_v18.rpt
csynth_down_v18.rpt
```

Build command:

```bash
cd /home/minhphat/ffn
export HLS_PART=xcu250-figd2104-2L-e
export HLS_CLOCK_NS=5
vitis_hls -f run_vitis_ffn_kernel.tcl bert_ffn_up_gelu_v18_staged_tilemajor_kernel bert_ffn_down_v18_staged_tilemajor_kernel
python3 collect_ffn_kernel_reports.py
```

Before comparing, verify report headers:

```text
Synthesis Summary Report of 'bert_ffn_up_gelu_v18_staged_tilemajor_kernel'
Synthesis Summary Report of 'bert_ffn_down_v18_staged_tilemajor_kernel'
```

| version | kernel | latency | II hot MAC | timing/slack | LUT | FF | BRAM | URAM | DSP | warnings | nhan xet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V15 | up | 1,501,064 | II=1 | Timing -0.00 | 95,238 | 115,221 | 302 | 32 | 1280 | no stride | baseline |
| V15 | down | 1,586,139 | II=1 | Timing -0.00 | 98,451 | 145,884 | 76 | 32 | 1280 | no stride | baseline |
| V18 | up | 1,660,040 | II=1 | Timing -0.00 | 95,341 | 115,249 | 310 | 32 | 1280 | no stride/no clobber | +10.59% latency, no top timing improvement |
| V18 | down | 1,787,910 | II=1 | Timing -0.00 | 98,663 | 145,855 | 100 | 32 | 1280 | no stride/no clobber | +12.72% latency, BRAM +24, no top timing improvement |

Acceptance check result:

- PASS: UP/DOWN hot MAC loops stay `II=1`.
- PASS: staged W1/W2 load loops stay `II=1`.
- PASS: staged output store loops stay `II=1`.
- PASS: `Stride is incompatible` remains gone.
- PASS: `Access is clobbered by call` is not found in the v18 reports.
- FAIL: top timing/slack does not improve versus V15; top timing remains `-0.00`.
- FAIL: latency increase is much larger than the `+2%` budget.
- PASS: DSP stays near `1280`.
- PASS: FF does not repeat the large increase caused by the previous `LATENCY min=2 max=20` pass.

Decision: do not use V18 as the baseline. Restore V15 and move the next pass to
post-route timing, congestion, SLR placement, and interface diagnosis.
