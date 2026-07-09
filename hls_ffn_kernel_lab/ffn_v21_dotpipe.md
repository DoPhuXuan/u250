# FFN V21 Dot-Pipeline Pass

## Change

V21 keeps the V15 tile schedule, `K_PAR=16`, `MAC_GROUP=16`, packed streams,
URAM/BRAM layout, DDR banks, and SLR placement. It changes only the
dot-product hierarchy:

```text
V20: INLINE dot16 helper; child BIND_OP is dissolved during inlining
V21: INLINE off + function PIPELINE II=1; fmul maxdsp latency=5
```

The separate hierarchy is intentional. Vitis HLS 2021.2 documents that
pragmas in a child context are ignored when the child function is inlined.

## Run

```bash
cd /home/minhphat/ffn
export HLS_PART=xcu250-figd2104-2L-e
export HLS_CLOCK_NS=3.333
vitis_hls -f run_vitis_ffn_kernel.tcl
python3 collect_ffn_kernel_reports.py
```

## Acceptance Gate

Do not run `v++ --link` unless all conditions pass:

1. A separate `v13_dot16_tree_margin` module appears in the hierarchy.
2. Its function pipeline has final `II=1`.
3. UP and DOWN hot MAC loops keep final `II=1`.
4. DSP remains near 1,280 per kernel; a large reduction indicates unintended
   function sharing and loss of parallelism.
5. The generated multiplier implementation or its reported latency differs
   from V20.
6. Kernel latency increase remains below 3% relative to the 300 MHz V20
   reports.
7. HLS timing has positive margin. Final acceptance still requires routed
   `WNS >= 0` at 300 MHz.

## 300 MHz Result

The accepted V21 csynth iteration scalarizes the 16 products and binds each
scalar multiplier to `fulldsp latency=7`. This creates a real RTL change:
`fmul_32ns_32ns_32_8_full_dsp`, two DSPs per multiplier, with reported
operator latency 4.

| Kernel | Latency | II | DSP | BRAM | URAM | LUT | FF |
|---|---:|---:|---:|---:|---:|---:|---:|
| V21 UP | 1,534,088 | 1 | 1,120 | 302 | 32 | 90,257 | 178,217 |
| V21 DOWN | 1,646,044 | 1 | 1,024 | 76 | 32 | 92,339 | 202,321 |

Delta from V20 at 300 MHz:

```text
UP:   latency +0.757%, DSP -12.50%, LUT +0.71%, FF +15.40%
DOWN: latency +1.132%, DSP -20.00%, LUT -0.55%, FF +10.18%
```

Passed:

```text
- 16 separate v13_dot16_tree_margin instances are visible.
- Each dot function has interval/II = 1.
- UP and DOWN hot MAC loops keep II = 1.
- Kernel latency growth remains below 3%.
- Dot multipliers now use fulldsp rather than the routed V15 maxdsp IP.
- DSP usage drops materially without reducing K_PAR or MAC_GROUP.
```

Remaining gate:

```text
- Top-level HLS slack is still rounded to 0.00 ns.
- UP still contains maxdsp operators outside the dot tree in GELU.
- Final acceptance requires full-link/post-route WNS >= 0 at 300 MHz.
```

Decision: V21 passes the csynth gate and is the current full-link candidate.
Do not create another source version until its routed timing and SLR
utilization are available.
