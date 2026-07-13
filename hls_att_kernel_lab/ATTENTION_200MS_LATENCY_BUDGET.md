# Attention + FFN latency budget for 200 ms

Target assumptions:

```text
clock              300 MHz
encoder layers     12
model target       < 200 ms
total cycle budget < 60,000,000
per-layer budget   < 5,000,000 cycles
```

## Pre-optimization baseline

| Kernel | Cycles |
|---|---:|
| QKV | 6,250,809 |
| Attention Core | 15,781,090 |
| Attention Out/Norm | 11,286,034 |
| FFN UP V21 | 1,536,392 |
| FFN DOWN V21 | 1,655,260 |

The old full-context boundary serialized Core and Out/Norm. Even the optimistic
largest-kernel estimate exceeded the five-million-cycle layer budget.

## Active v3 architecture

The six head groups form a pipeline:

```text
QKV group g -> Core group g -> Output projection accumulate group g
```

Core no longer stores and reorders a full context tensor. Output/Norm consumes
one group, updates the full projected accumulator, then accepts the next group.
The final normalized stream remains row-major for FFN.

Parallelism changes relative to the baseline report:

| Block | Baseline | Active v3 |
|---|---:|---:|
| QKV K lanes | 4 | 8 |
| Core dimension lanes | 4 | 8 |
| Core queries | 1 | 2 |
| Core heads/group | sequential | parallel |
| Output K lanes | 4 | 8 |
| Output lanes | 4 | 8 |

## Acceptance calculation after new csynth

Let `Q`, `C`, and `O` be the reported per-group latency of QKV, Core and Output
projection. For six groups, estimate the Attention pipeline as:

```text
attention_cycles = Q + C + O + 5 * max(Q, C, O) + norm_tail
layer_cycles     = attention_cycles + max(ffn_up, ffn_down)
model_ms         = 12 * layer_cycles / 300000
```

Acceptance requires `layer_cycles < 5,000,000`. A balanced design therefore
needs each Attention group stage near or below 400,000 cycles. The active v4 report is
required before claiming the 200 ms target; scaling the old report is only a
design estimate.

## V3 measured result and v4 action

Measured v3 group latency:

```text
QKV   927,110 cycles
Core  700,173 cycles
Out 1,852,558 cycles
```

The group-pipeline model is about 579 ms for 12 layers. V4 therefore targets
the measured hot loops rather than increasing global constants:

- Out: pipeline a complete 32-output row, direct tile partition, DSP-bound
  products and 32-bank projected accumulator.
- Core: unroll two context dimension chunks with 16-bank V/context storage.
- QKV: remove row helper unpacking whose partition directives were ignored.

V4 acceptance guards:

```text
Out group  <= 400,000 cycles, Out DSP <= 1,100, Out LUT <= 105,000
Core group <= 400,000 cycles, Core DSP <= 1,000, Core LUT <= 180,000
QKV group  <= 700,000 cycles in this incremental pass
```

## V4 measured result and balanced v5

Measured v4:

```text
QKV group   933,254 cycles
Core group  699,149 cycles
Out group   118,222 cycles
model       about 325.2 ms
```

V4 Out was intentionally rejected despite its latency: 252,494 LUT and
`-6.70 ns` slack would place SLR2 near 99% LUT with FFN UP. Balanced v5 makes
the following changes:

- Out: eight outputs per row/block, projected accumulator back to eight banks.
- Core: key-major 16-lane context updates into eight recurrence banks, followed
  by one bank reduction.
- QKV: cache each 64x768 Q/K/V weight tile in URAM and accumulate one row in
  local eight-bank partial sums.

V5 acceptance window:

```text
QKV group   <= 430,000 cycles, DSP <= 1,900, LUT <= 220,000, URAM <= 160
Core group  <= 250,000 cycles, DSP <= 1,000, LUT <= 150,000, URAM <= 160
Out group   <= 350,000 cycles, DSP <=   600, LUT <= 105,000, URAM <= 100
HLS slack   >= -0.20 ns for every kernel before full link
model       < 200 ms using the six-group pipeline equation
```

## V5 measured result and balanced v6

Measured v5:

```text
QKV group   1,849,426 cycles, slack -0.24 ns
Core group    439,309 cycles, slack -0.69 ns
Out group     413,134 cycles, slack -0.04 ns
model       about 547.0 ms
```

The QKV report identifies `PROJECT_INPUT_STEP` as the dominant regression:
the requested II=1 became II=4 because HLS conservatively assigned a
distance-one dependence to the rotating partial-sum banks. The two weight-tile
loads contribute only 12,312 cycles per group and are not the bottleneck.

Balanced v6 therefore makes only the report-driven changes below:

- QKV: declare the inter-iteration dependence false for the three fully
  partitioned rotating accumulator banks. The same element is revisited only
  after eight iterations. Increase output lanes from four to eight so each
  64-output tile uses eight rather than sixteen output blocks.
- Core: declare the equivalent false dependence for context banks and request
  II=3, not II=1. Reducing `CONTEXT_KEY_CHUNK` from II=4 to II=3 is sufficient
  to save about 65,000 cycles per group without another query-engine copy.
- Out/Norm: unchanged. Its v5 implementation is the accepted route-friendly
  point for the FFN-UP SLR.

V6 acceptance window:

```text
QKV PROJECT_INPUT_STEP final II = 1
QKV group   <= 400,000 cycles, DSP <= 1,900, LUT <= 260,000, URAM <= 240
Core CONTEXT_KEY_CHUNK final II <= 3
Core group  <= 390,000 cycles, DSP <= 1,000, LUT <= 180,000, URAM <= 160
Out group   <= 413,134 cycles and resource use must not regress from v5
HLS slack   >= -0.20 ns before accepting a kernel for full link
model       < 200 ms using the six-group pipeline equation
```

If QKV reaches II=1 but misses the group target, inspect output-block latency
before increasing any other parallelism. If Core latency passes but slack stays
below -0.20 ns, rewrite its dynamic bank selector into static bank lanes rather
than adding compute resources.

## V6 scheduler diagnosis and v6.1 recovery

The first v6 QKV attempt with eight output lanes did not complete scheduling in
a practical time. Vitis HLS 2022.1 scalarized the three fully partitioned
`[8][8][8]` partial-sum tensors, emitted 1,536 false-dependency diagnostics and
still retained a distance-one wire recurrence. The scheduler remained in
`PROJECT_INPUT_STEP` after trying II=1 through II=4. This run is rejected; it is
not evidence that more host RAM or a longer timeout is required.

V6.1 is the controlled recovery step:

- return QKV output parallelism to four lanes so scheduling complexity is no
  worse than the completed v5 run;
- represent the eight rotating accumulators as named fields;
- select one named field, perform one floating-point addition, then write the
  selected field back;
- retain the false inter-dependence declaration because the same named field is
  revisited only every eight input steps;
- leave Core and Out/Norm unchanged.

V6.1 is a scheduling proof, not the final latency point. Its acceptance gates
are:

```text
QKV csynth completes without a long scheduler stall
PROJECT_INPUT_STEP final II = 1
PROJECT_INPUT_STEP scheduling time < 5 minutes
QKV group expected near 650,000-720,000 cycles
no large LUT/FF regression relative to v5
```

Only after all gates pass should v6.2 restore `QKV_O_PAR=8`. With static banks,
eight output lanes are expected to reduce the v6.1 group latency toward the
400,000-cycle target without recreating the dynamic-bank scheduling graph.

## V6.1 measured result and v6.1r rotating banks

V6.1 completed, so reducing `QKV_O_PAR` successfully removed the scheduler
stall. It did not remove the recurrence:

```text
QKV top                         11,108,865 cycles
QKV group                        1,849,426 cycles
PROJECT_INPUT_STEP final II              4
PROJECT_INPUT_STEP scheduling       117.96 s
slack                              -2.71 ns
LUT                                 166,761
```

The read-switch/add/write-switch helper produced a distance-one PHI recurrence
between `current` and `updated`. Its selection mux also increased the estimated
clock period to 5.137 ns. V6.1 is therefore rejected for both latency and
timing.

V6.1r replaces both switches with a static eight-register rotation. Each input
step reads lane zero, performs one addition, shifts the other seven lanes and
writes the result into lane seven. A value returns to the adder after eight
iterations, exposing the real recurrence distance of eight without dynamic
indexing or a selection mux. Since `PROJECT_INPUT_STEP` has 96 iterations, the
rotation completes twelve full turns and the final reduction sees exactly the
same FP32 bank values and order as the original modulo-eight accumulator.

V6.1r acceptance gates:

```text
PROJECT_INPUT_STEP final II = 1
PROJECT_INPUT_STEP latency near 109 cycles
QKV group latency 650,000-720,000 cycles
QKV slack >= -0.20 ns
no long scheduling stall and no large LUT/FF regression
```

`QKV_O_PAR` remains four for this proof run. Do not restore eight lanes until
all v6.1r gates pass.

## V6.1r measured result and v6.1t timing pass

Rotating banks passed the functional scheduling gate:

```text
QKV top                         4,080,129 cycles
QKV group                         677,970 cycles
PROJECT_INPUT_STEP final II              1
PROJECT_INPUT_STEP latency             107 cycles
slack                              -0.30 ns
DSP                                  1,056
FF                                 252,054
LUT                                162,275
URAM                                   114
```

Latency improved by 2.72x and the dynamic-bank recurrence is resolved. The
remaining failure is the small negative timing margin in
`PROJECT_INPUT_STEP`. V6.1t binds only the rotating `updated` FP32 addition to
the proven FFN timing profile `fulldsp latency=5`. Eight recurrence lanes leave
three cycles of distance margin, so this binding must preserve II=1.

V6.1t acceptance gates:

```text
PROJECT_INPUT_STEP final II = 1
QKV group latency <= 680,000 cycles
QKV slack >= -0.20 ns, preferably >= 0
DSP <= 1,300
no material LUT/FF regression
```

If these gates pass, v6.2 may restore `QKV_O_PAR=8` for the final QKV latency
step.

## V6.1t measured result and v6.2 output lanes

The DSP-add timing experiment is rejected:

```text
                         v6.1r fabric    v6.1t fulldsp
QKV top                    4,080,129        4,055,553
QKV group                    677,970          673,874
PROJECT_INPUT_STEP II              1                1
slack                           -0.30 ns         -0.91 ns
DSP                              1,056            1,248
FF                             252,054          282,347
LUT                            162,275          178,221
```

Vitis implemented each requested latency-five FP32 add as a latency-six,
two-DSP operator. The 0.6% latency gain does not justify the timing and routing
regression. V6.2 removes this binding, returns to the accepted rotating fabric
adder and increases `QKV_O_PAR` from four to eight. This halves output blocks
per 64-output tile from sixteen to eight while preserving `PROJECT_INPUT_STEP`
II=1 and its 96-step accumulation order.

V6.2 exploratory acceptance gates:

```text
csynth completes without the old dynamic-bank scheduler stall
PROJECT_INPUT_STEP final II = 1
QKV group latency <= 380,000 cycles
QKV top latency <= 2,300,000 cycles
DSP <= 2,300
LUT <= 300,000
FF <= 450,000
URAM <= 240
slack >= -0.50 ns for further timing work
```

Passing latency alone is insufficient. A v6.2 result outside any resource or
timing guard is rejected before full linking.

## V6.2 measured result and v6.3 shared reduction

V6.2 reached the required QKV latency while keeping the rotating MAC at II=1:

```text
QKV top                         2,083,281 cycles
QKV group                         345,162 cycles
PROJECT_INPUT_STEP final II              1
slack                              -0.30 ns
DSP                                  2,112
FF                                 476,948
LUT                                283,779
URAM                                   216
```

The arithmetic inventory contains 192 max-DSP multipliers using 576 DSPs and
768 full-DSP adders using 1,536 DSPs. The fully unrolled `REDUCE_OUTPUT` loop is
the main source of replicated add trees. V6.3 keeps the eight-lane MAC and
changes only this outer reduction loop from complete unroll to pipeline II=1.
The eight K lanes and each balanced reduction tree remain unrolled inside one
shared, fully pipelined output reduction engine.

The current 162-cycle output block may grow to about 190 cycles while keeping
QKV group latency below the approximately 405,000-cycle bound required for the
final model target. V6.3 acceptance gates are:

```text
PROJECT_INPUT_STEP final II = 1
REDUCE_OUTPUT final II = 1
output-block latency <= 190 cycles
QKV group latency <= 405,000 cycles
DSP <= 1,400
LUT <= 240,000
FF <= 380,000
URAM <= 220
slack must not regress below -0.30 ns
```

No timing binding is combined with v6.3 so the report measures reduction
sharing independently. A separate fabric-adder timing pass is allowed only
after these resource and latency gates pass.

## V6.3 measured result and v6.4 fabric timing

V6.3 passed every resource and latency gate:

```text
QKV top                         2,230,737 cycles
QKV group                         369,738 cycles
PROJECT_INPUT_STEP II                    1
REDUCE_OUTPUT II                         1
output-block latency                    174 cycles
DSP                                    960
FF                                 283,348
LUT                                178,060
URAM                                   216
slack                              -0.30 ns
```

With the platform proxy included, SLR0 is approximately 31.3% DSP, 39.3% FF,
50.5% LUT and 67.5% URAM. The design therefore has sufficient placement and
routing capacity; only the small negative MAC-loop timing margin remains.

The v6.3 report implements rotating `updated` as a latency-four fabric FP32
adder. V6.4 keeps the adder in fabric and requests latency five. This avoids the
rejected accumulator-to-DSP route, adds no DSPs and remains below the true
eight-cycle recurrence distance.

V6.4 acceptance gates:

```text
PROJECT_INPUT_STEP final II = 1
REDUCE_OUTPUT final II = 1
QKV group latency <= 380,000 cycles
DSP <= 1,000
FF <= 310,000
LUT <= 190,000
URAM <= 220
QKV slack >= 0 ns
```

If Vitis does not improve slack with this explicit fabric latency, revert this
binding and test only `fmul maxdsp latency=4`; do not combine both experiments.

## V6.4 measured result and v6.5 multiplier timing

V6.4 moved the MAC submodule slack to +0.06 ns, but the top report remained
`-0.00 ns` and the requested fabric latency five was implemented as latency
four. More importantly, the explicit `updated` binding changed global operator
sharing:

```text
                         v6.3             v6.4
QKV group                369,738          369,738 cycles
slack                       -0.30            -0.00 ns
DSP                            960              960
FF                         283,348          342,127
LUT                        178,060          239,126
REDUCE_OUTPUT DSP                2              384
```

The timing gain is not worth the reduction-sharing and routing regression, so
v6.4 is rejected. V6.5 removes the `updated` binding and restores the v6.3
fabric accumulator. It changes only the three Q/K/V product arrays from the
automatic max-DSP multiplier latency three to the proven FFN timing setting
`maxdsp latency=4`. The extra multiplier stage is outside the rotating-adder
recurrence and must preserve II=1.

V6.5 acceptance gates:

```text
PROJECT_INPUT_STEP final II = 1
REDUCE_OUTPUT final II = 1
QKV group latency <= 380,000 cycles
DSP <= 1,000
FF <= 310,000
LUT <= 190,000
URAM <= 220
QKV top slack > 0 ns
REDUCE_OUTPUT DSP must return near the v6.3 value
```

## V6.5 log diagnosis and v6.6 FRP timing

V6.5 reproduced v6.3 exactly. Although the source requested multiplier latency
four, the generated operators remained 192 `maxdsp latency=3` multipliers. The
request is therefore removed.

The complete HLS log identifies the actual timing problem:

```text
PROJECT_INPUT_STEP estimated period     2.729 ns
effective delay budget                  2.433 ns
critical updated fabric fadd            2.340 ns
PROJECT_INPUT_STEP control max fanout 122,738
high-fanout expression                  ap_block_pp0_stage0_11001
project_qkv_group call path             2.532 ns
```

The arithmetic fadd alone is within the effective budget; the default stalled
pipeline control and module-call control account for the remaining margin.
V6.6 changes only `PROJECT_INPUT_STEP` from the default stalled pipeline to a
free-running pipeline:

```cpp
#pragma HLS PIPELINE II=1 style=frp
```

This is the Vitis HLS 2022.1 timing-oriented pipeline style intended to reduce
pipeline-control fanout. It is a hint, so the log must explicitly confirm that
FRP was enabled rather than silently falling back to STP.

V6.6 acceptance gates:

```text
log confirms free-running pipeline enabled for PROJECT_INPUT_STEP
PROJECT_INPUT_STEP final II = 1
REDUCE_OUTPUT final II = 1
PROJECT_INPUT_STEP max control fanout < 10,000
QKV group latency <= 380,000 cycles
DSP <= 1,000
FF <= 320,000
LUT <= 200,000
URAM <= 220
PROJECT_INPUT_STEP, project_qkv_group and QKV top slack > 0 ns
```

If Vitis rejects FRP for this sequential caller, v6.6 is not accepted and the
next timing work must introduce an explicit pipeline/module boundary rather
than changing floating-point operator bindings again.

## Attention Core v7.0 measurement gate

The available Core report predates the active source directives. It reports
`CONTEXT_KEY_CHUNK` with requested II=1 and achieved II=4, while the current
source declares the two context-bank inter dependencies false and requests
II=3. No architecture change may be layered on top until this source is
synthesized independently.

Using the old report dimensions, achieving II=3 should reduce each 512-trip
context loop by about 510 cycles. Across 64 query pairs and two parallel head
instances, the estimated Core group reduction is about 65,000 cycles:

```text
old Core group       439,309 cycles
estimated v7.0 Core  374,000 cycles
```

With QKV v6.3 at 369,738 cycles/group and Out at 413,134 cycles/group, this
would estimate the twelve-layer model near 198 ms. V7.0 acceptance gates are:

```text
CONTEXT_KEY_CHUNK final II <= 3
Core group latency <= 390,000 cycles
DSP <= 1,000
FF <= 320,000
LUT <= 200,000
URAM <= 80
Core slack must improve from -0.69 ns or identify a new exact critical path
```

## Core v7.0 measured result and v7.1 rotating EXP sum

Core v7.0 reached the predicted context target:

```text
Core top                         2,245,853 cycles
Core group                         374,285 cycles
CONTEXT_KEY_CHUNK final II                3
DSP                                      178
FF                                   178,250
LUT                                  138,234
URAM                                      24
slack                                -0.24 ns
```

The complete log shows that context now has +0.09 ns slack. The remaining
negative timing and II=4 are both in `EXP_KEY`, specifically the dynamic
`sum_lane0/1[n & 7]` accumulation; `hls::expf` is not the reported critical
operation. V7.1 replaces those two arrays with static eight-register rotating
accumulators.

Because `active_count` can be any value from zero to 128, the final physical
register rotation is reversed before `fp32_sum8_tree`. Logical bank `r` is read
from physical bank `(r - (active_count & 7)) & 7`. This preserves each bank's
FP32 addition sequence and restores the original bank order before the balanced
sum tree. Bitwise equivalence was checked for every active count from 0 through
128.

V7.1 acceptance gates:

```text
EXP_KEY final II = 1
EXP_KEY latency <= 170 cycles
CONTEXT_KEY_CHUNK final II = 3
Core group latency <= 335,000 cycles
DSP <= 250
FF <= 230,000
LUT <= 175,000
URAM <= 30
Core slack >= 0 ns
```
