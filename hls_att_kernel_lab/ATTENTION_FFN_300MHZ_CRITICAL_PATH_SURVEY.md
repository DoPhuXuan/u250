# BERT Attention + FFN 300 MHz critical-path survey

Ngày survey: 2026-07-13

Mục tiêu của survey này là xác định critical path thật trước khi tiếp tục sửa
source. Không dùng custom clock uncertainty và không xem HLS slack âm là bằng
chứng timing failure nếu RTL synthesis hoặc post-route chưa xác nhận.

## 1. Thứ tự bằng chứng

```text
1. HLS csynth
   - dùng để tìm module/loop/operator nghi ngờ
   - default Vitis HLS 2022.1 uncertainty 27%
   - effective scheduler budget ở 300 MHz khoảng 2.433 ns

2. RTL logic synthesis: export_design -flow syn
   - kiểm tra netlist ở clock thật 3.333 ns
   - chưa có placement và route delay thật

3. Full-link/post-route
   - bằng chứng cuối cho WNS/TNS, congestion và SLR crossing
```

Một kernel chỉ cần sửa datapath timing khi RTL synthesis hoặc post-route xác
nhận path âm. Việc ép HLS slack thành số dương không phải mục tiêu độc lập.

## 2. Survey kiến trúc tổng thể

```text
SLR0 QKV
  -> SLR1 score/mask/softmax/context
  -> SLR2 output projection/residual/LN
  -> SLR2 FFN UP/GELU
  -> SLR3 FFN DOWN
```

Giữa các Attention kernel, sáu head-group chạy pipeline. QKV là throughput
bottleneck với 369,738 cycles/group. Core và Out/Norm đã nhanh hơn nên không
tăng parallelism của chúng chỉ để sửa timing.

| SLR | Khối | Timing evidence hiện có | Critical class | Trạng thái |
|---|---|---|---|---|
| SLR0 | QKV | HLS -0.30 ns; RTL synthesis **+0.672 ns** | control/address net tới K-weight URAM | Đạt 300 MHz ở synthesis; khóa source |
| SLR1 | Attention Core | HLS -0.30 ns; RTL synthesis **+0.346 ns** | FSM/control tới LUTRAM write-enable của context bank | Đạt 300 MHz ở synthesis; khóa source |
| SLR2 | Out/Norm | HLS margin không dùng để closure; RTL synthesis **+0.854 ns** | URAM read qua một LUT tới projection/norm register | Đạt 300 MHz ở synthesis; khóa source |
| SLR2 | FFN UP V15 | post-route path âm | internal FP multiplier/DSP cluster | Timing failure thật; WNS contributor chính |
| SLR3 | FFN DOWN V15 | post-route path âm | internal FP multiplier/DSP cluster | Timing failure thật; contributor thứ hai |
| SLR2->3 | `gelu_stream` | routed SLL 12.11%; không nằm top paths | không phải bottleneck | Không đổi topology |

## 3. SLR0 — QKV

### HLS scheduler

```text
PROJECT_INPUT_STEP estimated period  2.729 ns
effective HLS budget                 2.433 ns
HLS slack                           -0.30 ns
fabric fadd                          2.340 ns
register/store                       0.387 ns
```

Các thử nghiệm binding v6.8/v6.9 đã bị Vitis bỏ qua và không thay RTL. Không
thử thêm operator-binding variant.

### RTL synthesis

```text
required period                     3.333 ns
achieved period                     2.661 ns
setup slack                        +0.672 ns
critical datapath                   2.168 ns
logic delay                         0.404 ns
net delay                           1.764 ns
logic levels                        2
max fanout                          khoảng 192
```

Critical path thật sau synthesis đi từ control/start register tới địa chỉ
K-weight URAM. Net delay chiếm khoảng 81%, không phải FP adder.

### Quyết định

- Khóa QKV v6.7/v6.3 datapath.
- Không tăng `QKV_O_PAR` hoặc `QKV_K_PAR`; URAM SLR0 đã khoảng 67.5%.
- Không sửa fadd để làm HLS report dương.
- Chỉ mở lại QKV nếu full-link path âm nằm trong QKV. Khi đó target là fanout
  và locality của URAM address/control, không phải arithmetic binding.

## 4. SLR1 — Attention Core

Current report:

```text
top latency                         1,955,549 cycles
group latency                         325,901 cycles
DSP                                       185
FF                                    180,508
LUT                                   140,172
URAM                                       24
HLS slack                              -0.30 ns
```

RTL synthesis mới nhất:

```text
required period                      3.333 ns
achieved period                      2.987 ns
setup slack                         +0.346 ns
post-synthesis LUT                   70,459
post-synthesis FF                   126,926
post-synthesis DSP                      185
post-synthesis BRAM                       3
post-synthesis URAM                      24
```

Top path không nằm trong `EXP_KEY`. Nó đi từ FSM của
`CONTEXT_KEY_CHUNK` tới write-enable của `context_bank0` LUTRAM:

```text
logic levels                              9
max fanout                              612
datapath delay                        2.744 ns
logic delay                           0.512 ns
net delay                             2.232 ns (81.3%)
```

Như vậy chẩn đoán HLS về rotating FP sum không còn là critical path sau RTL
synthesis. Core đạt 300 MHz; không bind fadd và không đổi `EXP_KEY`.

Các hot loop:

```text
SCORE_KEY                 II=4, 602 cycles
EXP_KEY                   II=1, 157 cycles
CONTEXT_KEY_CHUNK         II=3, 1,550 cycles
CONTEXT_BANK_REDUCE       II=2, 154 cycles
```

Log lịch sử đã xác nhận `CONTEXT_KEY_CHUNK` đạt khoảng +0.09 ns sau v7.0.
Negative path còn lại chuyển sang `EXP_KEY`. V7.1 đã thay dynamic
`sum_lane[n & 7]` bằng tám register quay tĩnh và đưa II từ 4 về 1. `hls::expf`
không phải operation được log chỉ ra trong lần chẩn đoán đó.

### Gate trước khi sửa — đã hoàn thành

Chạy RTL synthesis Core tại 3.333 ns và lấy:

- WNS/setup slack.
- startpoint/endpoint.
- logic delay so với net delay.
- có nằm trong fabric fadd, `fexp`, mux phase, control hay URAM address không.

Kết quả gate: timing đạt. Path là control/FSM tới LUTRAM write-enable. Không
thực hiện source experiment. Biên +0.346 ns nhỏ hơn QKV và Out/Norm, nên đây là
path cần theo dõi đầu tiên khi full-link, nhưng chưa đủ lý do để đổi lịch
`CONTEXT_KEY_CHUNK` II=3.

### Hướng tối ưu chỉ khi RTL synthesis âm

1. Nếu path là `EXP_KEY` rotating fabric fadd:
   - tách đúng hai accumulator add thành module `INLINE off` có ownership rõ;
   - thử full-DSP fadd native latency cho đúng hai phép cộng này;
   - giữ tám recurrence lanes nên latency tối đa phải nhỏ hơn hoặc bằng 8;
   - bắt buộc giữ `EXP_KEY II=1` và numerical addition order.
2. Nếu path là phase/reorder mux sau accumulator:
   - tách đọc logical bank khỏi pipelined `EXP_KEY` boundary;
   - register `sum_phase` và các physical bank trước mux;
   - không tăng `ATTN_D_PAR`.
3. Nếu path là high-fanout pipeline control:
   - tách `EXP_KEY` thành non-inline module trước;
   - kiểm tra fanout giảm trong RTL synthesis rồi mới giữ thay đổi.
4. Nếu RTL synthesis dương:
   - khóa Core v7.1; không tối ưu HLS slack.

Acceptance guard cho mọi Core experiment:

```text
RTL synthesis slack                 >= 0 ns at 3.333 ns
EXP_KEY II                           = 1
CONTEXT_KEY_CHUNK II                <= 3
group latency                       <= 335,000 cycles
DSP                                 <= 250
FF                                  <= 230,000
LUT                                 <= 175,000
URAM                                <= 30
```

## 5. SLR2 — Attention Out/Norm

Locked datapath:

```text
top latency                         1,365,434 cycles
group latency                         215,758 cycles
projection loop II                           2
DSP                                       393
FF                                     98,437
LUT                                    91,362
URAM                                       32
```

Projection không phải timing bottleneck. Lịch sử default-margin report chỉ ra:

```text
module                                NORM_REDUCE_PACK
estimated period                      2.470 ns
effective HLS budget                  2.433 ns
HLS slack                            -0.04 ns
critical operation                    sum_bank fabric fadd
operation delay                       2.080 ns
operator latency                      4 cycles
final loop II                         4
```

Report top hiện tại `0.00 ns` đến từ lần thử uncertainty 25%; không dùng nó để
kết luận timing. Tcl hiện không còn đặt custom uncertainty, do đó cần rerun.

RTL synthesis mới nhất đo trực tiếp clock 3.333 ns và đã đạt:

```text
required period                      3.333 ns
achieved period                      2.479 ns
setup slack                         +0.854 ns
post-synthesis LUT                   54,483
post-synthesis FF                    80,174
post-synthesis DSP                      393
post-synthesis BRAM                     107
post-synthesis URAM                      32
```

Top paths đi từ `projected` URAM qua một LUT tới register trong projection và
`NORM_REDUCE_PACK`:

```text
logic levels                              1
max fanout                                3
datapath delay                        2.406 ns
logic delay                           2.131 ns
net delay                             0.275 ns
```

Fabric norm fadd không còn là worst path của synthesized RTL. Biên +0.854 ns
đủ lớn để khóa Out/Norm ở bước này.

### Gate trước khi sửa — đã hoàn thành

1. Csynth lại ở default 27% margin.
2. RTL synthesis Out/Norm tại clock thật 3.333 ns.
3. Kiểm tra path có còn nằm trong norm recurrence hay chuyển sang projection,
   rsqrt, AXI write/control hoặc FFN-UP locality.

Kết quả gate: RTL timing đạt; path chuyển sang local URAM-output/register path.
Không thực hiện norm binding hoặc tăng projection resources.

### Hướng tối ưu chỉ khi RTL synthesis âm

1. Nếu path vẫn là norm fabric fadd:
   - giữ projection II=2 nguyên vẹn;
   - tách riêng norm accumulator ownership bằng module non-inline;
   - thử targeted implementation trên norm accumulator, không bind toàn kernel;
   - chấp nhận tăng norm II/tail nhẹ vì Out vẫn nhanh hơn QKV.
2. Nếu full-DSP fadd làm tăng routing trên SLR2:
   - ưu tiên register boundary/local fabric solution;
   - SLR2 đã khoảng 54.6% DSP khi cộng FFN UP, nên không nhân rộng DSP adders
     nếu synthesis path đã dương.
3. Không thay reduction order hoặc approximate `rsqrtf` nếu chưa có accuracy
   regression cho numerical change.
4. Nếu RTL synthesis dương:
   - khóa Out/Norm projection II=2 và norm hiện tại.

Acceptance guard:

```text
RTL synthesis slack                 >= 0 ns at 3.333 ns
ACCUMULATE_ROW_BLOCK II              = 2
group latency                       <= 220,000 cycles
top latency                         <= 1,380,000 cycles
DSP                                 <= 420
FF                                  <= 120,000
LUT                                 <= 105,000
URAM                                = 32
```

## 6. SLR2/SLR3 — FFN timing thật

V15 routed baseline có:

```text
WNS                                 -0.033 ns
TNS                                 -5.784 ns
failing setup endpoints             336
hold slack                          +0.005 ns
```

Top failing paths:

- Chủ yếu nằm trong FP multiplier/DSP cluster của FFN UP trên SLR2.
- Một phần nằm trong FP multiplier/DSP cluster của FFN DOWN trên SLR3.
- Route chiếm khoảng 62–65% path delay ở các path tiêu biểu.
- Không phải AXIS crossing, DDR adapter, URAM/BRAM hoặc SLL capacity.

Đây là timing issue cần tối ưu thật sau khi Attention RTL survey hoàn tất.

### Hướng tối ưu FFN ưu tiên

1. Không đổi SLR placement hoặc stream topology.
2. Dùng separate dot-product hierarchy để implementation directive không bị
   mất do inline.
3. Chỉ chấp nhận multiplier variant nếu generated RTL/IP thực sự đổi và hot
   loop vẫn II=1.
4. Vì WNS chỉ -0.033 ns và route chiếm đa số, đánh giá physical optimization
   hoặc register/locality quanh multiplier trước khi giảm parallelism.
5. Full-link/post-route là bắt buộc để chốt; csynth `0.00` không đủ.

Workspace FFN có nhiều report mang tên V21 với resource/operator inventory
khác nhau. Trước khi link, phải chốt đúng source hash, top name, `.xo`, report
và connectivity config cùng một version; không trộn số liệu giữa các V21.

Artifact audit hiện tại xác nhận chưa đồng nhất:

```text
ffn_kernel_u250.cfg               vẫn chọn V15 UP/DOWN
ffn_kernel_u250_v21.cfg           chọn V21 dotpipe UP/DOWN
bert_ffn_up_v21.xo                timestamp 2026-07-13
bert_ffn_down_v21.xo              timestamp 2026-07-13
csynth_up_v21.rpt trong FFN lab   báo khoảng 17.18M cycles
csynth_down_v21.rpt trong FFN lab báo khoảng 17.16M cycles
V21 report copy trong ATT lab     báo khoảng 1.54M/1.66M cycles
```

Do đó chưa được link các `.xo` V21 hiện tại rồi gọi đó là candidate 1.5M-cycle.
Bước FFN tiếp theo phải là clean rebuild một top/config duy nhất và lưu report
cùng build ID. Nếu latency sạch không nằm trong budget, quay lại V15 thay vì
tối ưu timing cho một V21 sai schedule.

## 7. Thứ tự hành động

### Pass A — Attention RTL survey, không đổi datapath — hoàn thành

Tcl đã được sửa để chạy `export_design -flow syn -rtl verilog` cho cả QKV,
Core và Out/Norm.

```bash
cd /home/minhphat/att
source /tools/Xilinx/Vitis_HLS/2022.1/settings64.sh
unset ATTENTION_CLOCK_MHZ
vitis_hls -f scripts/run_vitis_attention_csynth.tcl
```

`unset` bảo đảm script dùng default 300 MHz. Expected report directories:

```text
reports/rtl_syn/bert_qkv_kernel/
reports/rtl_syn/bert_attn_core_kernel/
reports/rtl_syn/bert_attn_out_norm_kernel/
```

Measured result:

```text
QKV       achieved 2.661 ns, slack +0.672 ns
Core      achieved 2.987 ns, slack +0.346 ns
Out/Norm  achieved 2.479 ns, slack +0.854 ns
```

Decision:

```text
Core RTL >= 0 và Out RTL >= 0
    -> điều kiện này đã đạt.
    -> khóa toàn bộ Attention; chuyển sang FFN routed timing.

Chỉ một kernel Attention RTL < 0
    -> sửa duy nhất kernel/path đó theo classification ở trên.

Cả hai RTL < 0
    -> ưu tiên path có slack xấu hơn; không sửa đồng thời hai source.
```

### Pass B — isolated source experiment

Mỗi pass chỉ thay một path class và phải so sánh:

```text
RTL path/startpoint/endpoint
RTL slack
latency và II
DSP/FF/LUT/BRAM/URAM
operator inventory
```

Pass B không cần thực hiện cho Attention vì cả ba RTL synthesis đều dương.

### Pass C — FFN candidate consistency và route

Sau khi Attention đạt RTL synthesis:

1. Chốt một FFN version duy nhất.
2. Rebuild đúng UP/DOWN `.xo` ở 300 MHz.
3. Full-link trên SLR2/SLR3 hoặc full 4-SLR system.
4. Chỉ kết luận đạt 300 MHz khi post-route WNS >= 0.

## 8. Kết luận survey

```text
QKV       khóa; +0.672 ns ở RTL synthesis.
Core      khóa; +0.346 ns, worst path là control -> LUTRAM WE.
Out/Norm  khóa; +0.854 ns, worst path là local URAM read -> register.
FFN UP    timing target thật số 1 theo routed evidence.
FFN DOWN  timing target thật số 2 theo routed evidence.
Crossing  không phải bottleneck hiện tại.
```

Do đó không thay đổi source Attention nữa. Hướng tối ưu timing tiếp theo chuyển
sang FFN UP/DOWN và phải bắt đầu bằng việc chốt một V21 artifact set nhất quán:
source, top name, csynth report, `.xo` và connectivity config phải cùng version.

## 9. QKV v7.0 HLS-slack experiment

Yêu cầu bổ sung là csynth ở 300 MHz/default 27% uncertainty cũng phải hết
`-0.30 ns`, dù RTL synthesis hiện tại đã đạt timing. Vì vậy QKV được mở lại cho
một pass cô lập; acceptance vẫn không dựa vào thay đổi uncertainty.

Critical HLS path v6.7:

```text
last V rotating accumulator fabric fadd  2.340 ns
register/store                           0.387 ns
total estimated period                  2.729 ns
effective HLS budget                    2.433 ns
```

V7.0 giữ 191 accumulator đang được HLS map tốt và chỉ tách accumulator V cuối
thành module `qkv_last_v_accumulate`:

```text
INLINE off
function PIPELINE II=1
fadd fulldsp latency=7
```

Điểm khác với v6.8/v6.9 là operator có hierarchy/ownership riêng; directive
không còn nằm trong helper inline hoặc direct parent scope đã từng bị Vitis bỏ
qua. Latency 7 nhỏ hơn recurrence distance 8 nên về nguyên tắc không phá II=1.
Phép cộng và thứ tự rotate FP32 không đổi.

Acceptance gate:

```text
qkv_last_v_accumulate xuất hiện thành module riêng
updated implements as full-DSP fadd latency=7
PROJECT_INPUT_STEP final II = 1
REDUCE_OUTPUT final II = 1
QKV csynth slack >= 0 ns với default 27% uncertainty
QKV group latency <= 375,000 cycles
QKV top latency <= 2,250,000 cycles
DSP <= 970
FF <= 310,000
LUT <= 190,000
URAM = 216
RTL synthesis slack >= 0 ns at 3.333 ns
```

Nếu module call làm II tăng hoặc binding vẫn bị bỏ qua, revert v7.0. Không thử
thêm global fadd binding; hướng tiếp theo sẽ là explicit producer/consumer
pipeline stage quanh accumulator, vì vấn đề khi đó là scheduling boundary chứ
không phải chọn DSP implementation.

## 10. Core v7.2 HLS-slack experiment

Core có cùng operator asymmetry: phần lớn FP adders dùng full-DSP nhưng một
fabric fadd còn lại giữ csynth top slack ở `-0.30 ns`. V7.2 giữ nguyên
`hls::expf`, probability storage, context banking và mọi loop II. Chỉ
`sum_acc1` trong `EXP_KEY` được tách thành
`core_last_sum_accumulate` với:

```text
INLINE off
function PIPELINE II=1
fadd fulldsp latency=7
```

`HEAD_PAR=2` có thể tạo hai instance của helper. Resource tăng dự kiến nhỏ và
SLR1 còn nhiều DSP headroom. Addition order của từng rotating bank không đổi.

Acceptance gate:

```text
core_last_sum_accumulate xuất hiện thành module riêng
updated implements as full-DSP fadd latency=7
EXP_KEY final II = 1
CONTEXT_KEY_CHUNK final II = 3
Core csynth slack >= 0 ns với default 27% uncertainty
Core group latency <= 335,000 cycles
Core top latency <= 2,020,000 cycles
DSP <= 200
FF <= 210,000
LUT <= 160,000
URAM = 24
RTL synthesis slack >= 0 ns at 3.333 ns
```

Out/Norm không nhận source change trong pass này. Nó không có `-0.30 ns`, và
RTL synthesis đã có +0.854 ns. Sau QKV/Core csynth, chỉ mở Out/Norm lại nếu
default-margin report mới thực sự còn âm và người dùng vẫn yêu cầu HLS slack
dương cho cả warning nhỏ đó.
