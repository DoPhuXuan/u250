# BERT Attention + FFN current split-kernel architecture

Ngày tổng hợp: 2026-07-13

Tài liệu này ghi lại trạng thái hiện tại của các khối Attention và FFN cho
BERT-base trên Alveo U250. Mục đích là chốt kiến trúc split-kernel, latency
budget và ước lượng tài nguyên theo từng SLR trước khi full-link.

Các con số trong tài liệu được gắn mức bằng chứng rõ ràng:

- **Attention csynth:** Vitis HLS 2022.1, clock 300 MHz.
- **QKV RTL synthesis:** Vivado logic synthesis từ `export_design -flow syn`;
  chưa place-and-route.
- **FFN routed baseline:** V15 trong architecture handoff, đã full-link và
  post-route; report gốc dùng tool 2021.2.
- **FFN latency reference:** report V21 hiện có trong workspace; V21 chưa được
  chấp nhận làm route candidate cuối.
- **SLR2 combined estimate:** cộng routed SLR2 baseline với standalone csynth
  của Attention Out/Norm. Đây là proxy bảo thủ, không thay thế utilization sau
  full-link.

## 1. Target hệ thống

```text
Board                  Alveo U250
Part                   xcu250-figd2104-2L-e
Attention HLS tool     Vitis HLS 2022.1
Clock                  300 MHz (3.333 ns)
Encoder layers         12
Sequence length        128
Hidden size            768
Attention heads        12
Head dimension         64
Intermediate size      3072
External stream width  512 bit = 16 x float32
Model latency target   < 200 ms
Total cycle budget     < 60,000,000 cycles
Per-layer budget       < 5,000,000 cycles
Route-friendly target  cố gắng giữ mỗi tài nguyên dưới khoảng 70%/SLR
```

Kích thước tensor chính:

```text
hidden/context tensor  128 * 768  / 16 = 6,144 word 512-bit
FFN intermediate       128 * 3072 / 16 = 24,576 word 512-bit
```

## 2. Kiến trúc split-kernel hiện tại

```text
DDR input + Q/K/V weights
          |
          v
SLR0  bert_qkv_kernel
      - fused Q, K, V projection
      - xử lý 2 head/group, tổng cộng 6 group
          |
          | q_stream + k_stream + v_stream
          | 512 bit, group order
          v
SLR1  bert_attn_core_kernel
      - QK^T
      - attention mask
      - softmax FP32
      - weighted sum tạo context
          |
          | context_stream
          | 512 bit, row-major [sequence][hidden_pack]
          v
SLR2  bert_attn_out_norm_kernel
      - output projection
      - residual add
      - attention LayerNorm
      - ghi attn_mid_ddr và phát attn_mid_stream
          |
          | attn_mid_stream: 6,144 word, row-major
          v
SLR2  bert_ffn_up_gelu_*_kernel
      - projection 768 -> 3072
      - GELU
          |
          | gelu_stream: 24,576 word
          | order [intermediate_tile][sequence]
          v
SLR3  bert_ffn_down_*_kernel
      - projection 3072 -> 768
      - output hiện tại ghi DDR
```

Phân bố cố định:

| SLR | Kernel/khối | Vai trò |
|---|---|---|
| SLR0 | `bert_qkv_kernel` | Fused Q/K/V projection |
| SLR1 | `bert_attn_core_kernel` | Score, mask, softmax, weighted sum |
| SLR2 | `bert_attn_out_norm_kernel` | Output projection, residual, Attention LayerNorm |
| SLR2 | FFN UP + GELU | Projection 768 -> 3072 và activation |
| SLR3 | FFN DOWN | Projection 3072 -> 768 |

FFN-only feeder trên SLR2 là test infrastructure. Khi full-link, output
`attn_mid_stream` của Out/Norm thay feeder và nối trực tiếp vào FFN UP.

## 3. Cấu hình song song Attention đã chốt

```text
HEAD_PAR             2
HEAD_GROUPS          6
QKV_TILE_O          64
QKV_K_PAR            8
QKV_O_PAR            8
QKV_ACC_BANKS        8
ATTN_D_PAR           8
ATTN_CONTEXT_BANKS  16
ATTN_ACC_BANKS       8
OUT_TILE_O          32
OUT_K_PAR            8
OUT_O_PAR            8
```

Các cấu hình này tạo pipeline sáu head-group:

```text
QKV group g -> Core group g -> Out projection accumulation group g
```

## 4. Tài nguyên và latency từng kernel

### 4.1 Attention — csynth hiện tại

| Kernel | Top latency | Group latency | BRAM | DSP | FF | LUT | URAM | Csynth slack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV | 2,230,737 | 369,738 | 0 | 960 | 283,348 | 178,060 | 216 | -0.30 ns |
| Attention Core | 1,955,549 | 325,901 | 2 | 185 | 180,508 | 140,172 | 24 | -0.30 ns |
| Attention Out/Norm | 1,365,434 | 215,758 | 16 | 393 | 98,437 | 91,362 | 32 | 0.00 ns* |

`*` Out/Norm report hiện tại dùng effective timing budget 2.50 ns từ lần đo
uncertainty 25%. Cấu hình này đã bị rút lại. Out/Norm phải csynth lại với Tcl
hiện tại, không đặt custom clock uncertainty, trước khi dùng slack để kết luận.

Tổng standalone csynth của ba kernel Attention là:

| BRAM | DSP | FF | LUT | URAM |
|---:|---:|---:|---:|---:|
| 18 | 1,538 | 562,293 | 409,594 | 272 |

Tổng này chỉ thể hiện lượng logic của Attention trên toàn thiết bị; không dùng
để đánh giá congestion vì ba kernel được đặt ở ba SLR khác nhau.

### 4.2 QKV — RTL logic synthesis tại 300 MHz

| Mục | Kết quả |
|---|---:|
| Requested period | 3.333 ns |
| Achieved post-synthesis period | 2.661 ns |
| Setup slack | **+0.672 ns** |
| Equivalent frequency | khoảng 375.8 MHz |
| LUT | 131,396 |
| FF | 230,591 |
| DSP | 960 |
| BRAM | 109 |
| URAM | 216 |

QKV đạt timing thật ở bước logic synthesis mà không giảm clock uncertainty.
Critical path sau synthesis là control/fanout tới địa chỉ K-weight URAM, với
net delay chiếm phần lớn datapath; nó không còn là critical FP adder của HLS
scheduler. Đây chưa phải post-route timing closure.

### 4.3 FFN — latency report hiện có

| Kernel | Version/report | Top latency | BRAM | DSP | FF | LUT | URAM | Trạng thái |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FFN UP + GELU | V21 | 1,536,392 | 272 | 864 | 190,406 | 134,244 | 32 | Latency reference; chưa là final route candidate |
| FFN DOWN | V21 | 1,655,260 | 16 | 768 | 209,957 | 142,758 | 32 | Latency reference; chưa là final route candidate |

V15 vẫn là route baseline ổn định của FFN. V21 chứng minh hierarchy/II nhưng
không thay được critical multiplier RTL như dự kiến, nên không được dùng để
tuyên bố timing closure.

### 4.4 FFN V15 routed baseline

Kernel utilization sau route:

| CU | LUT | REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| FFN-only feeder | 1,716 | 4,432 | 7 | 0 | 0 |
| V15 UP | 71,610 | 134,107 | 143 | 32 | 1,280 |
| V15 DOWN | 79,150 | 163,887 | 23 | 32 | 1,280 |

Full-SLR utilization của routed FFN-only design:

| SLR | LUT | REG | BRAM tile | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| SLR2 | 113,545 (26.28%) | 194,519 (22.51%) | 247 (36.76%) | 32 (10.00%) | 1,283 (41.76%) |
| SLR3 | 108,725 (25.17%) | 206,822 (23.94%) | 87 (12.95%) | 32 (10.00%) | 1,283 (41.76%) |

Timing routed baseline:

```text
WNS  = -0.033 ns
TNS  = -5.784 ns
WHS  =  0.005 ns
setup failing endpoints = 336
```

Critical paths là FP multiplier/DSP nội bộ UP và DOWN, không phải AXIS/SLL.

## 5. Tài nguyên gộp theo từng SLR

### 5.1 Bảng đánh giá hiện tại

| SLR | Thành phần | LUT | FF/REG | BRAM | URAM | DSP | Đánh giá |
|---|---|---:|---:|---:|---:|---:|---|
| SLR0 | QKV standalone csynth | 178,060 | 283,348 | 0 | 216 | 960 | URAM là giới hạn chính |
| SLR1 | Core standalone csynth | 140,172 | 180,508 | 2 | 24 | 185 | Còn nhiều headroom |
| SLR2 | Routed FFN SLR2 baseline + Out/Norm csynth | 204,907 | 292,956 | 263 | 64 | 1,676 | Proxy bảo thủ, mọi tài nguyên dưới 55% trừ cách đếm BRAM phụ thuộc stage |
| SLR3 | Routed FFN SLR3 baseline | 108,725 | 206,822 | 87 | 32 | 1,283 | Còn headroom cho FFN residual/LN |

Ước lượng tỷ lệ SLR2 sau khi cộng Out/Norm:

| Resource | Combined estimate | Tỷ lệ SLR2 |
|---|---:|---:|
| LUT | 204,907 | 47.43% |
| FF/REG | 292,956 | 33.91% |
| BRAM tile | 263 | 39.14% |
| URAM | 64 | 20.00% |
| DSP | 1,676 | 54.56% |

Đây là phép cộng khác mức báo cáo: routed full-SLR baseline cộng standalone
HLS estimate. Nó thích hợp để kiểm tra headroom ban đầu nhưng không phải con số
utilization cuối cùng. Chỉ full-link mới loại bỏ double-counting/static overhead
và cho biết congestion thật.

SLR0 proxy trong lịch sử tối ưu, khi tính thêm platform overhead, xấp xỉ:

```text
DSP   31.3%
FF    39.3%
LUT   50.5%
URAM  67.5%
```

Vì URAM đã gần ngưỡng route-friendly 70%, không tăng QKV parallelism nếu chưa
có lý do latency bắt buộc.

## 6. Latency budget dưới 200 ms

Gọi:

```text
Q = 369,738 cycles/group
C = 325,901 cycles/group
O = 215,758 cycles/group
```

Sáu group đi qua ba Attention kernel theo pipeline. Công thức estimate:

```text
attention_group_pipeline = Q + C + O + 5 * max(Q, C, O)
norm_tail                 = Out top - 6 * O
attention_cycles          = attention_group_pipeline + norm_tail
ffn_cycles                = max(FFN_UP, FFN_DOWN)
layer_cycles              = attention_cycles + ffn_cycles
model_cycles              = 12 * layer_cycles
model_ms                  = model_cycles / 300,000
```

Kết quả hiện tại:

| Thành phần | Cycles |
|---|---:|
| Attention group pipeline | 2,760,087 |
| Out/Norm tail | 70,886 |
| Attention total/layer | 2,830,973 |
| FFN critical stage/layer | 1,655,260 |
| **Tổng một encoder layer** | **4,486,233** |
| Per-layer budget | 5,000,000 |
| **12-layer estimate** | **53,834,796** |
| Total budget | 60,000,000 |
| **Estimated latency @ 300 MHz** | **179.45 ms** |
| **Budget margin** | **6,165,204 cycles = 20.55 ms** |

Model estimate đạt mục tiêu dưới 200 ms với khoảng 10.3% margin. Đây là
compute/pipeline estimate, chưa bao gồm mọi chi phí host launch, DDR contention,
full-link backpressure hoặc embedding/task head.

QKV là bottleneck của Attention vì có group latency lớn nhất. Core và Out/Norm
đã nhanh hơn QKV, nên giảm II hoặc tăng parallelism ở hai kernel này không làm
tăng throughput Attention hiện tại.

## 7. Stream contract và crossing SLR

| Crossing | Stream | Width | Word count | Logical order |
|---|---|---:|---:|---|
| SLR0 -> SLR1 | `q_stream` | 512 bit | 6,144 | six head-groups, packed FP32 |
| SLR0 -> SLR1 | `k_stream` | 512 bit | 6,144 | six head-groups, packed FP32 |
| SLR0 -> SLR1 | `v_stream` | 512 bit | 6,144 | six head-groups, packed FP32 |
| SLR1 -> SLR2 | `context_stream` | 512 bit | 6,144 | row-major `[sequence][hidden_pack]` |
| SLR2 Attention -> FFN UP | `attn_mid_stream` | 512 bit | 6,144 | row-major `[sequence][hidden_pack]` |
| SLR2 -> SLR3 | `gelu_stream` | 512 bit | 24,576 | `[intermediate_tile][sequence]` |

V15 đã route crossing SLR2 -> SLR3 với SLL utilization
`2,789 / 23,040 = 12.11%`. Không thay đổi order của `gelu_stream` nếu không sửa
đồng bộ FFN DOWN consumer.

## 8. Trạng thái chốt và phần còn thiếu

### Đã có

- Fused QKV projection trên SLR0.
- Attention score, mask, softmax và weighted sum trên SLR1.
- Attention output projection, residual và LayerNorm trên SLR2.
- FFN UP + GELU trên SLR2.
- FFN DOWN projection trên SLR3.
- ABI stream 512-bit giữa các kernel Attention.
- Latency estimate dưới 200 ms.
- QKV đạt logic-synthesis timing tại 300 MHz.
- Routed FFN baseline và SLR2-SLR3 stream crossing.

### Cần xác nhận trước full-link

1. Chạy `export_design -flow syn -rtl verilog` cho Core.
2. Csynth lại Out/Norm với default Vitis HLS uncertainty, sau đó export RTL
   synthesis như Core và QKV.
3. Chốt FFN route candidate: V15 vẫn là baseline; V21 chưa được chấp nhận.
4. Full-link cả bốn SLR và lấy per-SLR utilization/congestion report.
5. Xác nhận FIFO depth và không có deadlock/backpressure giữa các kernel.
6. Chạy accuracy regression cho Attention, softmax, GELU và LayerNorm.

### Chưa có trong datapath hoàn chỉnh

- FFN residual add và LayerNorm sau DOWN. Architecture handoff đề xuất fork
  Attention residual output từ SLR2 sang SLR3 rồi tích hợp vào DOWN buffer.
- Embedding đầu vào.
- Điều phối weights/activation qua 12 encoder layer.
- Pooler hoặc task-specific output head.
- End-to-end hardware runtime bao gồm DDR và host overhead.

## 9. Các kernel còn phải làm

Phần Attention đã có đủ ba kernel compute. FFN đã có UP và DOWN projection,
nhưng DOWN chưa có residual add và LayerNorm cuối encoder layer. Vì vậy roadmap
được chia thành kernel bắt buộc cho encoder và kernel tùy chọn theo task.

### 9.1 Danh sách tổng hợp

| Ưu tiên | Kernel/công việc | SLR | Bắt buộc | Latency target | Resource tăng tối đa dự kiến |
|---:|---|---|---|---:|---|
| 1 | Fuse residual + LayerNorm vào FFN DOWN | SLR3 | Có | tail <= 70,000 cycles/layer | 16 BRAM, 16 DSP, 25k FF, 15k LUT, không thêm URAM nếu reuse buffer |
| 2 | Thêm `ffn_residual_stream` vào Out/Norm và DOWN | SLR2 -> SLR3 | Có | II=1, overlap; không cộng tail tuần tự | FIFO/crossing nhỏ, không thêm compute DSP |
| 3 | `bert_embedding_norm_kernel` | SLR0 | Có cho full BERT | <= 100,000 cycles/model | 32 BRAM, 16 DSP, 25k FF, 20k LUT, không thêm URAM |
| 4 | Điều phối 12 encoder layer | Host/control | Có | overhead target <= 1,000,000 cycles/model | Không chiếm compute SLR đáng kể |
| 5 | `bert_pooler_classifier_kernel` | SLR3 | Chỉ classification | <= 50,000 cycles/model | 128 DSP, 60k FF, 40k LUT, 16 URAM |
| 6 | Full-link/connectivity build | Cả 4 SLR | Có | không làm giảm throughput group | Không phải kernel compute mới |

Các resource trên là **trần thiết kế ban đầu**, chưa phải kết quả csynth. Kernel
mới chỉ được chấp nhận nếu không vượt trần và tổng SLR vẫn giữ dưới khoảng 70%
khi có thể.

### 9.2 Bắt buộc: FFN DOWN + residual + LayerNorm trên SLR3

Không nên tạo một LayerNorm kernel riêng vì sẽ cần thêm buffer/stream boundary
và đọc lại toàn bộ hidden tensor. Hướng ưu tiên là đổi kernel DOWN hiện tại
thành một kernel fused, ví dụ:

```text
bert_ffn_down_residual_norm_kernel
```

Datapath:

```text
gelu_stream ----------------------+
                                  v
W2/B2 -> FFN DOWN projection -> residual add -> LayerNorm -> output
                                  ^
                                  |
ffn_residual_stream --------------+
```

Interface dự kiến:

```text
input  gelu_stream          512 bit, 24,576 word, tile-major
input  ffn_residual_stream  512 bit,  6,144 word, row-major
input  W2/B2                DDR[3]
input  final LN gamma/beta  DDR[3]
output hidden              DDR[0] hoặc 512-bit stream cho layer kế tiếp
```

Implementation:

- Reuse `output_buf` URAM hiện có của DOWN.
- Không tạo full residual buffer thứ hai nếu residual stream có thể được drain
  vào đúng lịch khởi tạo/output buffer.
- Giữ hot projection loop II=1.
- LayerNorm có thể dùng cấu trúc reduction-bank tương tự Attention Out/Norm.
- Không thay đổi GELU stream order.

Latency budget:

| Thành phần | Target |
|---|---:|
| FFN DOWN projection reference | 1,655,260 cycles |
| Residual + LayerNorm tail | <= 70,000 cycles |
| **Fused DOWN total** | **<= 1,725,260 cycles** |

Resource tăng cho phần residual/LN được lấy bảo thủ từ khối
`residual_norm_and_write` đang có ở Attention Out/Norm:

| Resource | Increment budget | SLR3 combined estimate sau khi thêm |
|---|---:|---:|
| LUT | <= 15,000 | <= 123,725, khoảng 28.6% SLR3 |
| FF | <= 25,000 | <= 231,822, khoảng 26.8% SLR3 |
| BRAM | <= 16 | <= 103, khoảng 15.3% SLR3 |
| URAM | 0 nếu reuse buffer | 32, khoảng 10.0% SLR3 |
| DSP | <= 16 | <= 1,299, khoảng 42.3% SLR3 |

SLR3 còn đủ headroom. Acceptance gate quan trọng hơn resource là timing của FP
multiplier và rsqrt/normalization tại 300 MHz.

### 9.3 Bắt buộc: residual stream từ SLR2 sang SLR3

Attention Out/Norm đã tạo hidden tensor sau Attention residual/LN. Tensor này
vừa là input FFN UP, vừa là residual phải cộng lại sau FFN DOWN.

Hướng ưu tiên là cho `bert_attn_out_norm_kernel` phát hai stream đồng thời:

```text
attn_mid_stream       -> FFN UP trên SLR2
ffn_residual_stream   -> fused FFN DOWN trên SLR3
```

Contract:

```text
width       512 bit
word count  6,144
order       [sequence][hidden_pack]
producer II 1
consumer    fused DOWN phải drain stream thật
```

Không tạo standalone fork kernel nếu Out/Norm có thể duplicate mỗi word khi
ghi. Việc duplicate phải overlap với output loop, nên latency tăng kỳ vọng gần
0 cycle; không được cộng thêm 6,144 cycles tuần tự vào critical layer path.

Chỉ thêm stream sau khi DOWN đã có consumer. Stream không được drain sẽ gây
deadlock toàn pipeline.

### 9.4 Bắt buộc cho full BERT: embedding trên SLR0

Kernel đề xuất:

```text
bert_embedding_norm_kernel
```

Chức năng:

```text
token embedding lookup
+ position embedding lookup
+ token-type embedding lookup
-> embedding LayerNorm
-> hidden tensor đầu vào layer 0
```

Placement: **SLR0**, gần QKV và DDR bank chứa input/embedding table. Kernel chỉ
chạy một lần trước 12 encoder layer nên không overlap lâu dài với QKV, nhưng
logic vẫn cùng chiếm tài nguyên vật lý SLR0.

Interface dự kiến:

```text
input  input_ids/token_type_ids/attention mask
input  token/position/type embedding tables từ DDR[0]
input  embedding LN gamma/beta
output hidden_in DDR[0] và hidden_ready token
```

Budget:

| Mục | Target |
|---|---:|
| Latency | <= 100,000 cycles = 0.333 ms/model |
| LUT | <= 20,000 |
| FF | <= 25,000 |
| DSP | <= 16 |
| BRAM | <= 32 |
| URAM | 0; tránh tăng SLR0 URAM đang ở 67.5% |

Kernel này thiên về DDR lookup, cộng FP và LayerNorm. Không cần unroll lớn vì
chỉ chạy một lần/model và không nằm trên per-layer throughput bottleneck.

### 9.5 Bắt buộc: điều phối 12 encoder layer

Đây nên là host/runtime control, không phải một compute kernel mới. Scheduler
cần:

1. Chọn offset W/Q/K/V/O, FFN và LayerNorm theo `layer_id`.
2. Launch/giữ đồng thời các producer-consumer kernel để stream không deadlock.
3. Đưa output layer `n` thành hidden/residual input layer `n+1`.
4. Ping-pong DDR buffer nếu chưa nối layer-to-layer hoàn toàn bằng stream.
5. Kiểm tra `hidden_ready`, `residual_ready`, `attn_mid_done` đúng protocol.

Overhead budget ban đầu:

```text
<= 1,000,000 cycles/model = 3.333 ms
```

Nếu host launch overhead vượt budget này, mới xem xét persistent kernel hoặc
device-side controller. Không tạo controller kernel trước khi đo.

### 9.6 Tùy chọn: pooler và classification head trên SLR3

Chỉ cần nếu mục tiêu là sequence classification. Encoder-only benchmark không
cần kernel này.

Kernel đề xuất:

```text
bert_pooler_classifier_kernel
```

Chức năng:

```text
đọc hidden vector của token [CLS]
-> dense 768 x 768
-> tanh
-> task classifier 768 x num_labels
-> logits
```

Placement: **SLR3**, ngay sau output cuối của fused DOWN. Kernel chạy một lần
cho toàn model, nên có thể chia sẻ bandwidth DDR[0]/DDR[3] theo lịch.

Budget cho classification head nhỏ:

| Mục | Target |
|---|---:|
| Latency | <= 50,000 cycles = 0.167 ms/model |
| LUT | <= 40,000 |
| FF | <= 60,000 |
| DSP | <= 128 |
| BRAM | <= 32 |
| URAM | <= 16 |

Với NER/token classification hoặc question answering, head phải xử lý toàn bộ
128 token và cần một budget riêng; không dùng trực tiếp con số classification
head ở trên.

### 9.7 Tài nguyên SLR dự kiến sau khi hoàn thiện

Bảng dưới cộng upper-bound của các kernel còn thiếu vào trạng thái hiện tại.
Nó dùng standalone HLS estimate cho SLR0/SLR1, routed-plus-csynth proxy cho
SLR2 và routed baseline cho SLR3. Vì vậy đây là bảng budget, không phải
full-link utilization report.

| SLR | Cấu hình hoàn thiện được tính | LUT | FF/REG | BRAM | URAM | DSP | Tỷ lệ cao nhất đáng chú ý |
|---|---|---:|---:|---:|---:|---:|---|
| SLR0 | QKV + Embedding upper-bound | 198,060 | 308,348 | 32 | 216 | 976 | URAM 67.5% trước platform adjustment |
| SLR1 | Attention Core | 140,172 | 180,508 | 2 | 24 | 185 | LUT khoảng 32.4% |
| SLR2 | Routed FFN SLR2 + Out/Norm + residual fork | khoảng 205,000 | khoảng 293,500 | khoảng 263 + FIFO | 64 | 1,676 | DSP khoảng 54.6% |
| SLR3 | Routed FFN SLR3 + fused residual/LN | 123,725 | 231,822 | 103 | 32 | 1,299 | DSP khoảng 42.3% |
| SLR3 optional | Dòng trên + classification head | 163,725 | 291,822 | 135 | 48 | 1,427 | DSP khoảng 46.5%, LUT khoảng 37.9% |

Nhận xét:

- SLR0 vẫn bị giới hạn bởi URAM. Embedding table phải đặt ở DDR, không cache
  toàn bộ bằng URAM.
- SLR1 còn nhiều headroom và không cần tăng compute khi Core đã nhanh hơn QKV.
- SLR2 có DSP cao nhất do Out projection và FFN UP cùng tồn tại, nhưng proxy
  vẫn dưới 55%.
- SLR3 còn đủ chỗ cho residual/LN và một classification head nhỏ.
- FIFO/AXIS infrastructure chỉ được chốt sau full-link; không giả định bằng 0
  trong utilization cuối.

### 9.8 Latency budget sau khi hoàn thiện các khối bắt buộc

Budget hiện tại `179.45 ms` chưa tính FFN residual/LayerNorm cuối mỗi layer.
Thêm tail tối đa 70,000 cycles/layer:

```text
current model estimate               53,834,796 cycles
FFN residual/LN, 12 * 70,000            840,000 cycles
embedding upper bound                   100,000 cycles
classification head upper bound          50,000 cycles (nếu dùng)
---------------------------------------------------------
complete classification compute      54,824,796 cycles
compute latency at 300 MHz              182.75 ms
remaining budget                      5,175,204 cycles
remaining time                           17.25 ms
```

Nếu chỉ tính encoder + embedding, bỏ classification head:

```text
54,774,796 cycles = 182.58 ms
```

Host/runtime overhead target 1,000,000 cycles vẫn giữ tổng dự kiến khoảng
186.08 ms, còn xấp xỉ 13.92 ms trước giới hạn 200 ms.

### 9.9 Những việc không nên tách thành kernel mới

- Không tách Q, K, V thành ba kernel mới: fused QKV hiện đã đạt latency,
  resource và logic-synthesis timing.
- Không tạo standalone Attention LayerNorm: đã có trong Out/Norm.
- Không tạo standalone FFN LayerNorm: fuse vào DOWN để reuse output buffer.
- Không tạo stream-fork kernel riêng nếu Out/Norm duplicate được hai output.
- Không tạo DDR reorder kernel giữa Core và Out/Norm; stream order đã khớp.
- Không tạo wait-stage kernel. Đồng bộ bằng producer/consumer stream và token
  pointer hiện có; wait stage không tạo estimate compute có giá trị.

## 10. Quyết định tối ưu hiện tại

```text
QKV       khóa datapath hiện tại; không tăng parallelism do URAM/routing.
Core      giữ source; chỉ sửa nếu RTL synthesis thật sự âm tại 3.333 ns.
Out/Norm  giữ projection II=2; không đổi sang II=1 vì không phải bottleneck và
          cùng chia sẻ SLR2 với FFN UP.
FFN       giữ V15 làm routed baseline cho integration; V21 chỉ dùng làm report
          latency cho đến khi có bằng chứng route tốt hơn.
```

Không dùng custom clock uncertainty để làm đẹp report. Timing cuối cùng phải
được đánh giá bằng RTL synthesis và sau đó bằng full-link/post-route ở 300 MHz.

## 11. Report nguồn

```text
reports/csynth_qkv.rpt
reports/csynth_attn_core.rpt
reports/csynth_attn_out_norm.rpt
reports/bert_qkv_kernel_export.rpt
reports/export_syn.rpt
csynth_up_v21.rpt
csynth_down_v21.rpt
ATTENTION_200MS_LATENCY_BUDGET.md
ATTENTION_FFN_SLR_ARCHITECTURE_HANDOFF.md
```
