# 2409.13975v1 — ProTEA: tinh túy tiling / DSP / runtime-programmable Transformer Encoder

**Nguồn:** `2409.13975v1.pdf`  
**Tên paper:** *ProTEA: Programmable Transformer Encoder Acceleration on FPGA*  
**Mức hữu ích cho tối ưu SLR U250:** Trung bình. Paper không target U250/SLR; target chính là Alveo U55C. Hữu ích nhất cho **tiling, DSP utilization, và chọn mức parallelism vừa tài nguyên**.

## Một câu tóm tắt

ProTEA cho thấy với Transformer encoder dense, performance phụ thuộc rất nhiều vào **tile size cố định lúc synthesis**, số head chạy song song, và cách tránh over-utilization LUT/DSP.

## Ý tưởng kiến trúc cốt lõi

ProTEA là accelerator runtime-programmable cho Transformer encoder dense. Hardware được synthesize một lần với tile size cố định, còn các thông số sau có thể đổi runtime:

```text
- number of attention heads
- number of layers
- embedding dimension
- sequence length
```

Điểm đáng học cho repo BERT:

```text
- Không cần resynthesis cho mọi biến thể model nếu datapath đủ tổng quát.
- Nhưng tile size/unroll/parallelism vẫn là design-time decision.
- Cần chọn tile size sao cho vừa latency vừa frequency vừa compile time.
```

## Tiling trong MHA và FFN

Paper chọn tile size bằng sweep frequency/latency. Kết quả quan trọng:

```text
TS_MHA = 64
TS_FFN = 128
Frequency tối đa: 200 MHz
Compile time khoảng 36 giờ cho encoder SOTA
```

Họ dùng tiling để load weight vào BRAM, input vào input BRAM, rồi compute theo từng tile.

Áp dụng cho repo:

```text
QKV/FFN packed layout nên là tile-major.
Tile size không chỉ để đúng toán học; nó phải giúp:
- burst-friendly load,
- array partition hợp lý,
- BRAM banking rõ,
- routing ít fanout.
```

## Parallel attention heads

ProTEA tìm được điểm tốt với:

```text
8 parallel attention heads
12 layers
embedding dimension 768
sequence length 64
```

Với cấu hình đó, họ đạt latency thấp nhất 279 ms và GOPS cao nhất 53 trong nhóm test runtime-programmable.

Áp dụng cho repo:

```text
- HEAD_PAR không nên tăng theo cảm tính.
- QKV/attention parallelism phải được bound bởi LUT/route, không chỉ DSP.
- Nếu QKV đang congestion, tăng head parallelism sẽ làm tệ hơn.
```

## Resource pressure

Paper report một điểm rất đáng chú ý:

```text
DSP usage: 3612 DSP = 40%
LUT usage: 993107 LUT = 76%
FF usage: 704115 FF = 27%
```

Họ nói DSP utilization thêm bị giới hạn bởi LUT sẵn có. Đây là bài học rất trực tiếp cho QKV của repo:

```text
DSP còn dư không có nghĩa là có thể tăng MAC array.
LUT/routing/fanout mới là giới hạn thực tế khi thiết kế lớn.
```

## Số liệu đáng dùng để chứng minh

Từ evaluation:

```text
Platform: Alveo U55C
Frequency: 200 MHz
Data format: 8-bit fixed point
Best test:
- Sequence length: 64
- Embedding: 768
- Heads: 8
- Layers: 12
- Latency: 279 ms
- GOPS: 53
Resource:
- DSP: 3612 / 40%
- LUT: 993107 / 76%
- FF: 704115 / 27%
```

Cross-platform comparison:

```text
- 2.5x faster than NVIDIA Titan XP GPU for one evaluated model.
- 16x faster than NVIDIA Titan XP GPU for another evaluated model.
- 1.3x–2.8x speedup versus some custom FPGA accelerators.
```

Nhưng cần ghi rõ: paper có case chậm hơn CPU/GPU khi baseline dùng pruning/aggressive sparsity. Không nên dùng ProTEA làm bằng chứng tuyệt đối rằng FPGA luôn nhanh hơn GPU.

## Liên hệ với tối ưu SLR

Paper không có SLR floorplan. Tuy vậy nó hỗ trợ ba quyết định SLR-level:

### 1. Chọn tile size để giảm route pressure

```text
QKV/FFN tile phải vừa on-chip buffer và vừa routing.
Tile size quá lớn có thể làm frequency giảm hoặc compile time tăng mạnh.
```

### 2. Tách QKV/FFN để đo tài nguyên riêng

ProTEA report MHA/FFN là các block lớn và phải tune tile riêng. Với split-kernel BERT:

```text
bert_qkv_kernel:
  tune MAC_LANES / OUT_PAR / tile layout

bert_ffn_up_gelu_kernel:
  tune FFN tile and activation buffering

bert_ffn_down_norm_kernel:
  tune down projection and residual/LN buffer
```

### 3. Không tăng DSP nếu LUT/congestion đã căng

Nếu report của repo cho thấy LUT cao hoặc WNS xấu, hướng đúng là:

```text
- giảm parallelism,
- sửa layout/burst,
- dùng DSP binding,
- chia stage rõ hơn,
```

không phải tăng unroll.

## Cách áp dụng cho Codex

```text
[ ] Khi sửa QKV, kiểm tra tile size có tạo contiguous load không.
[ ] Với FFN INT8, dùng tile-major layout từ đầu.
[ ] Không tăng HEAD_PAR nếu QKV route chưa sạch.
[ ] Ghi lại resource theo DSP/LUT/FF/BRAM, không chỉ latency.
[ ] Nếu LUT > 70% một SLR hoặc congestion cao, giảm parallelism trước.
```

## Không nên áp dụng máy móc

```text
- Không copy latency tuyệt đối vì ProTEA target U55C và full encoder runtime khác repo.
- Không dùng 8 parallel heads làm luật cứng cho U250.
- Không dùng ProTEA để justify SLR floorplan vì paper không làm SLR mapping.
```

## Kết luận dùng cho kế hoạch

ProTEA hữu ích như paper phụ để chứng minh rằng **tile size + parallelism + LUT pressure** là quyết định kiến trúc quan trọng. Với repo U250, nó ủng hộ hướng “QKV route-friendly trước, giữ parallelism vừa phải, rồi mới mở rộng FFN”.
