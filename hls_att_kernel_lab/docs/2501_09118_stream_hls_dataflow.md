# 2501.09118v1 — Stream-HLS: tinh túy multi-kernel dataflow / graph-level pipelining

**Nguồn:** `2501.09118v1.pdf`  
**Tên paper:** *Stream-HLS: Towards Automatic Dataflow Acceleration*  
**Mức hữu ích cho tối ưu SLR U250:** Cao cho phương pháp **multi-kernel scheduling/resource budgeting**. Không phải paper Transformer U250, nhưng có kết quả theo DSP limit của một SLR.

## Một câu tóm tắt

Stream-HLS chứng minh rằng tối ưu HLS từng loop/pragmas riêng lẻ là chưa đủ; với multi-kernel design phải tối ưu **graph-level pipelining, FIFO/shared-buffer choice, và resource allocation toàn graph**.

## Ý tưởng kiến trúc cốt lõi

Stream-HLS tự động biến chương trình multi-kernel tuần tự thành dataflow architecture với streaming. Các tối ưu chính:

```text
- Convert shared buffers to FIFOs when possible.
- Node-level pipelining.
- Graph-level pipelining.
- Node-level parallelization.
- Global scheduling/resource allocation bằng MINLP.
```

Điểm này khớp trực tiếp với split-kernel BERT:

```text
embedding_prep -> qkv -> attn_core -> attn_out_norm -> ffn_up_gelu -> ffn_down_norm
```

Nếu chỉ tối ưu từng kernel, pipeline end-to-end vẫn có thể nghẽn do stage imbalance hoặc FIFO/buffer không đúng.

## Bài học về SLR/resource budget

Paper dùng ba DSP limit trong evaluation:

```text
220 DSP
2560 DSP = one SLR limit trong paper
9024 DSP = full U280 board limit
```

Đây là tư duy rất hữu ích cho U250:

```text
Không chỉ hỏi "board còn đủ DSP không?"
Phải hỏi "kernel/stage này fit trong budget một SLR không?"
```

Áp dụng cho repo:

```text
- Tạo resource budget per-kernel.
- Ưu tiên QKV fit trong SLR0.
- Nếu QKV vượt timing/resource budget, giảm MAC_LANES trước khi đổi pipeline lớn.
```

## Số liệu đáng dùng để chứng minh

Stream-HLS report:

```text
- Geomean speedup up to 79.43x vs prior automatic frameworks.
- Geomean speedup up to 10.62x vs manually optimized abstraction-framework designs.
- Against baselines under different DSP limits, speedups range from 5.42x to 2270.25x.
```

Ablation với 2560 DSP limit:

```text
Opt1 baseline: 1.00x
Opt2 graph/node-level pipelining: 4.70x geomean
Opt3 node-level parallelization: 28.31x geomean
Opt4 separate combined stages: 152.30x geomean
Opt5 combined optimization: 314.89x geomean
```

Transformer-related kernels trong table:

```text
Feed Forward:
- Opt1: 6.73E+07 cycles
- Opt5: 6.60E+04 cycles
- Speedup: 1019.39x

Multi-Head Self Attention:
- Opt1: 1.05E+07 cycles
- Opt5: 3.51E+04 cycles
- Speedup: 299.67x
```

Lưu ý: đây là RTL cycle simulation cho benchmark kernels, không phải full BERT U250 post-route.

## Liên hệ với repo BERT

### 1. Chứng minh vì sao không chỉ sửa pragma trong QKV

Nếu QKV route-clean nhưng stage kế tiếp không consume đều, FIFO/stream vẫn gây stall. Nên đo:

```text
- latency từng kernel,
- II từng stream producer/consumer,
- FIFO occupancy nếu có sim,
- stream crossing giữa SLR.
```

### 2. Chứng minh vì sao cần graph-level view

Với split-kernel BERT:

```text
bert_qkv_kernel có thể là producer nhanh/chậm khác attn_core.
bert_attn_out_norm_kernel có residual/LN pattern khác FFN down.
FFN up/down có compute intensity lớn hơn attention nonlinear.
```

Do đó tối ưu theo graph nên gồm:

```text
- cân bằng throughput stage,
- tránh FIFO depth quá lớn,
- tránh stage quá to không fit SLR,
- không đổi SLR mapping trước khi có resource report toàn graph.
```

### 3. Resource budget theo một SLR

Dùng Stream-HLS để justify quy tắc:

```text
Mỗi major kernel phải có target budget per SLR.
Nếu QKV dùng quá nhiều LUT/routing, giảm parallelism dù board tổng còn dư.
```

## Cách áp dụng cho Codex

```text
[ ] Tạo bảng per-kernel: latency, LUT, FF, DSP, BRAM, URAM.
[ ] Tạo bảng per-edge stream: producer, consumer, depth, SLR crossing.
[ ] Không chỉ check csynth QKV; cần full link/post-route.
[ ] Nếu một kernel không fit timing trong SLR, giảm unroll/parallelism.
[ ] Nếu pipeline imbalance lớn, chỉnh FIFO/buffer sau khi QKV sạch.
```

## Checklist đo tiến bộ

```text
Before/After:
- QKV estimated clock
- QKV post-route WNS
- QKV LUT/DSP/BRAM/URAM
- number of SLR-crossing streams
- worst congestion region
- full pipeline latency
- per-kernel latency balance
```

## Không nên áp dụng máy móc

```text
- Stream-HLS không giải quyết hoàn chỉnh DDR/HBM bandwidth trong paper này.
- Evaluation target U280, không phải U250.
- Không có physical SLR floorplan cho BERT.
- Không thay thế Vitis kernel placement thủ công bằng MINLP ngay trong repo.
```

## Kết luận dùng cho kế hoạch

Stream-HLS là reference tốt để chứng minh rằng split-kernel không chỉ là chia file code. Cần tối ưu ở mức **dataflow graph**, với resource budget theo SLR và graph-level pipelining. Với repo hiện tại, bài này ủng hộ việc giữ split-kernel baseline, đo per-kernel resource, và chỉ đổi SLR mapping sau khi QKV route-clean.
