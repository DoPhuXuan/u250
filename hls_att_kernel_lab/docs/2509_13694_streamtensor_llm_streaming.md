# 2509.13694v2 — StreamTensor: tinh túy stream layout / kernel fusion / FIFO sizing cho LLM

**Nguồn:** `2509.13694v2.pdf`  
**Tên paper:** *StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs*  
**Mức hữu ích cho tối ưu SLR U250:** Cao cho stream/FIFO/layout giữa kernel. Không phải U250/SLR-specific; target evaluation chính là U55C/U280/GPU comparison.

## Một câu tóm tắt

StreamTensor chỉ ra rằng với LLM dataflow accelerator, vấn đề khó không chỉ là kernel compute mà là **stream layout compatibility, kernel fusion, DMA/memory layout, và FIFO sizing tránh deadlock/stall**.

## Ý tưởng kiến trúc cốt lõi

Paper mô tả accelerator dataflow gồm:

```text
DMA -> Kernel -> FIFO -> Kernel -> Layout Converter -> Kernel -> DMA
```

Các thành phần chính:

```text
- Kernel: operator hoặc coarse-grained task.
- Token: đơn vị dữ liệu trên stream.
- FIFO: cân bằng tốc độ producer/consumer, tránh stall/deadlock.
- Stream Layout Converter: đổi layout on-the-fly khi producer/consumer không cùng pattern.
- DMA: đổi memory-mapped interface thành stream interface và ngược lại.
```

Điểm này rất sát với repo:

```text
QKV stream layout
context stream layout
attn_mid stream / DDR hand-off
gelu stream
residual path
```

## Ba vấn đề quan trọng cho SLR optimization

### 1. Inter-kernel correlation

Producer/consumer phải có tiling, loop order, vectorization tương thích. Nếu QKV xuất stream theo layout không hợp với attention core, FIFO sẽ không cứu được hoàn toàn.

Áp dụng:

```text
- QKV output layout phải match attn_core consumption order.
- FFN INT8 packed layout phải match ffn_stream consumption.
- Nếu layout không match, cần converter nhỏ hoặc đổi tile order.
```

### 2. External memory access

DMA/memory layout phải match stream pattern. Paper nhấn mạnh:

```text
- overlap memory access với compute,
- layout match streaming,
- pack/vectorize để maximize bandwidth.
```

Áp dụng cho QKV:

```text
Row-major [out][in_pack] gây stride khi load tile.
Tile-major [out_tile][in_pack][out_lane] tốt hơn cho burst.
```

### 3. FIFO sizing

Paper cảnh báo FIFO quá nhỏ có thể tạo back-pressure, stall cascade, thậm chí deadlock. Nhưng FIFO quá lớn lại tốn BRAM và routing.

Áp dụng:

```text
- Không tăng FIFO depth lên TOKEN_WORDS nếu chưa chứng minh cần.
- Dùng conservative FIFO trước.
- Chỉ tăng depth dựa trên producer/consumer rate hoặc trace.
```

## Kernel fusion / memory reduction

StreamTensor cho thấy stream-based kernel fusion có thể giảm đáng kể intermediate storage. Với LLM, họ fuse cả transformer block lên một FPGA trong GPT-2 experiment, dùng layout converter và FIFO để tất cả intermediate đi on-chip.

Số liệu đáng dùng:

```text
Kernel fusion giảm on-chip intermediate memory xuống còn khoảng 14.8%–16.8% so với original design.
```

Áp dụng cho repo:

```text
- Mục tiêu dài hạn: giảm DDR hand-off giữa attention và FFN.
- Nhưng phải làm sau khi QKV/FFN route-clean.
- Nếu cần layout converter, đặt nó gần consumer để giảm fanout/crossing.
```

## Số liệu đáng dùng để chứng minh

GPT-2 comparison:

```text
Against Allo:
- Total latency geomean ratio: 0.76x
- TTFT geomean ratio: 0.40x
- Decode speed ratio: 1.06x

Against DFX:
- Total latency geomean ratio: 0.52x
- TTFT geomean ratio: 0.19x
- Decode speed ratio: 1.17x
```

GPU comparison:

```text
Against A100:
- Total latency geomean ratio: 0.64x
- Decode speed ratio: 1.89x
- TTFT worse: 10.65x ratio due to GPU fast TTFT in their setup

Energy:
- Up to 1.99x energy efficiency over GPU on Qwen.
- Up to 1.59x energy efficiency over GPU on Gemma.
```

Platform table:

```text
Ours: AMD U55C, 250 MHz, W4A8, 16GB HBM, 460GB/s, 41MB on-chip memory.
Allo/DFX: AMD U280.
A100/2080Ti: GPU baselines.
```

Không dùng các số này để claim U250 performance trực tiếp.

## Liên hệ với repo BERT U250

### QKV layout

StreamTensor ủng hộ việc đổi QKV layout:

```text
Old:
[out][in_pack]

New:
[out_tile_global][in_pack][out_lane]
```

Lý do:

```text
- load pattern match stream/tile consumption,
- giảm stride warning,
- cải thiện burst,
- giảm logic address irregularity.
```

### Residual DDR hand-off

Paper ủng hộ stream intermediate on-chip, nhưng cũng cảnh báo FIFO sizing. Vì vậy roadmap đúng là:

```text
1. QKV route-clean.
2. FFN INT8 route-clean.
3. Sau đó mới thay attn_mid_ddr bằng residual stream.
```

Không nên làm residual FIFO full-token ngay từ đầu.

### Layout converter

Nếu `bert_attn_out_norm_kernel` và `bert_ffn_down_norm_kernel` consume residual khác order:

```text
- Không ép một FIFO rất sâu để che mismatch.
- Cân nhắc converter nhỏ hoặc đổi producer/consumer tile order.
```

## Checklist cho Codex

```text
[ ] Ghi rõ stream token format cho qkv_stream/context_stream/gelu_stream.
[ ] Với mỗi stream, ghi producer order và consumer order.
[ ] Nếu order mismatch, sửa layout hoặc thêm converter tối thiểu.
[ ] Không tăng FIFO depth nếu không có producer-consumer rate analysis.
[ ] Khi bỏ DDR hand-off, đo BRAM/routing trước/sau.
```

## Không nên áp dụng máy móc

```text
- Không fuse toàn transformer block vào một kernel lớn trên U250.
- Không dùng kết quả U55C để claim U250 timing.
- Không triển khai compiler-level itensor; chỉ mượn nguyên tắc stream layout.
```

## Kết luận dùng cho kế hoạch

StreamTensor là reference tốt nhất cho phần **stream layout và FIFO sizing**. Nó củng cố quyết định đổi QKV weight layout sang tile-major và trì hoãn residual DDR-to-stream cho đến khi các compute kernel đã route-clean.
