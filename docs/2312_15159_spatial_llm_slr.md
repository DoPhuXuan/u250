# 2312.15159v2 — Spatial LLM Acceleration: tinh túy cho split-kernel / SLR

**Nguồn:** `2312.15159v2.pdf`  
**Tên paper:** *Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference*  
**Mức hữu ích cho tối ưu SLR U250:** Rất cao, nhưng paper target U280 3 SLR, không phải U250 4 SLR.

## Một câu tóm tắt

Paper này là blueprint tốt nhất cho hướng **Transformer/BERT spatial dataflow**: tách operator thành các kernel/PE chuyên biệt, nối bằng FIFO, giảm off-chip intermediate traffic, rồi partition theo SLR để dễ timing closure.

## Ý tưởng kiến trúc cốt lõi

Paper phân biệt hai hướng:

- **Temporal/overlay architecture:** reuse một PE cho nhiều operator/layer, linh hoạt nhưng phải ghi/đọc intermediate qua memory nhiều.
- **Spatial architecture:** mỗi operator/layer có hardware riêng, nối nhau bằng FIFO/multibuffer, xử lý pipeline qua nhiều stage.

Tinh túy cho repo hiện tại:

```text
Embedding / input prep
  -> QKV projection
  -> attention core / SDP / softmax
  -> attention output projection + residual + LN
  -> FFN up + GELU
  -> FFN down + residual + LN
```

Không nên quay lại monolithic GEMM engine hoặc overlay-style scheduler.

## Điểm quan trọng nhất về SLR

Paper nói rõ với multi-die FPGA, cần **explicit dataflow partitioning** để đạt timing. Trên U280, họ chia dataflow thành 3 region để mỗi region nằm gọn trong một SLR. Vitis sẽ chèn AXI Register Slice cho crossing giữa SLR.

Mapping tham khảo trong paper:

```text
SLR0: Q, K, V projection
SLR1: MHA / SDP / attention output projection
SLR2: FFN1 + FFN2
```

Tư duy cần giữ khi port sang U250:

```text
Một stage lớn phải fit trong một SLR càng nhiều càng tốt.
Cross-SLR stream phải ít, rõ ràng, và được pipeline.
Không tăng song song hóa nếu khiến một kernel không còn fit SLR.
```

## Bài học timing closure

Paper thử nhiều size systolic array và rút ra bài học rất sát với vấn đề route của `bert_qkv_kernel`:

```text
- Một SLR ở 250 MHz fit tối đa khoảng ba systolic array 8x16.
- Systolic array 16x16 đơn lẻ fail timing.
- LUT multiplier linh hoạt placement hơn nhưng dây inter-LUT làm frequency thấp hơn.
- DSP-based multiplier đạt frequency tốt hơn.
```

Áp dụng cho repo:

```text
- Giữ QKV INT8 MAC array vừa phải.
- 8x8 là profile hợp lý cho pass đầu.
- Nếu route fail, giảm MAC_LANES trước khi đổi toàn bộ SLR mapping.
- Không tăng QKV_OUT_PAR hoặc gộp thêm logic vào QKV nếu chưa có post-route sạch.
```

## Quantization / compute packing

Paper dùng:

```text
BERT: W4A8
GPT: W8A8
```

Với BERT, họ nhấn mạnh linear operators là bottleneck chính. Non-linear operators như Softmax, LayerNorm, GELU không phải target quantization đầu tiên vì có thể ảnh hưởng accuracy và không phải bottleneck lớn.

Áp dụng cho repo:

```text
Pass hiện tại:
- QKV W8A8 trước vì code đã có scaffold INT8.
- FFN W8A8 sau QKV.
- W4A8 chỉ đánh giá sau khi W8A8 route-clean.
- Không quantize GELU/Softmax/LayerNorm trong pass đầu.
```

## FIFO / dataflow

Paper nối các operator bằng FIFO, nhưng KV cache dùng double buffer. Intermediate activation được stream trực tiếp sang operator kế tiếp. Parameters nằm off-chip và loader che latency bằng overlap với compute.

Áp dụng cho repo:

```text
- Giữ stream/FIFO giữa split-kernel stages.
- Không tăng FIFO depth lớn trừ khi có bằng chứng deadlock/stall.
- DDR residual hand-off có thể tối ưu sau, nhưng không thay bằng FIFO TOKEN_WORDS quá sớm.
```

## Số liệu đáng dùng để chứng minh hướng đi

Từ paper:

```text
BERT trên U280:
- Frequency: 245 MHz
- Quantization: W4A8
- Latency: 26.01 ms với sequence length 512
- Throughput: 38.45 samples/s
- Speedup: 3.66x so với FQ-BERT, 13.36x so với TRAC
- Resource tổng: 389 BRAM, 1780 DSP, 653K FF, 569K LUT, 111 URAM
```

Per-SLR trong BERT accelerator:

```text
SLR0 latency: 4.86 ms
SLR1 latency: 14.63 ms
SLR2 latency: 19.81 ms

SLR0 resource: 130 BRAM, 482 DSP, 200K FF, 167K LUT, 3 URAM
SLR1 resource: 136 BRAM, 590 DSP, 240K FF, 212K LUT, 50 URAM
SLR2 resource: 123 BRAM, 708 DSP, 213K FF, 191K LUT, 58 URAM
```

Systolic kernel ablation:

```text
16x16 INT8 GEMM without DSP packing:
- Latency: 15.73 ms
- DSP: 256
- LUT: 168K

With DSP packing:
- Latency: 15.73 ms
- DSP: 128
- LUT: 244K

AutoSA:
- Latency: 15.71 ms
- DSP: 256
- BRAM: 514
- LUT: 244K
```

## Cách áp dụng trực tiếp cho repo BERT U250

### Quy tắc partition

U250 có 4 SLR, nên không copy mapping 3 SLR của U280. Hãy dùng nguyên tắc:

```text
SLR budget trước, latency sau.
Mỗi kernel lớn phải có resource report riêng.
Không để một kernel vừa cao LUT vừa nhiều SLR crossing.
```

Mapping hiện tại có thể giữ cho pass đầu:

```text
SLR1: bert_embedding_prep_kernel
SLR0: bert_qkv_kernel
SLR1: bert_attn_core_kernel
SLR2: bert_attn_out_norm_kernel
SLR2: bert_ffn_up_gelu_kernel
SLR3: bert_ffn_down_norm_kernel
```

Sau khi có report mới, đánh giá lại theo:

```text
- QKV có fit sạch trong SLR0 không?
- SLR1 có bị crossing nhiều giữa embedding/QKV/attn_core không?
- SLR2 có quá nặng vì vừa attn_out_norm vừa ffn_up_gelu không?
- SLR3 có đủ nhẹ để nhận thêm residual consumer không?
```

### Quy tắc cho `bert_qkv_kernel`

```text
1. Sửa weight layout để load burst-friendly.
2. Dùng DSP cho INT8 multiply trong route-friendly profile.
3. Giữ 8x8 trước.
4. Fallback 4x8 nếu route fail hoặc WNS xấu.
5. Không gộp embedding vào QKV.
```

## Không nên áp dụng máy móc

Không copy trực tiếp:

```text
- U280 3-SLR floorplan.
- W4A8 ngay lập tức nếu W8A8 QKV chưa ổn.
- Full residual streaming nếu chưa biết BRAM/routing pressure.
- 16x16 systolic array cho một kernel route-critical.
```

## Checklist cho Codex

```text
[ ] QKV INT8 layout chuyển sang tile-major.
[ ] HLS không còn "Stride is incompatible".
[ ] QKV vẫn là một kernel riêng, không duplicate path.
[ ] DSP binding bật qua BERT_QKV_INT8_MUL_USE_DSP.
[ ] MAC_LANES=8, OUT_PAR=8 profile trước.
[ ] Post-synth/post-route kiểm tra LUT, DSP, WNS, congestion.
[ ] Chưa đổi SLR mapping trước khi có report mới.
[ ] Chưa chuyển residual DDR sang FIFO lớn.
```

## Kết luận dùng cho kế hoạch

Paper này chứng minh đúng hướng **split-kernel spatial dataflow + SLR-aware partitioning**. Với U250, chỉ nên port **nguyên lý**, không port **floorplan**. Mục tiêu đúng là làm từng kernel fit/routable theo SLR trước, rồi mới tăng quantization/packing và giảm DDR hand-off.
