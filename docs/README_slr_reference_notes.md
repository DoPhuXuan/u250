# README — Bộ ghi chú paper cho tối ưu Split-Kernel BERT theo SLR

Bộ `.md` này tóm tắt các paper/tài liệu đã upload theo góc nhìn thực dụng: dùng được gì cho tối ưu `bert_qkv_kernel`, split-kernel pipeline, stream/FIFO, và floorplan theo SLR trên Alveo U250.

## Thứ tự nên đọc

```text
1. u250_ds962_resource_information.md
   -> Khóa platform: U250 = 4 SLR + 4 DDR4.

2. 2312_15159_spatial_llm_slr.md
   -> Blueprint chính: spatial split-kernel Transformer + SLR partition.

3. 2509_13694_streamtensor_llm_streaming.md
   -> Stream layout, kernel fusion, FIFO sizing, residual stream.

4. heteroflow_fpga2022_data_placement.md
   -> Data placement: DDR, FIFO, double buffer, bank scheduling.

5. 2501_09118_stream_hls_dataflow.md
   -> Graph-level pipelining, resource budget theo SLR, multi-kernel scheduling.

6. 2409_13975_protea_transformer_tiling.md
   -> Tile size, DSP/LUT pressure, runtime-programmable Transformer encoder.
```

## Paper nào chứng minh gì?

| File | Vai trò chính | Có SLR-specific không? | Dùng cho repo |
|---|---|---:|---|
| `2312_15159_spatial_llm_slr.md` | Spatial LLM/BERT dataflow + SLR partition | Có, nhưng U280 3 SLR | Blueprint kiến trúc |
| `u250_ds962_resource_information.md` | U250 platform constraints | Có, U250 4 SLR | Source of truth cho floorplan |
| `2509_13694_streamtensor_llm_streaming.md` | Stream layout, FIFO sizing, kernel fusion | Không | QKV layout, residual stream |
| `heteroflow_fpga2022_data_placement.md` | Data placement / FIFO / double buffer | Không | DDR/FIFO placement |
| `2501_09118_stream_hls_dataflow.md` | Multi-kernel graph-level optimization | Có DSP limit theo SLR trong eval | Per-kernel budget, pipeline balance |
| `2409_13975_protea_transformer_tiling.md` | Transformer tiling / DSP-LUT tradeoff | Không | Tile size và parallelism |

## Kết luận roadmap

```text
QKV W8A8 route-clean
  -> FFN W8A8 route-clean
  -> optional W4A8 evaluation
  -> residual DDR-to-stream optimization
  -> re-evaluate SLR mapping with per-kernel/per-SLR reports
```

## Metrics cần chứng minh tiến bộ

```text
- QKV HLS warning: no "Stride is incompatible"
- QKV estimated clock and post-route WNS
- QKV LUT/DSP/BRAM/URAM
- per-kernel utilization
- per-SLR utilization
- congestion severity
- number of SLR-crossing streams
- DDR channel/bank conflicts
- end-to-end latency
- tokens/s or samples/s
- power/GOPS/W if available
```
