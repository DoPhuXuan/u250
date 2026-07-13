# HeteroFlow FPGA 2022 — tinh túy data placement / FIFO / double buffer

**Nguồn:** `heteroflow-fpga2022.pdf`  
**Tên paper:** *HeteroFlow: An Accelerator Programming Model with Decoupled Data Placement for Software-Defined FPGAs*  
**Mức hữu ích cho tối ưu SLR U250:** Cao cho data placement và memory hierarchy. Không phải Transformer/SLR-specific.

## Một câu tóm tắt

HeteroFlow chứng minh rằng FPGA performance phụ thuộc rất mạnh vào việc đặt data đúng nơi, đúng thời điểm: host-device DMA, inter-kernel FIFO/double buffer, intra-kernel systolic data movement.

## Ý tưởng kiến trúc cốt lõi

HeteroFlow tách:

```text
Algorithm: tính gì?
Data placement: dữ liệu nằm/chạy qua đâu?
Compute schedule: pipeline/unroll/tile như thế nào?
```

Primitive trung tâm là `.to()`, dùng cho ba level:

```text
1. Host-accelerator placement
2. Inter-kernel placement
3. Intra-kernel placement
```

Đây là cách nghĩ rất hữu ích cho repo, dù không dùng DSL HeteroFlow.

## Các mode data placement quan trọng

### Host-accelerator

```text
Host DRAM / device DRAM / cache-coherent interface / DMA
```

Áp dụng:

```text
- QKV/FFN weights nên được pack để DMA burst tốt.
- Không để nhiều kernel tranh cùng DDR bank nếu có thể tách.
```

### Inter-kernel

```text
FIFO stream nếu producer/consumer access sequential cùng order.
Single/double buffer nếu access arbitrary hoặc cần reuse.
```

Áp dụng:

```text
- qkv_stream/context_stream phù hợp FIFO.
- residual path có thể cần double buffer hoặc DDR nếu consumer order không stream-friendly.
```

### Intra-kernel

```text
FIFOs hoặc shift registers giữa PE trong systolic/spatial array.
```

Áp dụng:

```text
- QKV MAC lanes nên có local data movement đều.
- Tránh broadcast/fanout rộng qua nhiều PE nếu gây congestion.
```

## Số liệu đáng dùng để chứng minh

Paper nhấn mạnh một nghiên cứu trước cho thấy quản lý data layout và communication schemes có thể cải thiện performance 3–8x.

Evaluation chính của HeteroFlow:

### Optical Flow

```text
Rosetta U280:
- Fmax: 300 MHz
- Runtime: 3.49 ms
- LoC: 742

HeteroFlow U280:
- Fmax: 300 MHz
- Runtime: 3.43 ms
- LoC: 206
```

=> Gần bằng manual optimized HLS với ít code hơn.

### GEMM systolic array

```text
HF-HLSC FP32 output-stationary:
- LUT/FF: 25.4K / 32.9K
- BRAM/DSP: 23 / 48
- GOPS: 2.03

HF-HLSC Fixed<16,4> output-stationary:
- LUT/FF: 10.2K / 15.2K
- BRAM/DSP: 15 / 16
- GOPS: 4.26
```

### KNN I/O optimization

```text
14 PEs:
- Baseline runtime: 49.37 s
- +mem-coalescing: 24.29 s, 2.03x
- +AXI-controller: 10.14 s, 4.82x

28 PEs:
- Baseline not synthesizable
- +io-scheduling: 9.31 s, 5.30x
```

### UltraNet systolic array

```text
Baseline:
- Fmax: 231 MHz
- Runtime: 2.97 ms

+Systolic Array:
- Fmax: 233.8 MHz
- Runtime: 2.27 ms
- Third layer speedup: 3.99x
```

## Liên hệ với repo BERT U250

### 1. QKV weight packing

HeteroFlow củng cố quyết định:

```text
Memory layout phải theo access pattern của compute.
```

Với QKV:

```text
Row-major [out][in_pack]
  -> tạo stride khi load output tile

Tile-major [out_tile][in_pack][out_lane]
  -> contiguous load theo tile
```

### 2. DDR bank / memory port pressure

Nếu nhiều kernel cùng đọc/ghi DDR:

```text
- weights,
- input activations,
- attn_mid_ddr,
- outputs,
```

cần kiểm tra bundle/bank assignment trong `system.cfg`.

HeteroFlow cho thấy memory banking và I/O scheduling có thể quyết định synthesizable/performance.

### 3. Residual hand-off

HeteroFlow phân biệt FIFO và double buffer rất đúng với residual:

```text
Use FIFO:
- nếu producer/consumer cùng order,
- consumer có thể chạy streaming.

Use double buffer/DDR:
- nếu access arbitrary,
- nếu consumer cần replay/reuse,
- nếu stream depth quá lớn làm routing xấu.
```

Vì vậy không nên bỏ `attn_mid_ddr` quá sớm.

## Checklist cho Codex

```text
[ ] Với mỗi tensor lớn, ghi rõ placement: DDR, BRAM, FIFO, double buffer.
[ ] Với mỗi inter-kernel edge, ghi access order: sequential hay arbitrary.
[ ] Nếu sequential: FIFO depth conservative.
[ ] Nếu arbitrary/reuse: double buffer hoặc DDR.
[ ] Với weights: pack theo access pattern của tile loader.
[ ] Với DDR: kiểm tra bank/bundle conflict.
```

## Không nên áp dụng máy móc

```text
- Không cần đưa HeteroFlow DSL vào repo.
- Không dùng Optical Flow/GEMM numbers để claim Transformer speedup.
- Không ép mọi edge thành FIFO; paper cũng cho double buffer khi cần.
```

## Kết luận dùng cho kế hoạch

HeteroFlow là reference tốt để biện minh rằng **data placement là một phần của kiến trúc**, không phải chi tiết code. Với repo U250, nó ủng hộ QKV tile-major packing, DDR/bank-aware scheduling, và quyết định giữ residual DDR hand-off cho đến khi chứng minh được stream-friendly.
