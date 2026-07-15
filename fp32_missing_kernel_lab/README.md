# BERT FP32 missing-kernel lab for Alveo U250

Lab này chỉ hoàn thiện datapath FP32 của **một encoder layer**. Không có
embedding, pooler/classifier hoặc controller 12-layer.

## SLR mapping

| SLR | Kernel | Trạng thái trong lab |
|---|---|---|
| SLR0 | `bert_qkv_kernel` | Copy nguyên source đã khóa |
| SLR1 | `bert_attn_core_kernel` | Copy nguyên source đã khóa |
| SLR2 | `bert_attn_out_norm_residual_kernel` | Đã thêm `ffn_residual_stream` |
| SLR2 | `bert_ffn_up_gelu_v21_dotpipe_kernel` | Copy FP32 V21 với GELU PQ8 |
| SLR3 | `bert_ffn_down_residual_norm_fp32_kernel` | Kernel mới: DOWN + residual + LayerNorm |

`bert_attn_out_norm_residual_kernel` phát cùng một Attention-normalized word
vào hai stream 512-bit:

```text
attn_mid_stream       -> FFN UP, 6,144 words, row-major
ffn_residual_stream   -> FFN DOWN, 6,144 words, row-major
```

DOWN drain residual trước khi chờ `gelu_stream`, ghi `bias + residual` trực
tiếp vào `output_buf`, rồi mới chạy projection và LayerNorm. Thứ tự này tránh
deadlock khi FIFO residual đầy trong lúc UP chưa đọc đủ Attention input.

## Cấu trúc

```text
include/             Header FP32 đã khóa
kernels/             Năm source kernel của một encoder layer
scripts/             Tcl csynth và bộ phân tích report/budget
config/              Connectivity một layer, không có embedding/controller
reports/baseline/    Report nguồn trước khi thêm hai phần còn thiếu
reports/current/     Report do Tcl mới tạo
build/hls/           Project HLS sinh tự động
```

Target mặc định:

```text
part   xcu250-figd2104-2L-e
clock  3.333 ns, xấp xỉ 300 MHz
tool   Vitis HLS 2022.1
```

## Csynth từng kernel đã sửa

Chạy từ thư mục lab trên máy Linux đã setup Vitis 2022.1:

```bash
cd fp32_missing_kernel_lab

FP32_KERNELS=bert_attn_out_norm_residual_kernel \
vitis_hls -f scripts/run_vitis_fp32_csynth.tcl

FP32_KERNELS=bert_ffn_down_residual_norm_fp32_kernel \
vitis_hls -f scripts/run_vitis_fp32_csynth.tcl
```

Không truyền top thì Tcl mặc định synth đúng hai kernel đã sửa:

```bash
vitis_hls -f scripts/run_vitis_fp32_csynth.tcl
```

Rebuild cả năm kernel để kiểm tra lại toàn bộ source đã copy:

```bash
FP32_KERNELS=all vitis_hls -f scripts/run_vitis_fp32_csynth.tcl
```

Đổi part hoặc clock bằng biến môi trường:

```bash
HLS_PART=xcu250-figd2104-2L-e \
HLS_CLOCK_NS=3.333 \
FP32_KERNELS=all \
vitis_hls -f scripts/run_vitis_fp32_csynth.tcl
```

Nếu muốn chạy thêm Vivado logic synthesis sau csynth:

```bash
RUN_RTL_SYN=1 \
FP32_KERNELS=bert_ffn_down_residual_norm_fp32_kernel \
vitis_hls -f scripts/run_vitis_fp32_csynth.tcl
```

Report chính được copy vào:

```text
reports/current/csynth_attn_out_norm_residual.rpt
reports/current/csynth_ffn_down_residual_norm_fp32.rpt
```

## Đánh giá tự động tài nguyên và budget 200 ms

Sau khi hai report mới tồn tại:

```bash
python3 scripts/analyze_csynth_budget.py
```

Script đọc latency/resource trực tiếp từ report, tính pipeline sáu Attention
group và dùng công thức:

```text
attention = Q + C + O + 5*max(Q,C,O) + norm_tail
ffn       = max(UP, fused_DOWN)
layer     = attention + ffn
model_ms  = 12*layer / 300000
```

Gate cho hai top mới:

| Metric | Out/Norm + fork | Fused DOWN + residual/LN |
|---|---:|---:|
| Latency | `<= 1,370,000` cycles | `<= 1,725,260` cycles |
| DSP | `<= 410` | `<= 1,040` |
| FF | `<= 115,000` | `<= 235,000` |
| LUT | `<= 105,000` | `<= 121,000` |
| BRAM18 | theo baseline | `<= 48` |
| URAM | `<= 32` | `<= 32` |

Pipeline/timing guards:

```text
HLS top slack                         >= -0.05 ns
Out ACCUMULATE_ROW_BLOCK              II <= 2
Out NORM_WRITE_PACK                   II  = 1
DOWN hot MAC                          II  = 1
DOWN residual drain                   II  = 1
DOWN LayerNorm reduction              II <= 4
DOWN LayerNorm write                  II  = 1
```

Gate hệ thống:

```text
one layer             <= 5,000,000 cycles
12-layer estimate     < 200 ms tại 300 MHz
SLR LUT/BRAM           <= 65%
SLR FF/URAM/DSP        <= 70%
```

Ước lượng per-SLR của script là pre-route và có platform proxy bảo thủ. Chỉ
được kết luận routing-safe sau full-link/post-route với `WNS >= 0`, `TNS = 0`
và không có setup endpoint fail.

## Connectivity để validation link

[`config/system_one_layer.cfg`](config/system_one_layer.cfg) nối đủ năm kernel
qua bốn SLR, gồm cả `gelu_stream` và `ffn_residual_stream`. File này chỉ dành
cho validation một layer sau khi hai csynth gate đã pass; nó không thêm
embedding hoặc controller.

## Lưu ý accuracy

FFN UP trong lab giữ nguyên GELU PQ8 tám khoảng từ source FP32 hiện tại. Không
thay bằng PWL ba đoạn cũ vì phiên bản đó làm cosine cuối 12 layer giảm xuống
khoảng `0.24578`; PQ8 hiện đạt khoảng `0.999246`.

## Full 12-layer C-simulation accuracy

The Hugging Face checkpoint exporter, 12-layer C++ testbench, Tcl runner, and
layer-by-layer accuracy reporter are documented in
[`validation/README.md`](validation/README.md). Embedding remains outside the
hardware kernels; the real Hugging Face embedding output is used as the common
layer-0 input.

## Persistent full 12-layer encoder

The performance architecture starts five CUs once. Each CU executes the same
fixed 12-layer loop; AXI streams preserve layer order and carry both residual
and next-layer state entirely on the device. Only layer 11 writes hidden state
to DDR.

New top functions:

```text
bert_qkv_12layer_kernel
bert_attn_core_12layer_kernel
bert_attn_out_norm_12layer_kernel
bert_ffn_up_gelu_12layer_kernel
bert_ffn_down_norm_feedback_12layer_kernel
```

The two state streams are:

```text
qkv12.attention_residual_stream -> out12.attention_residual_stream
down12.next_hidden_stream       -> qkv12.next_hidden_stream
```

Run bounded-FIFO persistent C-simulation:

```bash
vitis_hls -f scripts/run_bert_12layer_persistent_csim.tcl
```

Synthesize all persistent tops and gate the conservative latency/resource
projection:

```bash
FP32_KERNELS=full vitis_hls -f scripts/run_vitis_fp32_csynth.tcl
python3 scripts/analyze_persistent_budget.py
```

Compile/link hardware after selecting the installed U250 `.xpfm`:

```bash
bash scripts/build_12layer_hw.sh /path/to/xilinx_u250_platform.xpfm
bash scripts/build_persistent_host.sh
```

Run one device-resident inference (five CU starts, zero host layer loops):

```bash
mkdir -p validation/outputs/hw_persistent
build/full_12layer_hw/run_persistent_12layer \
  build/full_12layer_hw/bert_base_12layer_u250.xclbin \
  validation/data/bert_base_uncased_seq128 \
  validation/outputs/hw_persistent/final_encoder_output.f32
```

The host excludes xclbin programming and one-time weight upload from
`device_full_12layer_ms`; it includes all five starts, all 12 encoder layers,
and the final device completion wait. Output D2H time is reported separately.
