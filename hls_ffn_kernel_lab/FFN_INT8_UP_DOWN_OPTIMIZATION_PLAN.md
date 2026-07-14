# Kế hoạch tối ưu FFN INT8 UP/DOWN trên Alveo U250

## 1. Mục tiêu

Tạo một nhánh FFN INT8 mới từ baseline FP32 V21, tối ưu riêng hai kernel:

```text
SLR2: Attention output projection + residual/norm + FFN UP/GELU
SLR3: FFN DOWN + residual/norm/output
```

Mục tiêu chính:

- Tăng throughput FFN ít nhất 3 lần, mục tiêu tốt là 4 lần hoặc hơn so với V21 FP32.
- Giảm mạnh DSP dùng cho FFN để còn chỗ cho kernel Attention trên SLR2.
- Giữ hot loop ở `II=1` và đạt timing 300 MHz sau route.
- Giảm băng thông weight và stream trung gian khoảng 4 lần bằng INT8 packing.
- Giữ sai số đủ nhỏ để độ chính xác toàn model giảm không quá 0.5 điểm phần trăm; giới hạn loại bỏ là giảm trên 1.0 điểm phần trăm.
- Không thay đổi mapping SLR2/SLR3 nếu report chưa chứng minh cần thay đổi.

Đây là một nhánh phát triển mới. Không sửa hoặc ghi đè baseline FP32 V21 đã được chấp nhận.

## 2. Baseline FP32 phải giữ nguyên

Source of truth:

```text
src/bert_ffn_kernel_v21.cpp
src/bert_ffn_kernel_lab.h
config/ffn_kernel_u250_v21.cfg
reports/baseline/csynth_up_v21_p16_fulldsp.rpt
reports/baseline/csynth_down_v21_p16_fulldsp.rpt
```

Cấu hình model hiện tại:

```text
SEQ_LEN          = 128
HIDDEN_SIZE      = 768
INTERMEDIATE_SIZE = 3072
Clock target     = 300 MHz (3.333 ns)
```

Latest csynth dùng để so sánh:

| Kernel | Latency cycles | Latency @ 300 MHz | DSP | BRAM18K | URAM | LUT | FF |
|---|---:|---:|---:|---:|---:|---:|---:|
| V21 UP/GELU | 1,536,392 | 5.121 ms | 1,120 | 272 | 32 | 96,612 | 189,382 |
| V21 DOWN | 1,655,260 | 5.518 ms | 1,024 | 16 | 32 | 105,126 | 208,933 |

Hai report độc lập cộng lại là 10.639 ms. Khi UP và DOWN chạy dataflow qua stream, throughput hệ thống bị chặn bởi kernel chậm hơn, nên không được cộng cycle một cách máy móc. Phải đo cả:

```text
standalone latency UP
standalone latency DOWN
pipeline fill/drain
end-to-end interval của chuỗi UP -> DOWN
```

Post-route FFN-only hiện cho thấy:

| SLR | LUT | FF | BRAM tile | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| SLR2 | 25.61% | 25.41% | 33.93% | 10.00% | 36.56% |
| SLR3 | 25.27% | 25.25% | 11.38% | 10.00% | 33.43% |

SLR2 còn phải chứa Attention output projection và residual/norm. Vì vậy INT8 không chỉ phải nhanh hơn mà còn phải giảm footprint của FFN UP.

## 3. Kiến trúc số học được chọn

Pass đầu dùng mixed precision có kiểm soát:

```text
Attention/LayerNorm FP32 stream
        |
        v
quantize activation FP32 -> INT8 trong UP
        |
        v
W1 INT8 x A8 INT8 -> accumulate INT32
        |
        v
scale + bias + GELU, sau đó requantize INT8
        |
        v
512-bit gelu_i8_stream: 64 phần tử INT8 mỗi word
        |
        v
W2 INT8 x A8 INT8 -> accumulate INT32
        |
        v
dequantize + bias -> FP32
        |
        v
residual/LayerNorm/output FP32
```

Không lượng tử hóa LayerNorm, residual add hoặc output boundary trong pass đầu. Linear W1/W2 chiếm phần lớn MAC nên đây là nơi INT8 đem lại tốc độ và tiết kiệm tài nguyên lớn nhất, trong khi biên FP32 hạn chế tích lũy sai số giữa các encoder layer.

### 3.1 Định dạng lượng tử hóa mặc định

Profile được ưu tiên:

```text
Weights:     signed INT8, symmetric, per-output-channel
Activations: signed INT8, symmetric, static per-tensor
Zero point:  0
Range:       [-127, 127]
Accumulator: signed INT32
Bias:        signed INT32 ở miền accumulator
Rounding:    round-to-nearest-even
Overflow:    saturate, không wrap
```

Không dùng asymmetric zero-point trong pass đầu vì correction term làm tăng logic, latency và fanout. Không dùng W4 trước khi W8A8 đạt accuracy và route sạch.

### 3.2 Công thức UP

Với input scale `Sx`, weight scale theo output channel `Sw1[i]`:

```text
Xq      = clamp(round(X / Sx), -127, 127)
W1q[i] = clamp(round(W1[i] / Sw1[i]), -127, 127)
B1q[i] = round(B1[i] / (Sx * Sw1[i]))

ACC1[s,i] = B1q[i] + sum_h(Xq[s,h] * W1q[i,h])
Y1[s,i]   = GELU(ACC1[s,i] * Sx * Sw1[i])
Gq[s,i]   = clamp(round(Y1[s,i] / Sg), -127, 127)
```

`Sg` là một scale activation chung cho output GELU của mỗi layer. Không dùng scale khác nhau theo intermediate channel trong pass đầu vì DOWN sẽ cần rescale từng tích trước khi cộng.

### 3.3 Công thức DOWN

Với weight scale theo output hidden channel `Sw2[h]`:

```text
W2q[h] = clamp(round(W2[h] / Sw2[h]), -127, 127)
B2q[h] = round(B2[h] / (Sg * Sw2[h]))

ACC2[s,h] = B2q[h] + sum_i(Gq[s,i] * W2q[h,i])
Y2[s,h]   = ACC2[s,h] * Sg * Sw2[h]
```

`Y2` được trả về FP32 trước residual/LayerNorm. Nếu residual nằm ở DOWN, residual phải giữ FP32 và chỉ cộng sau dequantization.

### 3.4 Kiểm tra overflow INT32

Biên thô khi không tính bias:

```text
UP:   768  * 127 * 127 = 12,387,072
DOWN: 3072 * 127 * 127 = 49,548,288
```

Cả hai nhỏ hơn nhiều so với `INT32_MAX`. Script pack/calibration vẫn phải kiểm tra bias đã lượng tử hóa và log accumulator lớn nhất trên calibration set. Không được đổi accumulator xuống INT16.

## 4. Contract stream và memory

### 4.1 Input của UP

Giữ `attn_mid_stream` rộng 512 bit và chứa 16 FP32/word trong pass đầu. UP quantize mỗi activation một lần khi nạp vào buffer INT8.

Lợi ích:

- Không buộc Attention/LayerNorm đổi sang INT8 ngay.
- Có thể so sánh INT8 FFN với FP32 V21 mà không thay đổi phần còn lại của model.
- Tránh lượng tử hóa nhiều lần giữa các kernel.

Sau khi accuracy ổn, có thể thử optional profile để Attention output phát INT8 trực tiếp, nhưng đó là pass riêng và phải có full-model calibration.

### 4.2 Stream UP sang DOWN

Dùng cùng bề rộng AXIS 512 bit nhưng pack 64 activation INT8 mỗi word:

```text
typedef ap_uint<512> packed_i8x64_t;
typedef hls::stream<packed_i8x64_t> packed_i8_stream_t;
```

Số word cho tensor `[128][3072]`:

```text
FP32 V21: 128 * 3072 / 16 = 24,576 words
INT8:     128 * 3072 / 64 =  6,144 words
```

Như vậy crossing SLR2 -> SLR3 giảm 4 lần số beat. Producer order và consumer order phải giống nhau:

```text
[intermediate_tile][sequence][pack_in_tile]
```

Không thêm layout converter nếu có thể sửa loop order ở producer/consumer.

### 4.3 Weight layout

W1 và W2 phải được pack tile-major từ host, không transpose hoặc gather rời rạc trong kernel:

```text
W1: [out_tile][k_tile][k_lane][out_lane]
W2: [out_tile][k_tile][k_lane][out_lane]
```

Một AXI word 512 bit chứa 64 weight INT8. Layout chính xác phải làm cho loop load trong cùng một tile đọc địa chỉ tăng liên tục và đạt burst.

Scale và bias có thể dùng buffer riêng:

```text
W1 scale: 3072 FP32
W1 bias:  3072 INT32
W2 scale:  768 FP32
W2 bias:   768 INT32
```

Các mảng nhỏ này được preload vào BRAM/URAM và không đọc DDR trong hot MAC loop.

## 5. Kiến trúc compute HLS

### 5.1 MAC core

MAC core INT8 dùng reduction tree hoặc partial sums có pipeline rõ ràng:

```text
INT8 x INT8 products
    -> widen ngay sang INT32
    -> balanced adder tree INT32
    -> accumulate INT32
```

Không cộng trong INT8/INT16. Không tạo một adder chain dài. Mỗi hot loop phải đạt `II=1`.

### 5.2 Profile sweep

Không synthesize mọi tổ hợp. Chạy theo thứ tự sau:

| Profile | K_PAR | OUT_PAR | MAC/cycle | Mục đích |
|---|---:|---:|---:|---|
| I8-CORRECT | 16 | 16 | 256 | Bit-accurate, so sánh trực tiếp V21 |
| I8-ROUTE | 32 | 16 | 512 | Mục tiêu route dễ, khoảng 2x compute |
| I8-BALANCED | 32 | 32 | 1,024 | Profile ưu tiên, mục tiêu khoảng 4x compute |
| I8-WIDE-K | 64 | 16 | 1,024 | Chỉ thử nếu banking tốt hơn 32x32 |

`I8-BALANCED` là điểm bắt đầu hợp lý cho hiệu năng cuối. `I8-ROUTE` là fallback nếu 32x32 gây congestion hoặc timing âm.

Không tăng đồng thời tile size, `K_PAR`, `OUT_PAR` và FIFO depth trong một pass. Mỗi report phải chỉ ra thay đổi nào tạo ra kết quả.

### 5.3 DSP mapping

Thứ tự triển khai:

1. Map một signed INT8 multiply vào một DSP để có baseline ổn định.
2. Đo DSP, LUT, II và timing.
3. Chỉ sau khi bit-accurate và route sạch mới thử packing hai INT8 multiply vào một DSP48E2.
4. Packing phải có test exhaustive/random cho signed corner cases; không chấp nhận cross-term sai.

Mục tiêu của packing là 1,024 INT8 MAC/cycle với khoảng 512 DSP thay vì khoảng 1,024 DSP. Không dùng LUT multiplier hàng loạt để đổi DSP lấy congestion LUT.

### 5.4 Buffer và banking

Đề xuất ban đầu:

```text
input INT8 buffer: URAM hoặc BRAM, bank theo K_PAR
weight tile:       BRAM, partition theo OUT_PAR và K_PAR vừa đủ port
accumulator:       INT32 BRAM/register bank theo OUT_PAR
GELU tile:         stream trực tiếp, không lưu full [128][3072]
output FP32:       tile buffer; tránh full matrix nếu residual/norm consume stream được
```

Ưu tiên cyclic partition theo số lane. Complete partition chỉ dùng cho vector nhỏ trong một PE; không complete partition tensor lớn.

Dùng ping-pong weight tile để overlap load và compute sau khi core đơn đã đúng. Nếu dataflow sinh deadlock hoặc tăng BRAM quá nhiều, quay về buffer đơn và đo lại.

## 6. GELU và requantization

GELU không phải bottleneck MAC chính nhưng có thể trở thành bottleneck sau khi GEMM tăng 4 lần. Thử theo ba bước:

| GELU profile | Cách làm | Vai trò |
|---|---|---|
| G0 | Dequantize FP32 + PWL hiện tại + quantize INT8 | Golden hardware mixed-precision |
| G1 | Fixed-point PWL, hệ số đã lượng tử hóa | Giảm FP operator |
| G2 | LUT 256 hoặc 512 entry + interpolation tối thiểu | Profile hiệu năng cuối |

G0 phải pass accuracy trước. G1/G2 chỉ được chấp nhận nếu sai số toàn model vẫn trong budget.

LUT phải cover miền sau calibration, ví dụ `[-6, 6]`, và clamp ngoài miền. Không hard-code miền trước khi có histogram activation.

## 7. Calibration và kiểm soát accuracy

### 7.1 Dataset calibration

Dùng 512 đến 2,048 mẫu đại diện đúng task, đúng sequence length và preprocessing của model. Calibration phải thu histogram riêng cho từng encoder layer:

```text
input FFN sau LayerNorm
pre-GELU W1 accumulator đã dequantize
post-GELU activation
output W2 trước residual
```

Không calibrate chỉ bằng random tensor.

### 7.2 Thứ tự quantization

1. PTQ symmetric W8A8, weight per-channel và activation per-tensor.
2. Dùng percentile/MSE clipping thay vì luôn lấy absolute max.
3. Nếu activation outlier làm accuracy giảm, thử SmoothQuant offline để chuyển độ khó từ activation sang weight.
4. Nếu vẫn không đạt, thử activation per-token cho input UP, nhưng phải tính chi phí max-reduction và reciprocal.
5. Nếu PTQ vẫn thất bại, dùng QAT; không nhảy ngay xuống W4.

### 7.3 Golden references

Cần hai reference:

```text
Reference A: output FP32 V21 với GELU PWL hiện tại
Reference B: model framework FP32 với GELU chuẩn
```

Reference A tách sai số do INT8 khỏi sai số vốn có của PWL. Reference B đo ảnh hưởng thật lên model.

### 7.4 Ngưỡng chấp nhận

Ngưỡng đề xuất:

| Cấp kiểm tra | Pass ưu tiên | Reject |
|---|---:|---:|
| Cosine similarity output một FFN | >= 0.999 | < 0.995 |
| Cosine similarity final hidden 12 layer | >= 0.995 | < 0.990 |
| Saturation activation | < 0.1% | > 1.0% |
| Task accuracy/F1 drop | <= 0.5 điểm % | > 1.0 điểm % |

Ngoài ra phải report MAE, RMSE chuẩn hóa, max absolute error và metric task gốc. Không kết luận accuracy chỉ từ một tensor hoặc cosine similarity.

## 8. Mục tiêu performance và tài nguyên

### 8.1 Latency

Mục tiêu cho mỗi kernel tại 300 MHz:

| Mức | UP cycles | DOWN cycles | Ý nghĩa |
|---|---:|---:|---|
| Tối thiểu | <= 800,000 | <= 850,000 | Khoảng 2x so với V21 |
| Chấp nhận tốt | <= 500,000 | <= 550,000 | Khoảng 3x |
| Mục tiêu | <= 400,000 | <= 420,000 | Khoảng 4x |
| Stretch | <= 320,000 | <= 340,000 | Gần giới hạn 1,024 MAC/cycle |

Mục tiêu end-to-end cho pipeline UP -> DOWN là interval không quá 500,000 cycle, tương đương 1.667 ms tại 300 MHz, chưa gồm Attention và host transfer.

### 8.2 Tài nguyên

Mục tiêu kernel INT8:

| Kernel | DSP mục tiêu | BRAM/URAM | LUT/FF |
|---|---:|---|---|
| UP SLR2 | <= 700, ưu tiên <= 600 | Không cao hơn V21; ưu tiên giảm | Không tăng quá 15% so với V21 |
| DOWN SLR3 | <= 700, ưu tiên <= 600 | Không cao hơn V21 | Không tăng quá 15% so với V21 |

Sau full link với Attention:

```text
SLR2 và SLR3: ưu tiên < 60% mỗi loại resource
Hard route-friendly ceiling: < 70% LUT/BRAM/DSP trên mỗi SLR
SLL crossing SLR2-SLR3: không tăng so với FP32; số AXIS beat phải giảm khoảng 4x
```

### 8.3 Timing

Điều kiện bắt buộc:

```text
HLS target clock: 3.333 ns
Hot MAC loop: II=1
Post-route setup WNS: >= 0 ns
Target timing margin: WNS >= +0.20 ns trên kernel clock
TNS: 0
Hold violations: 0
```

`-0.00 ns` không được xem là margin. Kết quả chỉ đủ tin cậy khi xem giá trị có nhiều chữ số hơn và report post-route.

## 9. Các pass triển khai

### Pass INT8-0: Khóa golden và calibration

Deliverables:

```text
scripts/calibrate_ffn_int8.py
scripts/pack_ffn_int8_weights.py
reports/int8/quantization_summary.csv
reports/int8/accuracy_summary.md
```

Phải có scale cho từng layer, saturation rate và metric task trước khi sửa HLS.

### Pass INT8-1: Kernel correctness

Tạo source riêng, một extern C top trong mỗi file:

```text
src/int8/ffn_int8_common.h
src/int8/bert_ffn_up_int8.cpp
src/int8/bert_ffn_down_int8.cpp
```

Tên kernel ngắn để tránh giới hạn 64 ký tự của `v++`:

```text
ffn_up_i8_v1
ffn_down_i8_v1
```

Profile: `K_PAR=16`, `OUT_PAR=16`, GELU G0. So sánh bit-accurate với Python quantized golden. Csim có thể không dùng cho sweep thông thường, nhưng functional comparison cho INT8 là bắt buộc trước csynth lớn.

### Pass INT8-2: Route profile

Profile: `K_PAR=32`, `OUT_PAR=16`.

Mục tiêu:

- II=1.
- Không có port conflict.
- Latency giảm gần 2 lần.
- DSP thấp hơn FP32 V21.
- Csynth nhanh và hierarchy dễ đọc.

### Pass INT8-3: Balanced 4x profile

Profile: `K_PAR=32`, `OUT_PAR=32`.

Mục tiêu:

- Khoảng 1,024 MAC/cycle.
- UP/DOWN đạt khoảng 400k cycle.
- DSP <= 700 sau packing hoặc resource sharing có kiểm soát.
- Không tăng LUT routing quá mức.

Nếu timing âm, ưu tiên sửa reduction pipeline, register fanout, banking và tile hierarchy. Không giảm song song ngay nếu nguyên nhân là đường điều khiển hoặc address generation.

### Pass INT8-4: GELU fixed/LUT

Chỉ thực hiện sau khi GEMM profile đã ổn. So sánh G0/G1/G2 với cùng MAC configuration để biết chính xác lợi ích và sai số của GELU.

### Pass INT8-5: Full link SLR-aware

Connectivity giữ nguyên kiến trúc:

```text
slr=up8:SLR2
slr=down8:SLR3
stream_connect=up8.gelu_i8_stream:down8.gelu_i8_stream
```

Weight W1 ở DDR gần SLR2, W2 ở DDR gần SLR3. Full link phải chứa kernel Attention nhỏ trên SLR2 trước khi kết luận fit.

## 10. Ma trận report bắt buộc

Mỗi version phải thêm một dòng, không ghi đè version trước:

| Version | Quant | K_PAR | OUT_PAR | GELU | UP cyc | DOWN cyc | UP DSP | DOWN DSP | UP BRAM/URAM | DOWN BRAM/URAM | II | HLS clk | WNS | Accuracy drop |
|---|---|---:|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|

Ngoài bảng tổng hợp, lưu:

```text
csynth_up_int8_<version>.rpt
csynth_down_int8_<version>.rpt
kernel_util_routed_<version>.rpt
slr_util_routed_<version>.rpt
timing_summary_routed_<version>.rpt
accuracy_<version>.json
```

Tốc độ chỉ được tính bằng:

```text
speedup_up   = V21_UP_cycles / INT8_UP_cycles
speedup_down = V21_DOWN_cycles / INT8_DOWN_cycles
speedup_pipe = V21_pipeline_interval / INT8_pipeline_interval
```

Không dùng chỉ số clock estimate của csynth để claim bitstream performance.

## 11. Điều kiện chấp nhận kiến trúc INT8 cuối

Một version chỉ được chọn làm production INT8 khi đồng thời đạt:

```text
[ ] W1/W2 là W8A8 symmetric, accumulator INT32.
[ ] LayerNorm/residual và output boundary vẫn FP32.
[ ] UP/DOWN hot loop II=1.
[ ] Pipeline interval <= 500,000 cycle tại 300 MHz.
[ ] UP và DOWN mỗi kernel dùng <= 700 DSP.
[ ] Full Attention + FFN fit SLR2/SLR3 dưới ngưỡng route-friendly.
[ ] Post-route WNS >= 0, mục tiêu >= +0.20 ns; TNS = 0.
[ ] Không có hold violation hoặc severe congestion.
[ ] SLR2->SLR3 dùng packed INT8 stream và không deadlock.
[ ] Saturation < 0.1% trên calibration/validation set.
[ ] Final task accuracy/F1 giảm <= 0.5 điểm phần trăm.
[ ] Output được so với cả V21 PWL và framework FP32.
```

Nếu một version nhanh nhưng accuracy giảm trên 1 điểm phần trăm, version đó bị loại. Nếu accuracy tốt nhưng timing âm, version đó chưa được dùng làm production.

## 12. Những hướng chưa làm trong pass đầu

```text
- Không W4A8 hoặc W4A4.
- Không quantize LayerNorm/Softmax/residual ngay.
- Không chuyển mapping UP khỏi SLR2 hoặc DOWN khỏi SLR3.
- Không fuse UP và DOWN thành một kernel vượt SLR.
- Không tăng FIFO rất sâu để che producer/consumer mismatch.
- Không dùng LUT multiplier hàng loạt chỉ để giảm số DSP.
- Không thay full-model FP32 boundary trước khi có calibration.
```

## 13. Thứ tự thực hiện ngay tiếp theo

```text
1. Viết Python calibration/quantized golden cho W1, GELU, W2.
2. Chốt scale W1/W2 per-output-channel và activation scale per-layer.
3. Pack weight tile-major 64 INT8/512-bit word.
4. Implement ffn_up_i8_v1 và ffn_down_i8_v1 profile 16x16.
5. Kiểm tra bit-accurate, saturation và output error.
6. Csynth 16x16, sau đó 32x16.
7. Chỉ khi 32x16 II=1 mới mở 32x32.
8. Tối ưu GELU LUT sau khi MAC không còn là vấn đề correctness.
9. Link cùng Attention trên SLR2/SLR3 và kiểm tra post-route.
10. Chọn production INT8 theo latency + resource + timing + accuracy, không theo một metric riêng lẻ.
```

## 14. Reference nội bộ

Các nguyên tắc trong kế hoạch này bám theo các ghi chú sẵn có:

```text
../docs/2312_15159_spatial_llm_slr.md
../docs/2409_13975_protea_transformer_tiling.md
../docs/2501_09118_stream_hls_dataflow.md
../docs/2509_13694_streamtensor_llm_streaming.md
../docs/heteroflow_fpga2022_data_placement.md
../docs/u250_ds962_resource_information.md
```

Kết luận kiến trúc: bắt đầu bằng W8A8 cho hai GEMM, giữ nonlinear/boundary FP32, truyền GELU INT8 qua SLR2->SLR3, tích lũy INT32, và dùng profile 32x32 làm mục tiêu 4x. Chỉ mở rộng packing hoặc quantization mạnh hơn sau khi version W8A8 đạt đồng thời accuracy và post-route timing.
