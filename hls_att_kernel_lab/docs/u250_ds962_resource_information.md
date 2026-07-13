# DS962 — Alveo U200/U250 resource information: ràng buộc platform cho SLR optimization

**Nguồn:** `resource_information.pdf`  
**Tên tài liệu:** *Alveo U200 and U250 Data Center Accelerator Cards Data Sheet (DS962) — FPGA Resource Information*  
**Mức hữu ích cho tối ưu SLR U250:** Bắt buộc. Đây không phải paper kiến trúc, nhưng là source of truth cho platform U250.

## Một câu tóm tắt

U250 không phải U280: U250 dùng XCU250 với **4 SLR** và **4 DDR4 DIMM 16 GB**, nên mọi kế hoạch SLR phải theo floorplan U250, không copy floorplan 3 SLR của paper U280.

## Ràng buộc platform chính

Tài liệu ghi:

```text
U200: XCU200, 3 SLR
U250: XCU250, 4 SLR
```

U250 có:

```text
- 4 SLR: SLR0, SLR1, SLR2, SLR3
- PCIe Gen3 x16
- 4 DDR4 16 GB, 2400 MT/s, 64-bit ECC DIMMs
- Tổng DDR4: 64 GB
- QSFP28 connectors
```

## Floorplan logic

Theo hình floorplan XCU250:

```text
SLR3: gần DDR4 phía trên
SLR2: gần DDR4 và QSFP
SLR1: gần DDR4
SLR0: gần DDR4 và PCIe/static platform
```

Điểm cần nhớ:

```text
Static platform của Vitis dùng một phần resource FPGA.
Resource còn lại phụ thuộc version platform.
Không nên budget 100% theoretical resource cho user kernels.
```

## Hệ quả cho repo BERT

### 1. Không copy U280 3-SLR mapping

Paper 2312.15159 chia U280 thành 3 region. Repo target U250 nên cần tư duy 4 region:

```text
Candidate current mapping:
SLR1: embedding_prep
SLR0: qkv
SLR1: attn_core
SLR2: attn_out_norm + ffn_up_gelu
SLR3: ffn_down_norm
```

Mapping hiện tại có thể giữ cho pass đầu, nhưng đánh giá lại sau post-route.

### 2. DDR channel locality

Vì có 4 DDR4 channels, kế hoạch dài hạn nên kiểm tra:

```text
- weight buffers đọc từ DDR nào,
- attn_mid_ddr nằm DDR nào,
- host schedule có làm nhiều kernels tranh cùng DDR không,
- kernel ở SLR nào đọc DDR nào nhiều nhất.
```

### 3. Static region ảnh hưởng resource

Nếu report utilization gần ngưỡng, cần nhớ:

```text
Vitis static platform đã consume resource.
Pblock/SLR available resource thực tế thấp hơn raw device.
```

Vì vậy rule 70%/SLR là hợp lý hơn 90%/SLR cho route-friendly pass.

## SLR optimization checklist

```text
[ ] Mỗi major kernel có per-kernel utilization.
[ ] Full link có per-SLR utilization.
[ ] Stream crossing giữa SLR được liệt kê.
[ ] DDR bank assignment được liệt kê.
[ ] Không SLR nào vượt ~70% LUT/BRAM trong route-friendly pass.
[ ] QKV không tạo congestion hotspot trong SLR0.
[ ] Nếu SLR0 congested, giảm QKV parallelism trước khi đổi toàn pipeline.
```

## Mapping strategy gợi ý cho U250

### Pass 1: Giữ mapping hiện tại

```text
Không đổi SLR mapping khi QKV layout còn sai.
Sửa QKV tile-major + DSP binding trước.
```

### Pass 2: Sau QKV clean, đo lại

```text
Nếu SLR0 nghẽn:
- giảm QKV MAC_LANES 8 -> 4
- hoặc chia/đặt lại QKV stage nếu thật sự cần

Nếu SLR2 nghẽn:
- tách ffn_up_gelu hoặc attn_out_norm sang SLR khác
- xem SLR3 còn headroom không

Nếu crossing SLR1-SLR2/SRL2-SLR3 cao:
- xem lại stream order và FIFO depth
```

## Không nên làm

```text
- Không dùng U280/HBM assumption.
- Không giả định DDR bandwidth như U280 HBM.
- Không dùng hết resource mỗi SLR.
- Không đặt FIFO rất lớn qua SLR crossing nếu chưa đo routing.
```

## Kết luận dùng cho kế hoạch

DS962/resource_information là tài liệu nền để khóa lại target: **U250 = 4 SLR + 4 DDR4**, khác paper U280. Vì vậy hướng đúng là lấy nguyên lý split-kernel/SRL partition từ paper, nhưng quyết định mapping dựa trên U250 report thực tế.
