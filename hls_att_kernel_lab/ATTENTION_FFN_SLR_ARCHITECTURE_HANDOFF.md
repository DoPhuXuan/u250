# BERT U250 Attention-FFN SLR Architecture Handoff

Ngay cap nhat: 2026-07-09

Tai lieu nay la handoff cho nguoi toi uu Attention tren Alveo U250/XCU250.
Muc tieu la giu tuong thich voi FFN split-kernel hien tai va kien truc 4 SLR
da chon. Khong thay doi phan tang SLR neu chua co report post-route chung minh
mapping khong route duoc.

## 1. Target va tensor shape

```text
Board                 Alveo U250
Part                  xcu250-figd2104-2L-e
Tool                  Vitis/Vivado 2021.2
Target clock          300 MHz (3.333 ns)
Sequence length       128
Hidden size           768
Intermediate size     3072
External stream width 512 bit
Stream payload        16 x IEEE-754 float32
```

So word cua cac tensor chinh:

```text
hidden tensor:       128 * 768 / 16  = 6,144 word
intermediate tensor: 128 * 3072 / 16 = 24,576 word
```

## 2. Kien truc SLR muc tieu

| SLR | Kernel/block so huu | Trang thai |
|---|---|---|
| SLR0 | Q/K/V projection | Attention chua implement trong lab FFN |
| SLR1 | Attention score, mask, softmax, weighted sum | Attention chua implement trong lab FFN |
| SLR2 | Attention output projection, residual/norm, FFN UP + GELU | FFN UP da co; attention output/norm chua ghep |
| SLR3 | FFN DOWN, residual/norm, final output | FFN DOWN da co; residual/norm chua ghep |

Do thi full target:

```text
DDR input/weights
      |
      v
SLR0: QKV projection
      |
      | Q/K/V packed stream(s)
      v
SLR1: score -> mask -> softmax row -> weighted sum
      |
      | context_stream, row-major hidden
      v
SLR2: output projection -> attention residual/LN -> stream fork
      |                                      |
      | attn_mid_stream                      | ffn_residual_stream
      v                                      |
SLR2: FFN UP -> GELU                         |
      |                                      |
      | gelu_stream                          |
      +----------------------+---------------+
                             v
SLR3: FFN DOWN -> residual add -> layer norm -> output
```

Kien truc hien tai trong FFN-only link dung feeder DDR tren SLR2 de thay cho
Attention:

```text
DDR[2] -> bert_ffn_input_v15_feeder_kernel
       -> attn_mid_stream
       -> bert_ffn_up_gelu_*
       -> gelu_stream
       -> bert_ffn_down_*
       -> DDR[0]
```

Feeder chi la test infrastructure. Khi ghep full encoder, xoa feeder va noi
output cua `attention output projection + residual/LN` truc tiep vao
`attn_mid_stream`.

## 3. SLR0: Q/K/V projection contract

SLR0 so huu toan bo linear projection tao Q, K, V. Attention optimizer can
quyet dinh ba kernel rieng hay mot kernel fused, nhung phai bao cao:

```text
- per-kernel DSP/LUT/BRAM/URAM
- II va latency
- so stream cat bien SLR0 -> SLR1
- layout/order cua moi stream
- routed congestion trong SLR0
```

Khuyen nghi ban dau:

```text
- Wq/Wk/Wv dung packed 512-bit va tile-major layout.
- Reuse input tile cho ca Q, K, V neu fuse khong lam routing xau.
- Khong de Q/K/V quay ve DDR chi de attention core doc lai.
- Neu dung 3 stream, ca ba phai co order ro rang va consumer SLR1 doc deu.
- Neu fuse thanh 1 stream, packet phai co tag/lich co dinh; khong chen metadata
  tuy bien lam mat 512-bit payload efficiency.
```

Quyet dinh 1-vs-3 kernel la mo. Chot bang post-route va end-to-end throughput,
khong chi bang csynth latency.

## 4. SLR1: Attention core contract

SLR1 so huu:

```text
QK^T theo head/row
apply_attention_mask
softmax theo row
attention weighted sum
combine heads neu can
```

Muc tieu:

```text
- Xu ly theo head hoac row de khong luu full score/probability tensor.
- Stream probability row truc tiep vao weighted sum neu lich consumer cho phep.
- Softmax giu expf baseline truoc; LUT/approx la pass rieng co accuracy test.
- Output context theo row-major hidden order de SLR2 consume truc tiep.
```

SLR1 khong nen xuat mot layout buoc SLR2 transpose/reorder full tensor. Neu
layout QKV noi bo khac, dat converter trong SLR1 va bao cao chi phi cua no.

## 5. SLR1 -> SLR2 stream contract

Stream output cua Attention core nen co cung logical order voi FFN input:

```text
stream name     context_stream (ten tam)
width           512 bit
word payload    16 float32 lien tiep
order           for s = 0..127
                  for h_pack = 0..47
                    lanes = hidden[s][16*h_pack : 16*h_pack+15]
word count      6,144
target II       1 word/cycle sau khi pipeline day
```

SLR2 output projection + residual/LN phai giu order nay khi tao
`attn_mid_stream`.

## 6. SLR2: Attention output/norm va FFN UP

SLR2 da duoc danh rieng cho hai khoi:

```text
1. Attention output projection + residual + layer norm
2. FFN UP (768 -> 3072) + GELU
```

FFN UP stable baseline:

```text
bert_ffn_up_gelu_v15_tilemajor_uram_bindop_kernel
K_PAR       = 16
MAC_GROUP   = 16
TILE_I      = 16
input buf   = URAM, cyclic factor 16
W1 tile     = BRAM, tile-major
GELU        = piecewise-linear approximation
```

Current experiment:

```text
bert_ffn_up_gelu_v21_dotpipe_kernel
status: khong chap nhan lam final route candidate
result: hierarchy/II dung, nhung fmul van la _4_max_dsp; DSP/FF tang
```

### `attn_mid_stream` contract

```text
width       512 bit
order       [sequence][hidden_pack]
word count  6,144
lane order  hidden index tang dan trong moi word
```

FFN UP hien doc toan bo stream vao `input_buf[128][768]`, sau do compute W1.
Attention producer co the bat dau stream ngay khi row dau tien san sang; FIFO
khong can chua ca tensor.

### W1 layout

W1 tile-major memory order:

```text
[intermediate_tile_16][hidden_k][16 intermediate output lanes]
```

W1 va B1 dang gan `DDR[2]`.

## 7. SLR2 -> SLR3 `gelu_stream` contract

Day la crossing da route duoc trong V15 va khong nam trong top critical paths.

```text
width       512 bit
FIFO depth  512 word trong connectivity config
word count  24,576
order       for i0 = 0..3071 step 16
              for s = 0..127
                lanes = gelu[s][i0 : i0+15]
logical     [intermediate_tile][sequence]
target II   1 word/cycle tai write/read loops
```

Khong doi `gelu_stream` thanh row-major neu khong sua dong bo DOWN consumer.
Order tile-major cho phep DOWN consume tung tile 16 activation va cap nhat
output accumulator.

## 8. SLR3: FFN DOWN va residual/norm

FFN DOWN stable baseline:

```text
bert_ffn_down_v15_tilemajor_uram_bindop_kernel
projection   3072 -> 768
K_PAR        = 16
MAC_GROUP    = 16
TILE_O       = 32
output buf   = URAM, cyclic factor 16
W2 tile      = BRAM, tile-major
```

Current experiment:

```text
bert_ffn_down_v21_dotpipe_kernel
status: khong chap nhan lam final route candidate vi fmul critical IP khong doi
```

W2 va B2 dang gan `DDR[3]`. Packed output hien ghi vao `DDR[0]`.

### Khoang trong hien tai: FFN residual + layer norm

Kernel DOWN hien tai chi khoi tao accumulator bang B2, cong W2, roi ghi output.
No chua implement residual va layer norm.

Huong ghep full kernel de xuat:

```text
SLR2 attention residual/LN output
    -> fork A: attn_mid_stream vao FFN UP
    -> fork B: ffn_residual_stream sang SLR3

SLR3 DOWN init:
    output_buf[s][h] = residual[s][h] + b2[h]
    output_buf += W2 * GELU
    layer_norm(output_buf)
    write/stream final output
```

Cach nay tai su dung `output_buf` URAM hien co va khong can mot full residual
buffer thu hai. `ffn_residual_stream` nen dung cung row-major contract 6,144
word nhu `attn_mid_stream`.

Khong them residual stream vao link cho den khi DOWN kernel co consumer that;
mot stream khong duoc drain co the lam deadlock.

## 9. DDR/bank mapping muc tieu

Mapping hien tai da xac nhan cho FFN:

| Data | Bank |
|---|---|
| FFN input feeder, W1, B1 | DDR[2] |
| W2, B2 | DDR[3] |
| FFN packed output | DDR[0] |

Mapping goi y cho full Attention, can xac nhan bang link report:

| SLR | Data/weight | Bank tam de xuat |
|---|---|---|
| SLR0 | input, Wq, Wk, Wv | DDR[0] |
| SLR1 | mask hoac spill buffer neu bat buoc | DDR[1] |
| SLR2 | attention output weight, LN params, W1/B1 | DDR[2] |
| SLR3 | W2/B2, final LN params | DDR[3] |
| Final output | host-visible output | DDR[0] hoac bank do host schedule chon |

Day la starting point, khong phai ket qua floorplan cuoi. Neu hai master tranh
cung bank trong cung thoi gian, can doi bank hoac overlap schedule.

## 10. Routed FFN baseline va headroom

V15 la baseline da full-link/post-route tai 300 MHz:

```text
WNS  = -0.033 ns
TNS  = -5.784 ns
WHS  =  0.005 ns
setup failing endpoints = 336
```

Top failing paths la internal FP multiplier/DSP trong UP va DOWN, khong phai
AXIS crossing, DDR adapter, URAM/BRAM hay SLL.

Kernel routed utilization:

| CU | LUT | REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| FFN-only feeder | 1,716 | 4,432 | 7 | 0 | 0 |
| V15 UP | 71,610 | 134,107 | 143 | 32 | 1,280 |
| V15 DOWN | 79,150 | 163,887 | 23 | 32 | 1,280 |

Full SLR utilization cua FFN-only routed design:

| SLR | LUT | REG | BRAM tile | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| SLR2 | 113,545 (26.28%) | 194,519 (22.51%) | 247 (36.76%) | 32 (10.00%) | 1,283 (41.76%) |
| SLR3 | 108,725 (25.17%) | 206,822 (23.94%) | 87 (12.95%) | 32 (10.00%) | 1,283 (41.76%) |

SLR2-SLR3 SLL use la 2,789 / 23,040 (12.11%). Crossing hien tai con headroom.
Rui ro chinh la DSP/MAC timing local, khong phai so luong SLL.

Attention output projection va norm dat tren SLR2 phai tinh den FFN UP da dung
khoang 42% DSP cua SLR2 trong full linked design. Khong budget SLR2 dua tren
raw device resource ma bo qua static platform va FFN.

## 11. Trang thai toi uu FFN

| Version | Ket qua |
|---|---|
| V15 | Baseline tot nhat da route; II=1, latency/resource tot, WNS -0.033 ns |
| V18 | Rejected; staging tang latency UP 10.59%, DOWN 12.72% |
| V20 | Rejected; `BIND_OP latency=5` bi inline hap thu, RTL fmul khong doi |
| V21 | Partial pass, not accepted; hierarchy/II dung nhung fmul IP khong doi, FF/DSP tang |

V20 report tai 300 MHz:

| Kernel | Latency | DSP | BRAM | URAM | LUT | FF | Hot-loop II |
|---|---:|---:|---:|---:|---:|---:|---:|
| UP | 1,522,568 | 1,280 | 302 | 32 | 89,617 | 154,429 | 1 |
| DOWN | 1,627,612 | 1,280 | 76 | 32 | 92,851 | 183,625 | 1 |

V21 chi duoc chap nhan neu:

```text
- dot-product xuat hien thanh hierarchy rieng
- dot function II=1
- hot MAC UP/DOWN II=1
- DSP van gan 1,280/kernel
- latency tang <3% so voi V20 300 MHz
- multiplier RTL/latency thuc su khac V20
- post-route WNS >= 0 tai 300 MHz
```

Report V21 cho thay dieu kien hierarchy va II dat, nhung multiplier van la
`fmul_32ns_32ns_32_4_max_dsp` voi reported latency 3. V21 UP tang 96 DSP va
13.57% FF; V21 DOWN tang 8.92% FF. Vi critical multiplier IP khong doi, V21
khong duoc chon lam final route candidate. Pass tiep theo nen thay doi truc
tiep implementation cua fmul, thay vi chi them hierarchy/register.

## 12. Connectivity skeleton

FFN V21 config hien tai:

```ini
[connectivity]
nk=bert_ffn_input_v15_feeder_kernel:1:in1
nk=bert_ffn_up_gelu_v21_dotpipe_kernel:1:up1
nk=bert_ffn_down_v21_dotpipe_kernel:1:down1

stream_connect=in1.attn_mid_stream:up1.attn_mid_stream:512
stream_connect=up1.gelu_stream:down1.gelu_stream:512

slr=in1:SLR2
slr=up1:SLR2
slr=down1:SLR3

sp=in1.input_hidden:DDR[2]
sp=up1.w1:DDR[2]
sp=up1.b1:DDR[2]
sp=down1.w2:DDR[3]
sp=down1.b2:DDR[3]
sp=down1.output_hidden:DDR[0]
```

Full Attention se thay `in1` bang kernel SLR2 tao `attn_mid_stream`. Them
QKV/attention kernels theo aliases ngan de tranh gioi han 64 ky tu cua Vitis.

## 13. Rule cho nguoi toi uu Attention

1. Giu SLR0 = QKV, SLR1 = attention core trong pass dau.
2. Khong chuyen FFN UP khoi SLR2 hoac DOWN khoi SLR3 chi dua tren csynth.
3. Moi stream crossing phai ghi width, token order, word count, FIFO depth va
   producer/consumer II.
4. Khong luu full attention probability tensor neu row streaming duoc.
5. Khong tao DDR hand-off giua weighted sum va output projection neu stream
   row-major da match.
6. Bao cao per-kernel va per-SLR utilization sau full link.
7. Bao cao post-route WNS/TNS; HLS timing `0.00` khong du de ket luan.
8. Kiem tra top critical path thuoc compute, memory, clock hay SLR crossing
   truoc khi doi tile/unroll.
9. Giu route-friendly target duoi khoang 70% resource moi SLR khi co the.
10. Accuracy test bat buoc khi thay softmax exp, GELU, sqrt/rsqrt hoac datatype.

## 14. Quyet dinh con mo

```text
[ ] QKV la 1 fused kernel hay 3 kernel?
[ ] Q/K/V dung 3 stream hay 1 multiplexed stream?
[ ] Attention score tile theo head hay row?
[ ] Softmax expf hay LUT/approx sau baseline?
[ ] SLR1 -> SLR2 FIFO depth bao nhieu?
[ ] Attention output projection co fit cung FFN UP tren SLR2?
[ ] Khi nao them ffn_residual_stream SLR2 -> SLR3?
[ ] Final output stream tiep hay ghi DDR[0]?
```

Moi quyet dinh tren phai duoc chot bang csynth, full link, per-SLR utilization,
post-route timing va accuracy, khong bang latency estimate rieng le.

## 15. Source of truth trong repo

```text
hls_ffn_kernel_lab/bert_ffn_kernel_hls_packed.cpp
hls_ffn_kernel_lab/bert_ffn_kernel_v21.cpp
hls_ffn_kernel_lab/ffn_kernel_u250_v21.cfg
hls_ffn_kernel_lab/csynth_up_v20.rpt
hls_ffn_kernel_lab/csynth_down_v20.rpt
hls_ffn_kernel_lab/ffn_v19_postroute_diagnosis.md
hls_ffn_kernel_lab/ffn_v21_dotpipe.md
docs/README_slr_reference_notes.md
docs/2312_15159_spatial_llm_slr.md
docs/2409_13975_protea_transformer_tiling.md
docs/2501_09118_stream_hls_dataflow.md
docs/2509_13694_streamtensor_llm_streaming.md
docs/heteroflow_fpga2022_data_placement.md
docs/u250_ds962_resource_information.md
```
