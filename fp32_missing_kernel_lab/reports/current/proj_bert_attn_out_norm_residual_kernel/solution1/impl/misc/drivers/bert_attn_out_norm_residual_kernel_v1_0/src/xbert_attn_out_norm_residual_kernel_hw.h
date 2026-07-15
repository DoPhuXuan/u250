// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/COR)
//        bit 0 - ap_done (Read/COR)
//        bit 1 - ap_ready (Read/COR)
//        others - reserved
// 0x10 : Data signal of residual_in
//        bit 31~0 - residual_in[31:0] (Read/Write)
// 0x14 : Data signal of residual_in
//        bit 31~0 - residual_in[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of residual_ready
//        bit 31~0 - residual_ready[31:0] (Read/Write)
// 0x20 : Data signal of residual_ready
//        bit 31~0 - residual_ready[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of attn_o_w_all
//        bit 31~0 - attn_o_w_all[31:0] (Read/Write)
// 0x2c : Data signal of attn_o_w_all
//        bit 31~0 - attn_o_w_all[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of attn_o_b_all
//        bit 31~0 - attn_o_b_all[31:0] (Read/Write)
// 0x38 : Data signal of attn_o_b_all
//        bit 31~0 - attn_o_b_all[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of attn_norm_gamma_all
//        bit 31~0 - attn_norm_gamma_all[31:0] (Read/Write)
// 0x44 : Data signal of attn_norm_gamma_all
//        bit 31~0 - attn_norm_gamma_all[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of attn_norm_beta_all
//        bit 31~0 - attn_norm_beta_all[31:0] (Read/Write)
// 0x50 : Data signal of attn_norm_beta_all
//        bit 31~0 - attn_norm_beta_all[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of attn_mid_ddr
//        bit 31~0 - attn_mid_ddr[31:0] (Read/Write)
// 0x5c : Data signal of attn_mid_ddr
//        bit 31~0 - attn_mid_ddr[63:32] (Read/Write)
// 0x60 : reserved
// 0x64 : Data signal of attn_mid_done
//        bit 31~0 - attn_mid_done[31:0] (Read/Write)
// 0x68 : Data signal of attn_mid_done
//        bit 31~0 - attn_mid_done[63:32] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of layer_id
//        bit 31~0 - layer_id[31:0] (Read/Write)
// 0x74 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL                  0x00
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_GIE                      0x04
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_IER                      0x08
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ISR                      0x0c
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_IN_DATA         0x10
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_RESIDUAL_IN_DATA         64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_READY_DATA      0x1c
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_RESIDUAL_READY_DATA      64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_W_ALL_DATA        0x28
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_ATTN_O_W_ALL_DATA        64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_B_ALL_DATA        0x34
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_ATTN_O_B_ALL_DATA        64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_GAMMA_ALL_DATA 0x40
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_ATTN_NORM_GAMMA_ALL_DATA 64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_BETA_ALL_DATA  0x4c
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_ATTN_NORM_BETA_ALL_DATA  64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DDR_DATA        0x58
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_ATTN_MID_DDR_DATA        64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DONE_DATA       0x64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_ATTN_MID_DONE_DATA       64
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_LAYER_ID_DATA            0x70
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_BITS_LAYER_ID_DATA            32

