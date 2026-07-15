// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xbert_attn_out_norm_residual_kernel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XBert_attn_out_norm_residual_kernel_CfgInitialize(XBert_attn_out_norm_residual_kernel *InstancePtr, XBert_attn_out_norm_residual_kernel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XBert_attn_out_norm_residual_kernel_Start(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XBert_attn_out_norm_residual_kernel_IsDone(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XBert_attn_out_norm_residual_kernel_IsIdle(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XBert_attn_out_norm_residual_kernel_IsReady(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XBert_attn_out_norm_residual_kernel_EnableAutoRestart(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XBert_attn_out_norm_residual_kernel_DisableAutoRestart(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_AP_CTRL, 0);
}

void XBert_attn_out_norm_residual_kernel_Set_residual_in(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_IN_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_IN_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_residual_in(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_IN_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_IN_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_residual_ready(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_READY_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_READY_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_residual_ready(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_READY_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_RESIDUAL_READY_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_attn_o_w_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_W_ALL_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_W_ALL_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_attn_o_w_all(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_W_ALL_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_W_ALL_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_attn_o_b_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_B_ALL_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_B_ALL_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_attn_o_b_all(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_B_ALL_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_O_B_ALL_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_attn_norm_gamma_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_GAMMA_ALL_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_GAMMA_ALL_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_attn_norm_gamma_all(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_GAMMA_ALL_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_GAMMA_ALL_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_attn_norm_beta_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_BETA_ALL_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_BETA_ALL_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_attn_norm_beta_all(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_BETA_ALL_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_NORM_BETA_ALL_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_attn_mid_ddr(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DDR_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DDR_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_attn_mid_ddr(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DDR_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DDR_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_attn_mid_done(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DONE_DATA, (u32)(Data));
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DONE_DATA + 4, (u32)(Data >> 32));
}

u64 XBert_attn_out_norm_residual_kernel_Get_attn_mid_done(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DONE_DATA);
    Data += (u64)XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ATTN_MID_DONE_DATA + 4) << 32;
    return Data;
}

void XBert_attn_out_norm_residual_kernel_Set_layer_id(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_LAYER_ID_DATA, Data);
}

u32 XBert_attn_out_norm_residual_kernel_Get_layer_id(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_LAYER_ID_DATA);
    return Data;
}

void XBert_attn_out_norm_residual_kernel_InterruptGlobalEnable(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_GIE, 1);
}

void XBert_attn_out_norm_residual_kernel_InterruptGlobalDisable(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_GIE, 0);
}

void XBert_attn_out_norm_residual_kernel_InterruptEnable(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_IER);
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_IER, Register | Mask);
}

void XBert_attn_out_norm_residual_kernel_InterruptDisable(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_IER);
    XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XBert_attn_out_norm_residual_kernel_InterruptClear(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    //XBert_attn_out_norm_residual_kernel_WriteReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ISR, Mask);
}

u32 XBert_attn_out_norm_residual_kernel_InterruptGetEnabled(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_IER);
}

u32 XBert_attn_out_norm_residual_kernel_InterruptGetStatus(XBert_attn_out_norm_residual_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    // Current Interrupt Clear Behavior is Clear on Read(COR).
    return XBert_attn_out_norm_residual_kernel_ReadReg(InstancePtr->Control_BaseAddress, XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_CONTROL_ADDR_ISR);
}

