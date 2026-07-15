// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_H
#define XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xbert_attn_out_norm_residual_kernel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XBert_attn_out_norm_residual_kernel_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XBert_attn_out_norm_residual_kernel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XBert_attn_out_norm_residual_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XBert_attn_out_norm_residual_kernel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XBert_attn_out_norm_residual_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XBert_attn_out_norm_residual_kernel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XBert_attn_out_norm_residual_kernel_Initialize(XBert_attn_out_norm_residual_kernel *InstancePtr, u16 DeviceId);
XBert_attn_out_norm_residual_kernel_Config* XBert_attn_out_norm_residual_kernel_LookupConfig(u16 DeviceId);
int XBert_attn_out_norm_residual_kernel_CfgInitialize(XBert_attn_out_norm_residual_kernel *InstancePtr, XBert_attn_out_norm_residual_kernel_Config *ConfigPtr);
#else
int XBert_attn_out_norm_residual_kernel_Initialize(XBert_attn_out_norm_residual_kernel *InstancePtr, const char* InstanceName);
int XBert_attn_out_norm_residual_kernel_Release(XBert_attn_out_norm_residual_kernel *InstancePtr);
#endif

void XBert_attn_out_norm_residual_kernel_Start(XBert_attn_out_norm_residual_kernel *InstancePtr);
u32 XBert_attn_out_norm_residual_kernel_IsDone(XBert_attn_out_norm_residual_kernel *InstancePtr);
u32 XBert_attn_out_norm_residual_kernel_IsIdle(XBert_attn_out_norm_residual_kernel *InstancePtr);
u32 XBert_attn_out_norm_residual_kernel_IsReady(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_EnableAutoRestart(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_DisableAutoRestart(XBert_attn_out_norm_residual_kernel *InstancePtr);

void XBert_attn_out_norm_residual_kernel_Set_residual_in(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_residual_in(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_residual_ready(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_residual_ready(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_attn_o_w_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_attn_o_w_all(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_attn_o_b_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_attn_o_b_all(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_attn_norm_gamma_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_attn_norm_gamma_all(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_attn_norm_beta_all(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_attn_norm_beta_all(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_attn_mid_ddr(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_attn_mid_ddr(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_attn_mid_done(XBert_attn_out_norm_residual_kernel *InstancePtr, u64 Data);
u64 XBert_attn_out_norm_residual_kernel_Get_attn_mid_done(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_Set_layer_id(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Data);
u32 XBert_attn_out_norm_residual_kernel_Get_layer_id(XBert_attn_out_norm_residual_kernel *InstancePtr);

void XBert_attn_out_norm_residual_kernel_InterruptGlobalEnable(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_InterruptGlobalDisable(XBert_attn_out_norm_residual_kernel *InstancePtr);
void XBert_attn_out_norm_residual_kernel_InterruptEnable(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Mask);
void XBert_attn_out_norm_residual_kernel_InterruptDisable(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Mask);
void XBert_attn_out_norm_residual_kernel_InterruptClear(XBert_attn_out_norm_residual_kernel *InstancePtr, u32 Mask);
u32 XBert_attn_out_norm_residual_kernel_InterruptGetEnabled(XBert_attn_out_norm_residual_kernel *InstancePtr);
u32 XBert_attn_out_norm_residual_kernel_InterruptGetStatus(XBert_attn_out_norm_residual_kernel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
