// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xbert_attn_out_norm_residual_kernel.h"

extern XBert_attn_out_norm_residual_kernel_Config XBert_attn_out_norm_residual_kernel_ConfigTable[];

XBert_attn_out_norm_residual_kernel_Config *XBert_attn_out_norm_residual_kernel_LookupConfig(u16 DeviceId) {
	XBert_attn_out_norm_residual_kernel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XBERT_ATTN_OUT_NORM_RESIDUAL_KERNEL_NUM_INSTANCES; Index++) {
		if (XBert_attn_out_norm_residual_kernel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XBert_attn_out_norm_residual_kernel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XBert_attn_out_norm_residual_kernel_Initialize(XBert_attn_out_norm_residual_kernel *InstancePtr, u16 DeviceId) {
	XBert_attn_out_norm_residual_kernel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XBert_attn_out_norm_residual_kernel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XBert_attn_out_norm_residual_kernel_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

