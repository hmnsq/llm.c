/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#অন্তর্ভাৱ <stddef.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <cublas_v2.h>
#অন্তর্ভাৱ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#যদি defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#নোচেদ্যদি defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#অন্যথা // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ধ্রুৱ size_t cublaslt_workspace_size = 32 * 1024 * 1024;
শূন্য* cublaslt_workspace = রিক্ত;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
শূন্য cublasCheck(cublasStatus_t status, ধ্রুৱ ৱর্ণ *file, পূর্ণাঙ্ক সাৰি)
{
    যদি (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, সাৰি);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H