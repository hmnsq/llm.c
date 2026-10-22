/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <cublas_v2.h>
#અન્તર્ભાવ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#જો defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#અથવાજો defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#નહીંતર // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
અચલ size_t cublaslt_workspace_size = 32 * 1024 * 1024;
શૂન્ય* cublaslt_workspace = ખાલી;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
શૂન્ય cublasCheck(cublasStatus_t status, અચલ અક્ષર *file, પૂર્ણાંક લીટી)
{
    જો (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, લીટી);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H