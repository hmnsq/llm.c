/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <cublas_v2.h>
#അന്തര്ഭാവ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#എങ്കിൽ defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#അല്ലെങ്കിലെങ്കിൽ defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#അല്ലെങ്കിൽ // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
സ്ഥിരം size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ശൂന്യം* cublaslt_workspace = ശൂന്യ;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
ശൂന്യം cublasCheck(cublasStatus_t status, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി)
{
    എങ്കിൽ (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, വരി);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H