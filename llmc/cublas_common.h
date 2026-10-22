/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <cublas_v2.h>
#அந்தர்பாவ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#எனில் defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#அல்லதெனில் defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#இல்லையெனில் // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
மாறிலி size_t cublaslt_workspace_size = 32 * 1024 * 1024;
வெறுமை* cublaslt_workspace = வெற்று;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
வெறுமை cublasCheck(cublasStatus_t status, மாறிலி எழுத்து *file, முழு வரி)
{
    எனில் (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, வரி);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H