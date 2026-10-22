/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#شامل <stddef.h>
#شامل <stdlib.h>
#شامل <stdio.h>
#شامل <cublas_v2.h>
#شامل <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#اگر defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#وگرنهاگر defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#وگرنه // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ثابت size_t cublaslt_workspace_size = 32 * 1024 * 1024;
تهی* cublaslt_workspace = پوچ;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
تهی cublasCheck(cublasStatus_t status, ثابت نویسه *file, صحیح سطر)
{
    اگر (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, سطر);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H