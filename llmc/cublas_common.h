/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#ضمن <stddef.h>
#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#𐤀𐤌 defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#وإلاإذا defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#وإلا // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ثابت size_t cublaslt_workspace_size = 32 * 1024 * 1024;
فراغ* cublaslt_workspace = عدم;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
فراغ cublasCheck(cublasStatus_t status, ثابت حرف *file, صحيح line)
{
    𐤀𐤌 (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H