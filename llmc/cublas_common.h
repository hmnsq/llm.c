/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <cublas_v2.h>
#ଅନ୍ତର୍ଭାଵ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#ଯଦି defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#ନଚେତଯଦି defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#ଅନ୍ୟଥା // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ଅଚଳ size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ଶୂନ୍ୟ* cublaslt_workspace = ଖାଲି;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
ଶୂନ୍ୟ cublasCheck(cublasStatus_t status, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି)
{
    ଯଦି (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, ଧାଡ଼ି);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H