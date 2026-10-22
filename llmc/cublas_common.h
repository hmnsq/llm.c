/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <cublas_v2.h>
#ಅನ್ತರ್ಭಾವ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#ಯದಿ defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#ನೋಚೇದ್ಯದಿ defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#ಅನ್ಯಥಾ // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ಧ್ರುವ size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ಶೂನ್ಯ* cublaslt_workspace = ರಿಕ್ತ;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
ಶೂನ್ಯ cublasCheck(cublasStatus_t status, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು)
{
    ಯದಿ (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, ಸಾಲು);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H