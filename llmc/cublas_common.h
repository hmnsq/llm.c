/*
cuBLAS related utils
*/
#اگر_تعریف_نشده CUBLAS_COMMON_H
#تعریف_کن CUBLAS_COMMON_H

#ضمن <stddef.h>
#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#إذا defined(ENABLE_FP32)
#تعریف_کن CUBLAS_LOWP CUDA_R_32F
#وإلاإذا defined(ENABLE_FP16)
#تعریف_کن CUBLAS_LOWP CUDA_R_16F
#وإلا // default to bfloat16
#تعریف_کن CUBLAS_LOWP CUDA_R_16BF
#پایان_اگر

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ثابت size_t cublaslt_workspace_اندازه = 32 * 1024 * 1024;
فراغ* cublaslt_workspace = عدم;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
فراغ cublasCheck(cublas_وضعیت_t وضعیت, ثابت حرف *file, صحيح سطر)
{
    إذا (وضعیت != CUBLAS_وضعیت_SUCCESS) {
        چاپ_کن("[cuBLAS ERROR]: %d %s %d\n", وضعیت, file, سطر);
        خروج(EXIT_FAILURE);
    }
}
#تعریف_کن cublasCheck(وضعیت) { cublasCheck((وضعیت), __FILE__, __LINE__); }

#پایان_اگر // CUBLAS_COMMON_H