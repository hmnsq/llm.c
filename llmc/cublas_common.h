/*
cuBLAS related utils
*/
#اگر_غیر_تعریف_شدہ CUBLAS_COMMON_H
#تعریف_کرو CUBLAS_COMMON_H

#شامل <stddef.h>
#شامل <stdlib.h>
#شامل <stdio.h>
#شامل <cublas_v2.h>
#شامل <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#اگر defined(ENABLE_FP32)
#تعریف_کرو CUBLAS_LOWP CUDA_R_32F
#وگرنہ_اگر defined(ENABLE_FP16)
#تعریف_کرو CUBLAS_LOWP CUDA_R_16F
#ورنہ // default to bfloat16
#تعریف_کرو CUBLAS_LOWP CUDA_R_16BF
#اگر_ختم

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ثابت size_t cublaslt_workspace_سائز = 32 * 1024 * 1024;
خالی* cublaslt_workspace = تہی;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
خالی cublasCheck(cublas_حالت_t حالت_2, ثابت حرف *file, صحیح سطر)
{
    اگر (حالت_2 != CUBLAS_حالت_SUCCESS) {
        لکھو("[cuBLAS ERROR]: %d %s %d\n", حالت_2, file, سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}
#تعریف_کرو cublasCheck(حالت_2) { cublasCheck((حالت_2), __FILE__, __LINE__); }

#اگر_ختم // CUBLAS_COMMON_H