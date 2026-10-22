/*
cuBLAS related utils
*/
#جے_بنا_تعریف CUBLAS_COMMON_H
#تعریف_کرو CUBLAS_COMMON_H

#ضمن <stddef.h>
#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#إذا defined(ENABLE_FP32)
#تعریف_کرو CUBLAS_LOWP CUDA_R_32F
#وإلاإذا defined(ENABLE_FP16)
#تعریف_کرو CUBLAS_LOWP CUDA_R_16F
#وإلا // default to bfloat16
#تعریف_کرو CUBLAS_LOWP CUDA_R_16BF
#جے_انت

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ثابت size_t cublaslt_workspace_ناپ = 32 * 1024 * 1024;
فراغ* cublaslt_workspace = عدم;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
فراغ cublasCheck(cublas_حالت_t حالت, ثابت حرف *file, صحيح سطر)
{
    إذا (حالت != CUBLAS_حالت_SUCCESS) {
        لکھو("[cuBLAS ERROR]: %d %s %d\n", حالت, file, سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}
#تعریف_کرو cublasCheck(حالت) { cublasCheck((حالت), __FILE__, __LINE__); }

#جے_انت // CUBLAS_COMMON_H