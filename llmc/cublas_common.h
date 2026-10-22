/*
cuBLAS related utils
*/
#که_نه_تعریف_شوی CUBLAS_COMMON_H
#تعریف_کړه CUBLAS_COMMON_H

#ضمن <stddef.h>
#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#إذا defined(ENABLE_FP32)
#تعریف_کړه CUBLAS_LOWP CUDA_R_32F
#وإلاإذا defined(ENABLE_FP16)
#تعریف_کړه CUBLAS_LOWP CUDA_R_16F
#وإلا // default to bfloat16
#تعریف_کړه CUBLAS_LOWP CUDA_R_16BF
#د_که_پای

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
فراغ cublasCheck(cublas_حالت_t حالت, ثابت حرف *file, صحيح کرښه)
{
    إذا (حالت != CUBLAS_حالت_SUCCESS) {
        چاپ_کړه("[cuBLAS ERROR]: %d %s %d\n", حالت, file, کرښه);
        وتل(EXIT_FAILURE);
    }
}
#تعریف_کړه cublasCheck(حالت) { cublasCheck((حالت), __FILE__, __LINE__); }

#د_که_پای // CUBLAS_COMMON_H