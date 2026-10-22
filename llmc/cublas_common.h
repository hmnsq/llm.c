/*
cuBLAS related utils
*/
#جيڪڏهن_وصف_نه_ٿيل CUBLAS_COMMON_H
#وصف_ڪريو CUBLAS_COMMON_H

#ضمن <stddef.h>
#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#إذا defined(ENABLE_FP32)
#وصف_ڪريو CUBLAS_LOWP CUDA_R_32F
#وإلاإذا defined(ENABLE_FP16)
#وصف_ڪريو CUBLAS_LOWP CUDA_R_16F
#وإلا // default to bfloat16
#وصف_ڪريو CUBLAS_LOWP CUDA_R_16BF
#جيڪڏهن_پورو

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ثابت size_t cublaslt_workspace_سائيز = 32 * 1024 * 1024;
فراغ* cublaslt_workspace = عدم;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
فراغ cublasCheck(cublas_حالت_t حالت, ثابت حرف *file, صحيح سٽ)
{
    إذا (حالت != CUBLAS_حالت_SUCCESS) {
        لکو("[cuBLAS ERROR]: %d %s %d\n", حالت, file, سٽ);
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}
#وصف_ڪريو cublasCheck(حالت) { cublasCheck((حالت), __FILE__, __LINE__); }

#جيڪڏهن_پورو // CUBLAS_COMMON_H