/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <cublas_v2.h>
#अन्तर्भाव <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#यदि defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#नोचेद्यदि defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#अन्यथा // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ध्रुव माप_प्रकार cublaslt_workspace_size = 32 * 1024 * 1024;
शून्य* cublaslt_workspace = रिक्त;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
शून्य cublasCheck(cublasStatus_t status, ध्रुव वर्ण *file, पूर्णाङ्क पंक्ति)
{
    यदि (status != CUBLAS_STATUS_SUCCESS) {
        म_लिखो("[cuBLAS ERROR]: %d %s %d\n", status, file, पंक्ति);
        निकास(निकास_त्रुटि);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H