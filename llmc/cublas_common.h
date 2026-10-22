/*
cuBLAS related utils
*/
#अगर_अघोषित CUBLAS_COMMON_H
#घोषणा CUBLAS_COMMON_H

#समावेश <stddef.h>
#समावेश <stdlib.h>
#समावेश <stdio.h>
#समावेश <cublas_v2.h>
#समावेश <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#अगर defined(ENABLE_FP32)
#घोषणा CUBLAS_LOWP CUDA_R_32F
#या_अगर defined(ENABLE_FP16)
#घोषणा CUBLAS_LOWP CUDA_R_16F
#अन्यथा // default to bfloat16
#घोषणा CUBLAS_LOWP CUDA_R_16BF
#पूर्ण_अगर

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
स्थिर माप_प्रकार cublaslt_workspace_आकार = 32 * 1024 * 1024;
व्योम* cublaslt_workspace = शून्य;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
व्योम cublasCheck(cublasStatus_t status, स्थिर अक्षर *file, पूर्णांक पंक्ति)
{
    अगर (status != CUBLAS_STATUS_SUCCESS) {
        म_लिखो("[cuBLAS ERROR]: %d %s %d\n", status, file, पंक्ति);
        निकास(निकास_त्रुटि);
    }
}
#घोषणा cublasCheck(status) { cublasCheck((status), __खाता__, __LINE__); }

#पूर्ण_अगर // CUBLAS_COMMON_H