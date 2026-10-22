/*
cuBLAS related utils
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ CUBLAS_COMMON_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMMON_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cublas_v2.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#ਜੇ defined(ENABLE_FP32)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_LOWP CUDA_R_32F
#ਜਾਂਜੇ defined(ENABLE_FP16)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_LOWP CUDA_R_16F
#ਨਹੀਂਤਾਂ // default to bfloat16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_LOWP CUDA_R_16BF
#ਜੇ_ਅੰਤ

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ਅਚਲ size_t cublaslt_workspace_ਮਾਪ = 32 * 1024 * 1024;
ਖਾਲੀ* cublaslt_workspace = ਨਿਰਾ;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
ਖਾਲੀ cublasCheck(cublas_ਸਥਿਤੀ_t ਸਥਿਤੀ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ)
{
    ਜੇ (ਸਥਿਤੀ != CUBLAS_ਸਥਿਤੀ_SUCCESS) {
        ਲਿਖੋ("[cuBLAS ERROR]: %d %s %d\n", ਸਥਿਤੀ, file, ਸਤਰ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasCheck(ਸਥਿਤੀ) { cublasCheck((ਸਥਿਤੀ), __FILE__, __LINE__); }

#ਜੇ_ਅੰਤ // CUBLAS_COMMON_H