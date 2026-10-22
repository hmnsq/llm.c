/*
cuBLAS related utils
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ CUBLAS_COMMON_H
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMMON_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <cublas_v2.h>
#ಅನ್ತರ್ಭಾವ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#ಆದರೆ defined(ENABLE_FP32)
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_LOWP CUDA_R_32F
#ಅಥವಾಆದರೆ defined(ENABLE_FP16)
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_LOWP CUDA_R_16F
#ಇಲ್ಲದಿದ್ದರೆ // default to bfloat16
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_LOWP CUDA_R_16BF
#ಆದರೆ_ಅಂತ್ಯ

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
ಸ್ಥಿರಾಂಕ size_t cublaslt_workspace_ಗಾತ್ರ = 32 * 1024 * 1024;
ಶೂನ್ಯ* cublaslt_workspace = ಖಾಲಿ;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
ಶೂನ್ಯ cublasCheck(cublas_ಸ್ಥಿತಿ_t ಸ್ಥಿತಿ_2, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು)
{
    ಆದರೆ (ಸ್ಥಿತಿ_2 != CUBLAS_ಸ್ಥಿತಿ_SUCCESS) {
        ಬರೆ("[cuBLAS ERROR]: %d %s %d\n", ಸ್ಥಿತಿ_2, file, ಸಾಲು);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}
#ವ್ಯಾಖ್ಯಾನಿಸು cublasCheck(ಸ್ಥಿತಿ_2) { cublasCheck((ಸ್ಥಿತಿ_2), __FILE__, __LINE__); }

#ಆದರೆ_ಅಂತ್ಯ // CUBLAS_COMMON_H