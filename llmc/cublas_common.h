/*
cuBLAS related utils
*/
#నిర్వచితం_కాకపోతే CUBLAS_COMMON_H
#నిర్వచించు CUBLAS_COMMON_H

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <cublas_v2.h>
#అన్తర్భావ <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS Precision settings

#ఒకవేళ defined(ENABLE_FP32)
#నిర్వచించు CUBLAS_LOWP CUDA_R_32F
#లేదావేళ defined(ENABLE_FP16)
#నిర్వచించు CUBLAS_LOWP CUDA_R_16F
#లేకపోతే // default to bfloat16
#నిర్వచించు CUBLAS_LOWP CUDA_R_16BF
#అయితే_ముగింపు

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
స్థిరరాశి size_t cublaslt_workspace_కొలత = 32 * 1024 * 1024;
శూన్య* cublaslt_workspace = ఖాళీ;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
శూన్య cublasCheck(cublas_స్థితి_t స్థితి_2, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి)
{
    ఒకవేళ (స్థితి_2 != CUBLAS_స్థితి_SUCCESS) {
        రాయి("[cuBLAS ERROR]: %d %s %d\n", స్థితి_2, file, పంక్తి);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}
#నిర్వచించు cublasCheck(స్థితి_2) { cublasCheck((స్థితి_2), __FILE__, __LINE__); }

#అయితే_ముగింపు // CUBLAS_COMMON_H