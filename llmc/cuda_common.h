/*
Common utilities for CUDA code.
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ CUDA_COMMON_H
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_COMMON_H

#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <type_traits>      // std::bool_constant
#ಅನ್ತರ್ಭಾವ <cuda_runtime.h>
#ಅನ್ತರ್ಭಾವ <nvtx3/nvToolsExt.h>
#ಅನ್ತರ್ಭಾವ <nvtx3/nvToolsExtCudaRt.h>
#ಅನ್ತರ್ಭಾವ <cuda_profiler_api.h>
#ಅನ್ತರ್ಭಾವ <cuda_bf16.h>
#ಅನ್ತರ್ಭಾವ <cuda_fp16.h>

#ಅನ್ತರ್ಭಾವ "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
ಬಾಹ್ಯ cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#ವ್ಯಾಖ್ಯಾನಿಸು WARP_ಗಾತ್ರ 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#ಆದರೆ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_1024_ಎಳೆಗಳು_ಖಂಡಗಳು 2
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_1024_ಎಳೆಗಳು_ಖಂಡಗಳು 1
#ಆದರೆ_ಅಂತ್ಯ

// convenience macro for calculating grid/block dimensions for kernels
#ವ್ಯಾಖ್ಯಾನಿಸು CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
ಸಾಲೊಳಗೆ ಶೂನ್ಯ cudaCheck_(cudaError_t error, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
  ಆದರೆ (error != cudaSuccess) {
    ಬರೆ("[CUDA ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು, cudaGetErrorString(error));
    ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
  }
};
#ವ್ಯಾಖ್ಯಾನಿಸು cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
ಮಾದರಿ<class T>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ cudaFreeCheck(T** ptr, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    cudaError_t error = cudaFree(*ptr);
    ಆದರೆ (error != cudaSuccess) {
        ಬರೆ("[CUDA ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು, cudaGetErrorString(error));
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#ವ್ಯಾಖ್ಯಾನಿಸು cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

ಎಣಿಕೆ Precision_ಬಹುಲಕ {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#ಆದರೆ defined(ENABLE_FP32)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ floatX;
#ವ್ಯಾಖ್ಯಾನಿಸು PRECISION_ಬಹುಲಕ PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#ಅಥವಾಆದರೆ defined(ENABLE_FP16)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ half floatX;
#ವ್ಯಾಖ್ಯಾನಿಸು PRECISION_ಬಹುಲಕ PRECISION_FP16
#ಇಲ್ಲದಿದ್ದರೆ // Default to bfloat16
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __nv_bfloat16 floatX;
#ವ್ಯಾಖ್ಯಾನಿಸು PRECISION_ಬಹುಲಕ PRECISION_BF16
#ಆದರೆ_ಅಂತ್ಯ

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#ಆದರೆ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ಸ್ಥಿರಾಂಕ floatX* ವಿಳಾಸ) {
    ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ bf = __ldcs(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ*>(ವಿಳಾಸ));
    ಹಿಂತಿರುಗಿ __nv_bfloat16_raw{bf};
}

__device__ ಶೂನ್ಯ __stcs(floatX* ವಿಳಾಸ, floatX ಮೌಲ್ಯ) {
    __stcs(reinterpret_cast<ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ*>(ವಿಳಾಸ), ((__nv_bfloat16_raw)ಮೌಲ್ಯ).x);
}
#ಆದರೆ_ಅಂತ್ಯ

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 ಸಾರ್ವಜನಿಕ:
    NvtxRange(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* s) { nvtx_Range_ತಳ್ಳು(s); }
    NvtxRange(ಸ್ಥಿರಾಂಕ std::string& ತಳ_str, ಪೂರ್ಣಾಂಕ ಸಂಖ್ಯೆ) {
        std::string range_string = ತಳ_str + " " + std::ಗಮ್ಯ_string(ಸಂಖ್ಯೆ);
        nvtx_Range_ತಳ್ಳು(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_ತೆಗೆ(); }
};
#ವ್ಯಾಖ್ಯಾನಿಸು NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
ಸಾಲೊಳಗೆ ಶೂನ್ಯ device_ಗಮ್ಯ_file(FILE* dest, ಶೂನ್ಯ* src, size_t ಅಂಕೆ_ಬೈಟುಗಳು, size_t ಬಫರ್_ಗಾತ್ರ, cuda_ಹರಿವು_t ಹರಿವು) {
    // allocate pinned buffer for faster, async transfer
    ಅಕ್ಷರ* ಬಫರ್_space;
    cudaCheck(cudaMallocHost(&ಬಫರ್_space, 2*ಬಫರ್_ಗಾತ್ರ));
    // split allocation in two
    ಶೂನ್ಯ* read_ಬಫರ್ = ಬಫರ್_space;
    ಶೂನ್ಯ* write_ಬಫರ್ = ಬಫರ್_space + ಬಫರ್_ಗಾತ್ರ;

    // prime the read buffer; first copy means we have to wait
    ಅಕ್ಷರ* gpu_read_ptr = (ಅಕ್ಷರ*)src;
    size_t copy_amount = std::min(ಬಫರ್_ಗಾತ್ರ, ಅಂಕೆ_ಬೈಟುಗಳು);
    cudaCheck(cudaMemcpyAsync(read_ಬಫರ್, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_ಗಮ್ಯ_Host, ಹರಿವು));
    cudaCheck(cuda_ಹರಿವು_Synchronize(ಹರಿವು));
    size_t rest_ಬೈಟುಗಳು = ಅಂಕೆ_ಬೈಟುಗಳು - copy_amount;
    size_t write_ಬಫರ್_ಗಾತ್ರ = copy_amount;
    gpu_read_ptr += copy_amount;

    std::ಅದಲುಬದಲು(read_ಬಫರ್, write_ಬಫರ್);
    // now the main loop; as long as there are bytes left
    ಆಗ(rest_ಬೈಟುಗಳು > 0) {
        // initiate next read
        copy_amount = std::min(ಬಫರ್_ಗಾತ್ರ, rest_ಬೈಟುಗಳು);
        cudaCheck(cudaMemcpyAsync(read_ಬಫರ್, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_ಗಮ್ಯ_Host, ಹರಿವು));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_ಬಫರ್, 1, write_ಬಫರ್_ಗಾತ್ರ, dest);
        cudaCheck(cuda_ಹರಿವು_Synchronize(ಹರಿವು));     // wait for both buffers to be ready.

        std::ಅದಲುಬದಲು(read_ಬಫರ್, write_ಬಫರ್);
        rest_ಬೈಟುಗಳು -= copy_amount;
        write_ಬಫರ್_ಗಾತ್ರ = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_ಬಫರ್, 1, write_ಬಫರ್_ಗಾತ್ರ, dest);
    cudaCheck(cudaFreeHost(ಬಫರ್_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
ಸಾಲೊಳಗೆ ಶೂನ್ಯ file_ಗಮ್ಯ_device(ಶೂನ್ಯ* dest, FILE* src, size_t ಅಂಕೆ_ಬೈಟುಗಳು, size_t ಬಫರ್_ಗಾತ್ರ, cuda_ಹರಿವು_t ಹರಿವು) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    ಅಕ್ಷರ* ಬಫರ್_space;
    cudaCheck(cudaMallocHost(&ಬಫರ್_space, 2*ಬಫರ್_ಗಾತ್ರ, cudaHostAllocWriteCombined));
    // split allocation in two
    ಶೂನ್ಯ* read_ಬಫರ್ = ಬಫರ್_space;
    ಶೂನ್ಯ* write_ಬಫರ್ = ಬಫರ್_space + ಬಫರ್_ಗಾತ್ರ;

    // prime the read buffer;
    ಅಕ್ಷರ* gpu_write_ptr = (ಅಕ್ಷರ*)dest;
    size_t copy_amount = std::min(ಬಫರ್_ಗಾತ್ರ, ಅಂಕೆ_ಬೈಟುಗಳು);
    freadCheck(read_ಬಫರ್, 1, copy_amount, src);

    size_t rest_ಬೈಟುಗಳು = ಅಂಕೆ_ಬೈಟುಗಳು - copy_amount;
    size_t write_ಬಫರ್_ಗಾತ್ರ = copy_amount;
    std::ಅದಲುಬದಲು(read_ಬಫರ್, write_ಬಫರ್);

    // now the main loop; as long as there are bytes left
    ಆಗ(rest_ಬೈಟುಗಳು > 0) {
        // initiate next read
        copy_amount = std::min(ಬಫರ್_ಗಾತ್ರ, rest_ಬೈಟುಗಳು);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_ಬಫರ್, write_ಬಫರ್_ಗಾತ್ರ, cuda_Memcpy_Host_ಗಮ್ಯ_Device, ಹರಿವು));
        gpu_write_ptr += write_ಬಫರ್_ಗಾತ್ರ;
        // while this is going on, read from disk
        freadCheck(read_ಬಫರ್, 1, copy_amount, src);
        cudaCheck(cuda_ಹರಿವು_Synchronize(ಹರಿವು));     // wait for both buffers to be ready.

        std::ಅದಲುಬದಲು(read_ಬಫರ್, write_ಬಫರ್);
        rest_ಬೈಟುಗಳು -= copy_amount;
        write_ಬಫರ್_ಗಾತ್ರ = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_ಬಫರ್, write_ಬಫರ್_ಗಾತ್ರ, cuda_Memcpy_Host_ಗಮ್ಯ_Device, ಹರಿವು));
    cudaCheck(cuda_ಹರಿವು_Synchronize(ಹರಿವು));
    cudaCheck(cudaFreeHost(ಬಫರ್_space));
}

#ಆದರೆ_ಅಂತ್ಯ // CUDA_COMMON_H