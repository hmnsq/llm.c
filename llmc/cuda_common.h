/*
Common utilities for CUDA code.
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ CUDA_COMMON_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDA_COMMON_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <type_traits>      // std::bool_constant
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_runtime.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <nvtx3/nvToolsExt.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <nvtx3/nvToolsExtCudaRt.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_profiler_api.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_bf16.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_fp16.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
ਬਾਹਰੀ cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WARP_ਮਾਪ 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#ਜੇ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_1024_ਧਾਗੇ_ਖੰਡਾਂ 2
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_1024_ਧਾਗੇ_ਖੰਡਾਂ 1
#ਜੇ_ਅੰਤ

// convenience macro for calculating grid/block dimensions for kernels
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
ਸਤਰਅੰਦਰ ਖਾਲੀ cudaCheck_(cudaError_t error, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
  ਜੇ (error != cudaSuccess) {
    ਲਿਖੋ("[CUDA ERROR] at file %s:%d:\n%s\n", file, ਸਤਰ, cudaGetErrorString(error));
    ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
  }
};
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
ਸਾਂਚਾ<class T>
ਸਤਰਅੰਦਰ ਖਾਲੀ cudaFreeCheck(T** ptr, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    cudaError_t error = cudaFree(*ptr);
    ਜੇ (error != cudaSuccess) {
        ਲਿਖੋ("[CUDA ERROR] at file %s:%d:\n%s\n", file, ਸਤਰ, cudaGetErrorString(error));
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

ਗਿਣਤੀ Precision_ਬਹੁਲਕ {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#ਜੇ defined(ENABLE_FP32)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ floatX;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PRECISION_ਬਹੁਲਕ PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#ਜਾਂਜੇ defined(ENABLE_FP16)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ half floatX;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PRECISION_ਬਹੁਲਕ PRECISION_FP16
#ਨਹੀਂਤਾਂ // Default to bfloat16
ਕਿਸਮਪਰਿਭਾਸ਼ਾ __nv_bfloat16 floatX;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PRECISION_ਬਹੁਲਕ PRECISION_BF16
#ਜੇ_ਅੰਤ

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#ਜੇ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ਅਚਲ floatX* ਪਤਾ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ bf = __ldcs(reinterpret_cast<ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ*>(ਪਤਾ));
    ਵਾਪਸ __nv_bfloat16_raw{bf};
}

__device__ ਖਾਲੀ __stcs(floatX* ਪਤਾ, floatX ਮੁੱਲ) {
    __stcs(reinterpret_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ*>(ਪਤਾ), ((__nv_bfloat16_raw)ਮੁੱਲ).x);
}
#ਜੇ_ਅੰਤ

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 ਜਨਤਕ:
    NvtxRange(ਅਚਲ ਅੱਖਰ* s) { nvtx_Range_ਧੱਕੋ(s); }
    NvtxRange(ਅਚਲ std::string& ਅਧਾਰ_str, ਪੂਰਨ ਸੰਖਿਆ) {
        std::string range_string = ਅਧਾਰ_str + " " + std::ਮੰਜ਼ਿਲ_string(ਸੰਖਿਆ);
        nvtx_Range_ਧੱਕੋ(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_ਕੱਢੋ(); }
};
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
ਸਤਰਅੰਦਰ ਖਾਲੀ device_ਮੰਜ਼ਿਲ_file(FILE* dest, ਖਾਲੀ* src, size_t ਅੰਕ_ਬਾਈਟਾਂ, size_t ਬਫਰ_ਮਾਪ, cuda_ਧਾਰਾ_t ਧਾਰਾ) {
    // allocate pinned buffer for faster, async transfer
    ਅੱਖਰ* ਬਫਰ_space;
    cudaCheck(cudaMallocHost(&ਬਫਰ_space, 2*ਬਫਰ_ਮਾਪ));
    // split allocation in two
    ਖਾਲੀ* read_ਬਫਰ = ਬਫਰ_space;
    ਖਾਲੀ* write_ਬਫਰ = ਬਫਰ_space + ਬਫਰ_ਮਾਪ;

    // prime the read buffer; first copy means we have to wait
    ਅੱਖਰ* gpu_read_ptr = (ਅੱਖਰ*)src;
    size_t copy_amount = std::min(ਬਫਰ_ਮਾਪ, ਅੰਕ_ਬਾਈਟਾਂ);
    cudaCheck(cudaMemcpyAsync(read_ਬਫਰ, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_ਮੰਜ਼ਿਲ_Host, ਧਾਰਾ));
    cudaCheck(cuda_ਧਾਰਾ_Synchronize(ਧਾਰਾ));
    size_t rest_ਬਾਈਟਾਂ = ਅੰਕ_ਬਾਈਟਾਂ - copy_amount;
    size_t write_ਬਫਰ_ਮਾਪ = copy_amount;
    gpu_read_ptr += copy_amount;

    std::ਅਦਲਾ_ਬਦਲੀ(read_ਬਫਰ, write_ਬਫਰ);
    // now the main loop; as long as there are bytes left
    ਜਦੋਂ(rest_ਬਾਈਟਾਂ > 0) {
        // initiate next read
        copy_amount = std::min(ਬਫਰ_ਮਾਪ, rest_ਬਾਈਟਾਂ);
        cudaCheck(cudaMemcpyAsync(read_ਬਫਰ, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_ਮੰਜ਼ਿਲ_Host, ਧਾਰਾ));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_ਬਫਰ, 1, write_ਬਫਰ_ਮਾਪ, dest);
        cudaCheck(cuda_ਧਾਰਾ_Synchronize(ਧਾਰਾ));     // wait for both buffers to be ready.

        std::ਅਦਲਾ_ਬਦਲੀ(read_ਬਫਰ, write_ਬਫਰ);
        rest_ਬਾਈਟਾਂ -= copy_amount;
        write_ਬਫਰ_ਮਾਪ = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_ਬਫਰ, 1, write_ਬਫਰ_ਮਾਪ, dest);
    cudaCheck(cudaFreeHost(ਬਫਰ_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
ਸਤਰਅੰਦਰ ਖਾਲੀ file_ਮੰਜ਼ਿਲ_device(ਖਾਲੀ* dest, FILE* src, size_t ਅੰਕ_ਬਾਈਟਾਂ, size_t ਬਫਰ_ਮਾਪ, cuda_ਧਾਰਾ_t ਧਾਰਾ) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    ਅੱਖਰ* ਬਫਰ_space;
    cudaCheck(cudaMallocHost(&ਬਫਰ_space, 2*ਬਫਰ_ਮਾਪ, cudaHostAllocWriteCombined));
    // split allocation in two
    ਖਾਲੀ* read_ਬਫਰ = ਬਫਰ_space;
    ਖਾਲੀ* write_ਬਫਰ = ਬਫਰ_space + ਬਫਰ_ਮਾਪ;

    // prime the read buffer;
    ਅੱਖਰ* gpu_write_ptr = (ਅੱਖਰ*)dest;
    size_t copy_amount = std::min(ਬਫਰ_ਮਾਪ, ਅੰਕ_ਬਾਈਟਾਂ);
    freadCheck(read_ਬਫਰ, 1, copy_amount, src);

    size_t rest_ਬਾਈਟਾਂ = ਅੰਕ_ਬਾਈਟਾਂ - copy_amount;
    size_t write_ਬਫਰ_ਮਾਪ = copy_amount;
    std::ਅਦਲਾ_ਬਦਲੀ(read_ਬਫਰ, write_ਬਫਰ);

    // now the main loop; as long as there are bytes left
    ਜਦੋਂ(rest_ਬਾਈਟਾਂ > 0) {
        // initiate next read
        copy_amount = std::min(ਬਫਰ_ਮਾਪ, rest_ਬਾਈਟਾਂ);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_ਬਫਰ, write_ਬਫਰ_ਮਾਪ, cuda_Memcpy_Host_ਮੰਜ਼ਿਲ_Device, ਧਾਰਾ));
        gpu_write_ptr += write_ਬਫਰ_ਮਾਪ;
        // while this is going on, read from disk
        freadCheck(read_ਬਫਰ, 1, copy_amount, src);
        cudaCheck(cuda_ਧਾਰਾ_Synchronize(ਧਾਰਾ));     // wait for both buffers to be ready.

        std::ਅਦਲਾ_ਬਦਲੀ(read_ਬਫਰ, write_ਬਫਰ);
        rest_ਬਾਈਟਾਂ -= copy_amount;
        write_ਬਫਰ_ਮਾਪ = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_ਬਫਰ, write_ਬਫਰ_ਮਾਪ, cuda_Memcpy_Host_ਮੰਜ਼ਿਲ_Device, ਧਾਰਾ));
    cudaCheck(cuda_ਧਾਰਾ_Synchronize(ਧਾਰਾ));
    cudaCheck(cudaFreeHost(ਬਫਰ_space));
}

#ਜੇ_ਅੰਤ // CUDA_COMMON_H