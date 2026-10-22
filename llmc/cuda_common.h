/*
Common utilities for CUDA code.
*/
#اگر_غیر_تعریف_شدہ CUDA_COMMON_H
#تعریف_کرو CUDA_COMMON_H

#شامل <stdlib.h>
#شامل <stdio.h>
#شامل <math.h>
#شامل <string>
#شامل <type_traits>      // std::bool_constant
#شامل <cuda_runtime.h>
#شامل <nvtx3/nvToolsExt.h>
#شامل <nvtx3/nvToolsExtCudaRt.h>
#شامل <cuda_profiler_api.h>
#شامل <cuda_bf16.h>
#شامل <cuda_fp16.h>

#شامل "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
خارجی cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#تعریف_کرو WARP_سائز 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#اگر __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#تعریف_کرو MAX_1024_دھاگے_بلاکیں 2
#ورنہ
#تعریف_کرو MAX_1024_دھاگے_بلاکیں 1
#اگر_ختم

// convenience macro for calculating grid/block dimensions for kernels
#تعریف_کرو CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
اندرونی خالی cudaCheck_(cudaError_t error, ثابت حرف *file, صحیح سطر) {
  اگر (error != cudaSuccess) {
    لکھو("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر, cudaGetErrorString(error));
    باہر_نکلو(EXIT_FAILURE);
  }
};
#تعریف_کرو cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
سانچہ<class T>
اندرونی خالی cudaFreeCheck(T** ptr, ثابت حرف *file, صحیح سطر) {
    cudaError_t error = cudaFree(*ptr);
    اگر (error != cudaSuccess) {
        لکھو("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر, cudaGetErrorString(error));
        باہر_نکلو(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#تعریف_کرو cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

شمار Precision_منوال {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#اگر defined(ENABLE_FP32)
قسمتعریف اعشاری floatX;
#تعریف_کرو PRECISION_منوال PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وگرنہ_اگر defined(ENABLE_FP16)
قسمتعریف half floatX;
#تعریف_کرو PRECISION_منوال PRECISION_FP16
#ورنہ // Default to bfloat16
قسمتعریف __nv_bfloat16 floatX;
#تعریف_کرو PRECISION_منوال PRECISION_BF16
#اگر_ختم

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#اگر defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* پتہ) {
    غیرعلامتی مختصر bf = __ldcs(reinterpret_cast<ثابت غیرعلامتی مختصر*>(پتہ));
    واپس __nv_bfloat16_raw{bf};
}

__device__ خالی __stcs(floatX* پتہ, floatX قدر) {
    __stcs(reinterpret_cast<غیرعلامتی مختصر*>(پتہ), ((__nv_bfloat16_raw)قدر).x);
}
#اگر_ختم

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 عوامی:
    NvtxRange(ثابت حرف* s) { nvtx_Range_ڈالو(s); }
    NvtxRange(ثابت std::string& بنیاد_str, صحیح نمبر) {
        std::string range_string = بنیاد_str + " " + std::مقصد_string(نمبر);
        nvtx_Range_ڈالو(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_نکالو(); }
};
#تعریف_کرو NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
اندرونی خالی device_مقصد_file(FILE* dest, خالی* src, size_t عدد_بائٹیں, size_t بفر_سائز, cuda_دھارا_t دھارا) {
    // allocate pinned buffer for faster, async transfer
    حرف* بفر_space;
    cudaCheck(cudaMallocHost(&بفر_space, 2*بفر_سائز));
    // split allocation in two
    خالی* read_بفر = بفر_space;
    خالی* write_بفر = بفر_space + بفر_سائز;

    // prime the read buffer; first copy means we have to wait
    حرف* gpu_read_ptr = (حرف*)src;
    size_t copy_amount = std::min(بفر_سائز, عدد_بائٹیں);
    cudaCheck(cudaMemcpyAsync(read_بفر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_مقصد_Host, دھارا));
    cudaCheck(cuda_دھارا_Synchronize(دھارا));
    size_t rest_بائٹیں = عدد_بائٹیں - copy_amount;
    size_t write_بفر_سائز = copy_amount;
    gpu_read_ptr += copy_amount;

    std::تبادلہ(read_بفر, write_بفر);
    // now the main loop; as long as there are bytes left
    جب(rest_بائٹیں > 0) {
        // initiate next read
        copy_amount = std::min(بفر_سائز, rest_بائٹیں);
        cudaCheck(cudaMemcpyAsync(read_بفر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_مقصد_Host, دھارا));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_بفر, 1, write_بفر_سائز, dest);
        cudaCheck(cuda_دھارا_Synchronize(دھارا));     // wait for both buffers to be ready.

        std::تبادلہ(read_بفر, write_بفر);
        rest_بائٹیں -= copy_amount;
        write_بفر_سائز = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_بفر, 1, write_بفر_سائز, dest);
    cudaCheck(cudaFreeHost(بفر_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
اندرونی خالی file_مقصد_device(خالی* dest, FILE* src, size_t عدد_بائٹیں, size_t بفر_سائز, cuda_دھارا_t دھارا) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    حرف* بفر_space;
    cudaCheck(cudaMallocHost(&بفر_space, 2*بفر_سائز, cudaHostAllocWriteCombined));
    // split allocation in two
    خالی* read_بفر = بفر_space;
    خالی* write_بفر = بفر_space + بفر_سائز;

    // prime the read buffer;
    حرف* gpu_write_ptr = (حرف*)dest;
    size_t copy_amount = std::min(بفر_سائز, عدد_بائٹیں);
    freadCheck(read_بفر, 1, copy_amount, src);

    size_t rest_بائٹیں = عدد_بائٹیں - copy_amount;
    size_t write_بفر_سائز = copy_amount;
    std::تبادلہ(read_بفر, write_بفر);

    // now the main loop; as long as there are bytes left
    جب(rest_بائٹیں > 0) {
        // initiate next read
        copy_amount = std::min(بفر_سائز, rest_بائٹیں);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_بفر, write_بفر_سائز, cuda_Memcpy_Host_مقصد_Device, دھارا));
        gpu_write_ptr += write_بفر_سائز;
        // while this is going on, read from disk
        freadCheck(read_بفر, 1, copy_amount, src);
        cudaCheck(cuda_دھارا_Synchronize(دھارا));     // wait for both buffers to be ready.

        std::تبادلہ(read_بفر, write_بفر);
        rest_بائٹیں -= copy_amount;
        write_بفر_سائز = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_بفر, write_بفر_سائز, cuda_Memcpy_Host_مقصد_Device, دھارا));
    cudaCheck(cuda_دھارا_Synchronize(دھارا));
    cudaCheck(cudaFreeHost(بفر_space));
}

#اگر_ختم // CUDA_COMMON_H