/*
Common utilities for CUDA code.
*/
#که_نه_تعریف_شوی CUDA_COMMON_H
#تعریف_کړه CUDA_COMMON_H

#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <math.h>
#ضمن <string>
#ضمن <type_traits>      // std::bool_constant
#ضمن <cuda_runtime.h>
#ضمن <nvtx3/nvToolsExt.h>
#ضمن <nvtx3/nvToolsExtCudaRt.h>
#ضمن <cuda_profiler_api.h>
#ضمن <cuda_bf16.h>
#ضمن <cuda_fp16.h>

#ضمن "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
خارجي cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#تعریف_کړه WARP_اندازه 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#إذا __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#تعریف_کړه MAX_1024_تارونه_بلاکونه 2
#وإلا
#تعریف_کړه MAX_1024_تارونه_بلاکونه 1
#د_که_پای

// convenience macro for calculating grid/block dimensions for kernels
#تعریف_کړه CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
مضمن فراغ cudaCheck_(cudaError_t error, ثابت حرف *file, صحيح کرښه) {
  إذا (error != cudaSuccess) {
    چاپ_کړه("[CUDA ERROR] at file %s:%d:\n%s\n", file, کرښه, cudaGetErrorString(error));
    وتل(EXIT_FAILURE);
  }
};
#تعریف_کړه cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
کينډۍ<class T>
مضمن فراغ cudaFreeCheck(T** ptr, ثابت حرف *file, صحيح کرښه) {
    cudaError_t error = cudaFree(*ptr);
    إذا (error != cudaSuccess) {
        چاپ_کړه("[CUDA ERROR] at file %s:%d:\n%s\n", file, کرښه, cudaGetErrorString(error));
        وتل(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#تعریف_کړه cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

تعداد Precision_ډېرپېښ {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
تعريفنوع عائم floatX;
#تعریف_کړه PRECISION_ډېرپېښ PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
تعريفنوع half floatX;
#تعریف_کړه PRECISION_ډېرپېښ PRECISION_FP16
#وإلا // Default to bfloat16
تعريفنوع __nv_bfloat16 floatX;
#تعریف_کړه PRECISION_ډېرپېښ PRECISION_BF16
#د_که_پای

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#إذا defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* پته) {
    بینښه لنډ bf = __ldcs(reinterpret_cast<ثابت بینښه لنډ*>(پته));
    أرجع __nv_bfloat16_raw{bf};
}

__device__ فراغ __stcs(floatX* پته, floatX ارزښت) {
    __stcs(reinterpret_cast<بینښه لنډ*>(پته), ((__nv_bfloat16_raw)ارزښت).x);
}
#د_که_پای

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 عام:
    NvtxRange(ثابت حرف* s) { nvtx_Range_ټېل(s); }
    NvtxRange(ثابت std::string& بنسټ_str, صحيح شمېره) {
        std::string range_string = بنسټ_str + " " + std::منزل_string(شمېره);
        nvtx_Range_ټېل(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_راوباسه(); }
};
#تعریف_کړه NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
مضمن فراغ device_منزل_file(FILE* dest, فراغ* src, size_t عدد_بایټونه, size_t بفر_اندازه, cuda_بهیر_t بهیر) {
    // allocate pinned buffer for faster, async transfer
    حرف* بفر_space;
    cudaCheck(cudaMallocHost(&بفر_space, 2*بفر_اندازه));
    // split allocation in two
    فراغ* read_بفر = بفر_space;
    فراغ* write_بفر = بفر_space + بفر_اندازه;

    // prime the read buffer; first copy means we have to wait
    حرف* gpu_read_ptr = (حرف*)src;
    size_t copy_amount = std::min(بفر_اندازه, عدد_بایټونه);
    cudaCheck(cudaMemcpyAsync(read_بفر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_منزل_Host, بهیر));
    cudaCheck(cuda_بهیر_Synchronize(بهیر));
    size_t rest_بایټونه = عدد_بایټونه - copy_amount;
    size_t write_بفر_اندازه = copy_amount;
    gpu_read_ptr += copy_amount;

    std::بدلون(read_بفر, write_بفر);
    // now the main loop; as long as there are bytes left
    طالما(rest_بایټونه > 0) {
        // initiate next read
        copy_amount = std::min(بفر_اندازه, rest_بایټونه);
        cudaCheck(cudaMemcpyAsync(read_بفر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_منزل_Host, بهیر));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_بفر, 1, write_بفر_اندازه, dest);
        cudaCheck(cuda_بهیر_Synchronize(بهیر));     // wait for both buffers to be ready.

        std::بدلون(read_بفر, write_بفر);
        rest_بایټونه -= copy_amount;
        write_بفر_اندازه = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_بفر, 1, write_بفر_اندازه, dest);
    cudaCheck(cudaFreeHost(بفر_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
مضمن فراغ file_منزل_device(فراغ* dest, FILE* src, size_t عدد_بایټونه, size_t بفر_اندازه, cuda_بهیر_t بهیر) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    حرف* بفر_space;
    cudaCheck(cudaMallocHost(&بفر_space, 2*بفر_اندازه, cudaHostAllocWriteCombined));
    // split allocation in two
    فراغ* read_بفر = بفر_space;
    فراغ* write_بفر = بفر_space + بفر_اندازه;

    // prime the read buffer;
    حرف* gpu_write_ptr = (حرف*)dest;
    size_t copy_amount = std::min(بفر_اندازه, عدد_بایټونه);
    freadCheck(read_بفر, 1, copy_amount, src);

    size_t rest_بایټونه = عدد_بایټونه - copy_amount;
    size_t write_بفر_اندازه = copy_amount;
    std::بدلون(read_بفر, write_بفر);

    // now the main loop; as long as there are bytes left
    طالما(rest_بایټونه > 0) {
        // initiate next read
        copy_amount = std::min(بفر_اندازه, rest_بایټونه);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_بفر, write_بفر_اندازه, cuda_Memcpy_Host_منزل_Device, بهیر));
        gpu_write_ptr += write_بفر_اندازه;
        // while this is going on, read from disk
        freadCheck(read_بفر, 1, copy_amount, src);
        cudaCheck(cuda_بهیر_Synchronize(بهیر));     // wait for both buffers to be ready.

        std::بدلون(read_بفر, write_بفر);
        rest_بایټونه -= copy_amount;
        write_بفر_اندازه = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_بفر, write_بفر_اندازه, cuda_Memcpy_Host_منزل_Device, بهیر));
    cudaCheck(cuda_بهیر_Synchronize(بهیر));
    cudaCheck(cudaFreeHost(بفر_space));
}

#د_که_پای // CUDA_COMMON_H