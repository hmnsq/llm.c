/*
Common utilities for CUDA code.
*/
#اگر_تعریف_نشده CUDA_COMMON_H
#تعریف_کن CUDA_COMMON_H

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
#تعریف_کن WARP_اندازه 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#إذا __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#تعریف_کن MAX_1024_نخها_بلوکها 2
#وإلا
#تعریف_کن MAX_1024_نخها_بلوکها 1
#پایان_اگر

// convenience macro for calculating grid/block dimensions for kernels
#تعریف_کن CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
مضمن فراغ cudaCheck_(cudaError_t error, ثابت حرف *file, صحيح سطر) {
  إذا (error != cudaSuccess) {
    چاپ_کن("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر, cudaGetErrorString(error));
    خروج(EXIT_FAILURE);
  }
};
#تعریف_کن cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
قالب<class T>
مضمن فراغ cudaFreeCheck(T** ptr, ثابت حرف *file, صحيح سطر) {
    cudaError_t error = cudaFree(*ptr);
    إذا (error != cudaSuccess) {
        چاپ_کن("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر, cudaGetErrorString(error));
        خروج(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#تعریف_کن cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

تعداد Precision_نما {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
تعريفنوع عائم floatX;
#تعریف_کن PRECISION_نما PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
تعريفنوع half floatX;
#تعریف_کن PRECISION_نما PRECISION_FP16
#وإلا // Default to bfloat16
تعريفنوع __nv_bfloat16 floatX;
#تعریف_کن PRECISION_نما PRECISION_BF16
#پایان_اگر

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#إذا defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* آدرس) {
    بیعلامت کوتاه bf = __ldcs(reinterpret_cast<ثابت بیعلامت کوتاه*>(آدرس));
    أرجع __nv_bfloat16_raw{bf};
}

__device__ فراغ __stcs(floatX* آدرس, floatX مقدار) {
    __stcs(reinterpret_cast<بیعلامت کوتاه*>(آدرس), ((__nv_bfloat16_raw)مقدار).x);
}
#پایان_اگر

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 عام:
    NvtxRange(ثابت حرف* s) { nvtx_Range_افزودن(s); }
    NvtxRange(ثابت std::string& پایه_str, صحيح عدد) {
        std::string range_string = پایه_str + " " + std::مقصد_string(عدد);
        nvtx_Range_افزودن(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_برداشت(); }
};
#تعریف_کن NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
مضمن فراغ device_مقصد_file(FILE* dest, فراغ* src, size_t رقم_بایتها, size_t میانگیر_اندازه, cuda_جریان_t جریان) {
    // allocate pinned buffer for faster, async transfer
    حرف* میانگیر_space;
    cudaCheck(cudaMallocHost(&میانگیر_space, 2*میانگیر_اندازه));
    // split allocation in two
    فراغ* read_میانگیر = میانگیر_space;
    فراغ* write_میانگیر = میانگیر_space + میانگیر_اندازه;

    // prime the read buffer; first copy means we have to wait
    حرف* gpu_read_ptr = (حرف*)src;
    size_t copy_amount = std::min(میانگیر_اندازه, رقم_بایتها);
    cudaCheck(cudaMemcpyAsync(read_میانگیر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_مقصد_Host, جریان));
    cudaCheck(cuda_جریان_Synchronize(جریان));
    size_t rest_بایتها = رقم_بایتها - copy_amount;
    size_t write_میانگیر_اندازه = copy_amount;
    gpu_read_ptr += copy_amount;

    std::جابجا(read_میانگیر, write_میانگیر);
    // now the main loop; as long as there are bytes left
    طالما(rest_بایتها > 0) {
        // initiate next read
        copy_amount = std::min(میانگیر_اندازه, rest_بایتها);
        cudaCheck(cudaMemcpyAsync(read_میانگیر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_مقصد_Host, جریان));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_میانگیر, 1, write_میانگیر_اندازه, dest);
        cudaCheck(cuda_جریان_Synchronize(جریان));     // wait for both buffers to be ready.

        std::جابجا(read_میانگیر, write_میانگیر);
        rest_بایتها -= copy_amount;
        write_میانگیر_اندازه = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_میانگیر, 1, write_میانگیر_اندازه, dest);
    cudaCheck(cudaFreeHost(میانگیر_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
مضمن فراغ file_مقصد_device(فراغ* dest, FILE* src, size_t رقم_بایتها, size_t میانگیر_اندازه, cuda_جریان_t جریان) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    حرف* میانگیر_space;
    cudaCheck(cudaMallocHost(&میانگیر_space, 2*میانگیر_اندازه, cudaHostAllocWriteCombined));
    // split allocation in two
    فراغ* read_میانگیر = میانگیر_space;
    فراغ* write_میانگیر = میانگیر_space + میانگیر_اندازه;

    // prime the read buffer;
    حرف* gpu_write_ptr = (حرف*)dest;
    size_t copy_amount = std::min(میانگیر_اندازه, رقم_بایتها);
    freadCheck(read_میانگیر, 1, copy_amount, src);

    size_t rest_بایتها = رقم_بایتها - copy_amount;
    size_t write_میانگیر_اندازه = copy_amount;
    std::جابجا(read_میانگیر, write_میانگیر);

    // now the main loop; as long as there are bytes left
    طالما(rest_بایتها > 0) {
        // initiate next read
        copy_amount = std::min(میانگیر_اندازه, rest_بایتها);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_میانگیر, write_میانگیر_اندازه, cuda_Memcpy_Host_مقصد_Device, جریان));
        gpu_write_ptr += write_میانگیر_اندازه;
        // while this is going on, read from disk
        freadCheck(read_میانگیر, 1, copy_amount, src);
        cudaCheck(cuda_جریان_Synchronize(جریان));     // wait for both buffers to be ready.

        std::جابجا(read_میانگیر, write_میانگیر);
        rest_بایتها -= copy_amount;
        write_میانگیر_اندازه = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_میانگیر, write_میانگیر_اندازه, cuda_Memcpy_Host_مقصد_Device, جریان));
    cudaCheck(cuda_جریان_Synchronize(جریان));
    cudaCheck(cudaFreeHost(میانگیر_space));
}

#پایان_اگر // CUDA_COMMON_H