/*
Common utilities for CUDA code.
*/
#جيڪڏهن_وصف_نه_ٿيل CUDA_COMMON_H
#وصف_ڪريو CUDA_COMMON_H

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
#وصف_ڪريو WARP_سائيز 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#إذا __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#وصف_ڪريو MAX_1024_ڌاڳا_بلاڪون 2
#وإلا
#وصف_ڪريو MAX_1024_ڌاڳا_بلاڪون 1
#جيڪڏهن_پورو

// convenience macro for calculating grid/block dimensions for kernels
#وصف_ڪريو CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
مضمن فراغ cudaCheck_(cudaError_t error, ثابت حرف *file, صحيح سٽ) {
  إذا (error != cudaSuccess) {
    لکو("[CUDA ERROR] at file %s:%d:\n%s\n", file, سٽ, cudaGetErrorString(error));
    ٻاهر_نڪرو(EXIT_FAILURE);
  }
};
#وصف_ڪريو cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
سانچو<class T>
مضمن فراغ cudaFreeCheck(T** ptr, ثابت حرف *file, صحيح سٽ) {
    cudaError_t error = cudaFree(*ptr);
    إذا (error != cudaSuccess) {
        لکو("[CUDA ERROR] at file %s:%d:\n%s\n", file, سٽ, cudaGetErrorString(error));
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#وصف_ڪريو cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

تعداد Precision_گهڻائي {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
تعريفنوع عائم floatX;
#وصف_ڪريو PRECISION_گهڻائي PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
تعريفنوع half floatX;
#وصف_ڪريو PRECISION_گهڻائي PRECISION_FP16
#وإلا // Default to bfloat16
تعريفنوع __nv_bfloat16 floatX;
#وصف_ڪريو PRECISION_گهڻائي PRECISION_BF16
#جيڪڏهن_پورو

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#إذا defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* پتو) {
    بنانشان ننڍو bf = __ldcs(reinterpret_cast<ثابت بنانشان ننڍو*>(پتو));
    أرجع __nv_bfloat16_raw{bf};
}

__device__ فراغ __stcs(floatX* پتو, floatX قدر) {
    __stcs(reinterpret_cast<بنانشان ننڍو*>(پتو), ((__nv_bfloat16_raw)قدر).x);
}
#جيڪڏهن_پورو

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 عام:
    NvtxRange(ثابت حرف* s) { nvtx_Range_ڌڪو(s); }
    NvtxRange(ثابت std::string& بنياد_str, صحيح عدد) {
        std::string range_string = بنياد_str + " " + std::منزل_string(عدد);
        nvtx_Range_ڌڪو(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_ڪڍو(); }
};
#وصف_ڪريو NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
مضمن فراغ device_منزل_file(FILE* dest, فراغ* src, size_t انگ_بائيٽون, size_t بفر_سائيز, cuda_وهڪرو_t وهڪرو) {
    // allocate pinned buffer for faster, async transfer
    حرف* بفر_space;
    cudaCheck(cudaMallocHost(&بفر_space, 2*بفر_سائيز));
    // split allocation in two
    فراغ* read_بفر = بفر_space;
    فراغ* write_بفر = بفر_space + بفر_سائيز;

    // prime the read buffer; first copy means we have to wait
    حرف* gpu_read_ptr = (حرف*)src;
    size_t copy_amount = std::min(بفر_سائيز, انگ_بائيٽون);
    cudaCheck(cudaMemcpyAsync(read_بفر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_منزل_Host, وهڪرو));
    cudaCheck(cuda_وهڪرو_Synchronize(وهڪرو));
    size_t rest_بائيٽون = انگ_بائيٽون - copy_amount;
    size_t write_بفر_سائيز = copy_amount;
    gpu_read_ptr += copy_amount;

    std::مٽا_سٽا(read_بفر, write_بفر);
    // now the main loop; as long as there are bytes left
    طالما(rest_بائيٽون > 0) {
        // initiate next read
        copy_amount = std::min(بفر_سائيز, rest_بائيٽون);
        cudaCheck(cudaMemcpyAsync(read_بفر, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_منزل_Host, وهڪرو));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_بفر, 1, write_بفر_سائيز, dest);
        cudaCheck(cuda_وهڪرو_Synchronize(وهڪرو));     // wait for both buffers to be ready.

        std::مٽا_سٽا(read_بفر, write_بفر);
        rest_بائيٽون -= copy_amount;
        write_بفر_سائيز = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_بفر, 1, write_بفر_سائيز, dest);
    cudaCheck(cudaFreeHost(بفر_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
مضمن فراغ file_منزل_device(فراغ* dest, FILE* src, size_t انگ_بائيٽون, size_t بفر_سائيز, cuda_وهڪرو_t وهڪرو) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    حرف* بفر_space;
    cudaCheck(cudaMallocHost(&بفر_space, 2*بفر_سائيز, cudaHostAllocWriteCombined));
    // split allocation in two
    فراغ* read_بفر = بفر_space;
    فراغ* write_بفر = بفر_space + بفر_سائيز;

    // prime the read buffer;
    حرف* gpu_write_ptr = (حرف*)dest;
    size_t copy_amount = std::min(بفر_سائيز, انگ_بائيٽون);
    freadCheck(read_بفر, 1, copy_amount, src);

    size_t rest_بائيٽون = انگ_بائيٽون - copy_amount;
    size_t write_بفر_سائيز = copy_amount;
    std::مٽا_سٽا(read_بفر, write_بفر);

    // now the main loop; as long as there are bytes left
    طالما(rest_بائيٽون > 0) {
        // initiate next read
        copy_amount = std::min(بفر_سائيز, rest_بائيٽون);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_بفر, write_بفر_سائيز, cuda_Memcpy_Host_منزل_Device, وهڪرو));
        gpu_write_ptr += write_بفر_سائيز;
        // while this is going on, read from disk
        freadCheck(read_بفر, 1, copy_amount, src);
        cudaCheck(cuda_وهڪرو_Synchronize(وهڪرو));     // wait for both buffers to be ready.

        std::مٽا_سٽا(read_بفر, write_بفر);
        rest_بائيٽون -= copy_amount;
        write_بفر_سائيز = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_بفر, write_بفر_سائيز, cuda_Memcpy_Host_منزل_Device, وهڪرو));
    cudaCheck(cuda_وهڪرو_Synchronize(وهڪرو));
    cudaCheck(cudaFreeHost(بفر_space));
}

#جيڪڏهن_پورو // CUDA_COMMON_H