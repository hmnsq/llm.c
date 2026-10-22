/*
Common utilities for CUDA code.
*/
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string>
#অন্তর্ভাব <type_traits>      // std::bool_constant
#অন্তর্ভাব <cuda_runtime.h>
#অন্তর্ভাব <nvtx3/nvToolsExt.h>
#অন্তর্ভাব <nvtx3/nvToolsExtCudaRt.h>
#অন্তর্ভাব <cuda_profiler_api.h>
#অন্তর্ভাব <cuda_bf16.h>
#অন্তর্ভাব <cuda_fp16.h>

#অন্তর্ভাব "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
বাহ্যিক cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#define WARP_আকার 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#যদি __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#অন্যথা
#define MAX_1024_THREADS_BLOCKS 1
#endif

// convenience macro for calculating grid/block dimensions for kernels
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
অন্তঃরেখ শূন্য cudaCheck_(cudaError_t error, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
  যদি (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, সারি, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
template<class T>
অন্তঃরেখ শূন্য cudaFreeCheck(T** ptr, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    cudaError_t error = cudaFree(*ptr);
    যদি (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, সারি, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

গণনা PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#যদি defined(ENABLE_FP32)
প্রকারলেখ দশমিক floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#নতুবা_যদি defined(ENABLE_FP16)
প্রকারলেখ half floatX;
#define PRECISION_MODE PRECISION_FP16
#অন্যথা // Default to bfloat16
প্রকারলেখ __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#যদি defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ধ্রুবক floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<ধ্রুবক unsigned short*>(address));
    ফেরত __nv_bfloat16_raw{bf};
}

__device__ শূন্য __stcs(floatX* address, floatX মান) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)মান).x);
}
#endif

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 সর্বজনীন:
    NvtxRange(ধ্রুবক অক্ষর* s) { nvtx_Range_ঠেলো(s); }
    NvtxRange(ধ্রুবক std::string& base_str, পূর্ণসংখ্যা সংখ্যা) {
        std::string range_string = base_str + " " + std::to_string(সংখ্যা);
        nvtx_Range_ঠেলো(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_বের_করো(); }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
অন্তঃরেখ শূন্য device_to_file(FILE* dest, শূন্য* src, size_t অঙ্ক_bytes, size_t বাফার_আকার, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    অক্ষর* বাফার_space;
    cudaCheck(cudaMallocHost(&বাফার_space, 2*বাফার_আকার));
    // split allocation in two
    শূন্য* read_বাফার = বাফার_space;
    শূন্য* write_বাফার = বাফার_space + বাফার_আকার;

    // prime the read buffer; first copy means we have to wait
    অক্ষর* gpu_read_ptr = (অক্ষর*)src;
    size_t copy_amount = std::min(বাফার_আকার, অঙ্ক_bytes);
    cudaCheck(cudaMemcpyAsync(read_বাফার, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes = অঙ্ক_bytes - copy_amount;
    size_t write_বাফার_আকার = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_বাফার, write_বাফার);
    // now the main loop; as long as there are bytes left
    যখন(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(বাফার_আকার, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_বাফার, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_বাফার, 1, write_বাফার_আকার, dest);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_বাফার, write_বাফার);
        rest_bytes -= copy_amount;
        write_বাফার_আকার = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_বাফার, 1, write_বাফার_আকার, dest);
    cudaCheck(cudaFreeHost(বাফার_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
অন্তঃরেখ শূন্য file_to_device(শূন্য* dest, FILE* src, size_t অঙ্ক_bytes, size_t বাফার_আকার, cudaStream_t stream) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    অক্ষর* বাফার_space;
    cudaCheck(cudaMallocHost(&বাফার_space, 2*বাফার_আকার, cudaHostAllocWriteCombined));
    // split allocation in two
    শূন্য* read_বাফার = বাফার_space;
    শূন্য* write_বাফার = বাফার_space + বাফার_আকার;

    // prime the read buffer;
    অক্ষর* gpu_write_ptr = (অক্ষর*)dest;
    size_t copy_amount = std::min(বাফার_আকার, অঙ্ক_bytes);
    freadCheck(read_বাফার, 1, copy_amount, src);

    size_t rest_bytes = অঙ্ক_bytes - copy_amount;
    size_t write_বাফার_আকার = copy_amount;
    std::swap(read_বাফার, write_বাফার);

    // now the main loop; as long as there are bytes left
    যখন(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(বাফার_আকার, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_বাফার, write_বাফার_আকার, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_বাফার_আকার;
        // while this is going on, read from disk
        freadCheck(read_বাফার, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_বাফার, write_বাফার);
        rest_bytes -= copy_amount;
        write_বাফার_আকার = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_বাফার, write_বাফার_আকার, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(বাফার_space));
}

#endif // CUDA_COMMON_H