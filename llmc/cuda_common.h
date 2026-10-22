/*
Common utilities for CUDA code.
*/
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <type_traits>      // std::bool_constant
#અન્તર્ભાવ <cuda_runtime.h>
#અન્તર્ભાવ <nvtx3/nvToolsExt.h>
#અન્તર્ભાવ <nvtx3/nvToolsExtCudaRt.h>
#અન્તર્ભાવ <cuda_profiler_api.h>
#અન્તર્ભાવ <cuda_bf16.h>
#અન્તર્ભાવ <cuda_fp16.h>

#અન્તર્ભાવ "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
બાહ્ય cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#જો __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#નહીંતર
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
અંતઃરેખ શૂન્ય cudaCheck_(cudaError_t error, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
  જો (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, લીટી, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
template<class T>
અંતઃરેખ શૂન્ય cudaFreeCheck(T** ptr, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    cudaError_t error = cudaFree(*ptr);
    જો (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, લીટી, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

ગણના PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#જો defined(ENABLE_FP32)
પ્રકારવ્યાખ્યા દશાંશ floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#અથવાજો defined(ENABLE_FP16)
પ્રકારવ્યાખ્યા half floatX;
#define PRECISION_MODE PRECISION_FP16
#નહીંતર // Default to bfloat16
પ્રકારવ્યાખ્યા __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#જો defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(અચલ floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<અચલ unsigned short*>(address));
    પાછા __nv_bfloat16_raw{bf};
}

__device__ શૂન્ય __stcs(floatX* address, floatX કિંમત) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)કિંમત).x);
}
#endif

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 જાહેર:
    NvtxRange(અચલ અક્ષર* s) { nvtx_Range_ધકેલો(s); }
    NvtxRange(અચલ std::string& base_str, પૂર્ણાંક સંખ્યા) {
        std::string range_string = base_str + " " + std::to_string(સંખ્યા);
        nvtx_Range_ધકેલો(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_કાઢો(); }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
અંતઃરેખ શૂન્ય device_to_file(FILE* dest, શૂન્ય* src, size_t અંક_bytes, size_t buffer_size, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    અક્ષર* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    // split allocation in two
    શૂન્ય* read_buffer = buffer_space;
    શૂન્ય* write_buffer = buffer_space + buffer_size;

    // prime the read buffer; first copy means we have to wait
    અક્ષર* gpu_read_ptr = (અક્ષર*)src;
    size_t copy_amount = std::min(buffer_size, અંક_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes = અંક_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // now the main loop; as long as there are bytes left
    જ્યારે(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    cudaCheck(cudaFreeHost(buffer_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
અંતઃરેખ શૂન્ય file_to_device(શૂન્ય* dest, FILE* src, size_t અંક_bytes, size_t buffer_size, cudaStream_t stream) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    અક્ષર* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
    // split allocation in two
    શૂન્ય* read_buffer = buffer_space;
    શૂન્ય* write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    અક્ષર* gpu_write_ptr = (અક્ષર*)dest;
    size_t copy_amount = std::min(buffer_size, અંક_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes = અંક_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // now the main loop; as long as there are bytes left
    જ્યારે(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;
        // while this is going on, read from disk
        freadCheck(read_buffer, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(buffer_space));
}

#endif // CUDA_COMMON_H