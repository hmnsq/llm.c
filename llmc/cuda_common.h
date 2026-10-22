/*
Common utilities for CUDA code.
*/
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <string>
#अन्तर्भाव <type_traits>      // std::bool_constant
#अन्तर्भाव <cuda_runtime.h>
#अन्तर्भाव <nvtx3/nvToolsExt.h>
#अन्तर्भाव <nvtx3/nvToolsExtCudaRt.h>
#अन्तर्भाव <cuda_profiler_api.h>
#अन्तर्भाव <cuda_bf16.h>
#अन्तर्भाव <cuda_fp16.h>

#अन्तर्भाव "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
बाह्य cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#जर __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#अन्यथा
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
ओळीत शून्य cudaCheck_(cudaError_t error, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
  जर (error != cudaSuccess) {
    म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, ओळ, cudaGetErrorString(error));
    निकास(निकास_त्रुटि);
  }
};
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
ढाँचा<class T>
ओळीत शून्य cudaFreeCheck(T** ptr, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    cudaError_t error = cudaFree(*ptr);
    जर (error != cudaSuccess) {
        म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, ओळ, cudaGetErrorString(error));
        निकास(निकास_त्रुटि);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

गणना PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#जर defined(ENABLE_FP32)
प्रकारव्याख्या दशांश floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#किंवाजर defined(ENABLE_FP16)
प्रकारव्याख्या half floatX;
#define PRECISION_MODE PRECISION_FP16
#अन्यथा // Default to bfloat16
प्रकारव्याख्या __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#जर defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(स्थिरांक floatX* address) {
    अचिन्हित लघु bf = __ldcs(reinterpret_cast<स्थिरांक अचिन्हित लघु*>(address));
    परत __nv_bfloat16_raw{bf};
}

__device__ शून्य __stcs(floatX* address, floatX मूल्य) {
    __stcs(reinterpret_cast<अचिन्हित लघु*>(address), ((__nv_bfloat16_raw)मूल्य).x);
}
#endif

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 सार्वजनिक:
    NvtxRange(स्थिरांक अक्षर* s) { nvtx_Range_ढकला(s); }
    NvtxRange(स्थिरांक std::string& base_str, पूर्णांक संख्या) {
        std::string range_string = base_str + " " + std::to_string(संख्या);
        nvtx_Range_ढकला(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_काढा(); }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
ओळीत शून्य device_to_file(खाता* dest, शून्य* src, माप_प्रकार अंक_bytes, माप_प्रकार buffer_size, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    अक्षर* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    // split allocation in two
    शून्य* read_buffer = buffer_space;
    शून्य* write_buffer = buffer_space + buffer_size;

    // prime the read buffer; first copy means we have to wait
    अक्षर* gpu_read_ptr = (अक्षर*)src;
    माप_प्रकार copy_amount = std::min(buffer_size, अंक_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    माप_प्रकार rest_bytes = अंक_bytes - copy_amount;
    माप_प्रकार write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // now the main loop; as long as there are bytes left
    पर्यंत(rest_bytes > 0) {
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
ओळीत शून्य file_to_device(शून्य* dest, खाता* src, माप_प्रकार अंक_bytes, माप_प्रकार buffer_size, cudaStream_t stream) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    अक्षर* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
    // split allocation in two
    शून्य* read_buffer = buffer_space;
    शून्य* write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    अक्षर* gpu_write_ptr = (अक्षर*)dest;
    माप_प्रकार copy_amount = std::min(buffer_size, अंक_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    माप_प्रकार rest_bytes = अंक_bytes - copy_amount;
    माप_प्रकार write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // now the main loop; as long as there are bytes left
    पर्यंत(rest_bytes > 0) {
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