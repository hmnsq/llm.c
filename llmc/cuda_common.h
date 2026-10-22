/*
Common utilities for CUDA code.
*/
#अगर_अघोषित CUDA_COMMON_H
#घोषणा CUDA_COMMON_H

#समावेश <stdlib.h>
#समावेश <stdio.h>
#समावेश <math.h>
#समावेश <string>
#समावेश <type_traits>      // std::bool_constant
#समावेश <cuda_runtime.h>
#समावेश <nvtx3/nvToolsExt.h>
#समावेश <nvtx3/nvToolsExtCudaRt.h>
#समावेश <cuda_profiler_api.h>
#समावेश <cuda_bf16.h>
#समावेश <cuda_fp16.h>

#समावेश "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
बाह्य cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#घोषणा WARP_आकार 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#अगर __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#घोषणा MAX_1024_THREADS_BLOCKS 2
#अन्यथा
#घोषणा MAX_1024_THREADS_BLOCKS 1
#पूर्ण_अगर

// convenience macro for calculating grid/block dimensions for kernels
#घोषणा CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
अंतरभूत व्योम cudaCheck_(cudaError_t error, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
  अगर (error != cudaSuccess) {
    म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, पंक्ति, cudaGetErrorString(error));
    निकास(निकास_त्रुटि);
  }
};
#घोषणा cudaCheck(err) (cudaCheck_(err, __खाता__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
ढाँचा<class T>
अंतरभूत व्योम cudaFreeCheck(T** ptr, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    cudaError_t error = cudaFree(*ptr);
    अगर (error != cudaSuccess) {
        म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, पंक्ति, cudaGetErrorString(error));
        निकास(निकास_त्रुटि);
    }
    *ptr = nullptr;
}
#घोषणा cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __खाता__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

क्रमागत PrecisionMode {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#अगर defined(ENABLE_FP32)
प्रकार भग्न floatX;
#घोषणा PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#या_अगर defined(ENABLE_FP16)
प्रकार half floatX;
#घोषणा PRECISION_MODE PRECISION_FP16
#अन्यथा // Default to bfloat16
प्रकार __nv_bfloat16 floatX;
#घोषणा PRECISION_MODE PRECISION_BF16
#पूर्ण_अगर

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#अगर defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(स्थिर floatX* address) {
    अचिन्हित लघु bf = __ldcs(reinterpret_cast<स्थिर अचिन्हित लघु*>(address));
    वापस __nv_bfloat16_raw{bf};
}

__device__ व्योम __stcs(floatX* address, floatX मूल्य) {
    __stcs(reinterpret_cast<अचिन्हित लघु*>(address), ((__nv_bfloat16_raw)मूल्य).x);
}
#पूर्ण_अगर

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 खुला:
    NvtxRange(स्थिर अक्षर* s) { nvtx_Range_धकेलो(s); }
    NvtxRange(स्थिर std::string& base_str, पूर्णांक संख्या) {
        std::string range_string = base_str + " " + std::to_string(संख्या);
        nvtx_Range_धकेलो(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_निकालो(); }
};
#घोषणा NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
अंतरभूत व्योम device_to_file(खाता* dest, व्योम* src, माप_प्रकार अंक_bytes, माप_प्रकार बफर_आकार, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    अक्षर* बफर_space;
    cudaCheck(cudaMallocHost(&बफर_space, 2*बफर_आकार));
    // split allocation in two
    व्योम* read_बफर = बफर_space;
    व्योम* write_बफर = बफर_space + बफर_आकार;

    // prime the read buffer; first copy means we have to wait
    अक्षर* gpu_read_ptr = (अक्षर*)src;
    माप_प्रकार copy_amount = std::min(बफर_आकार, अंक_bytes);
    cudaCheck(cudaMemcpyAsync(read_बफर, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    माप_प्रकार rest_bytes = अंक_bytes - copy_amount;
    माप_प्रकार write_बफर_आकार = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_बफर, write_बफर);
    // now the main loop; as long as there are bytes left
    जबतक(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(बफर_आकार, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_बफर, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_बफर, 1, write_बफर_आकार, dest);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_बफर, write_बफर);
        rest_bytes -= copy_amount;
        write_बफर_आकार = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_बफर, 1, write_बफर_आकार, dest);
    cudaCheck(cudaFreeHost(बफर_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
अंतरभूत व्योम file_to_device(व्योम* dest, खाता* src, माप_प्रकार अंक_bytes, माप_प्रकार बफर_आकार, cudaStream_t stream) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    अक्षर* बफर_space;
    cudaCheck(cudaMallocHost(&बफर_space, 2*बफर_आकार, cudaHostAllocWriteCombined));
    // split allocation in two
    व्योम* read_बफर = बफर_space;
    व्योम* write_बफर = बफर_space + बफर_आकार;

    // prime the read buffer;
    अक्षर* gpu_write_ptr = (अक्षर*)dest;
    माप_प्रकार copy_amount = std::min(बफर_आकार, अंक_bytes);
    freadCheck(read_बफर, 1, copy_amount, src);

    माप_प्रकार rest_bytes = अंक_bytes - copy_amount;
    माप_प्रकार write_बफर_आकार = copy_amount;
    std::swap(read_बफर, write_बफर);

    // now the main loop; as long as there are bytes left
    जबतक(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(बफर_आकार, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_बफर, write_बफर_आकार, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_बफर_आकार;
        // while this is going on, read from disk
        freadCheck(read_बफर, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_बफर, write_बफर);
        rest_bytes -= copy_amount;
        write_बफर_आकार = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_बफर, write_बफर_आकार, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(बफर_space));
}

#पूर्ण_अगर // CUDA_COMMON_H