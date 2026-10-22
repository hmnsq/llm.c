/*
Common utilities for CUDA code.
*/
#నిర్వచితం_కాకపోతే CUDA_COMMON_H
#నిర్వచించు CUDA_COMMON_H

#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <string>
#అన్తర్భావ <type_traits>      // std::bool_constant
#అన్తర్భావ <cuda_runtime.h>
#అన్తర్భావ <nvtx3/nvToolsExt.h>
#అన్తర్భావ <nvtx3/nvToolsExtCudaRt.h>
#అన్తర్భావ <cuda_profiler_api.h>
#అన్తర్భావ <cuda_bf16.h>
#అన్తర్భావ <cuda_fp16.h>

#అన్తర్భావ "utils.h"

// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
బాహ్య cudaDeviceProp deviceProp;

// WarpSize is not a compile time constant
// Defining here like this possibly allows the compiler to optimize better
#నిర్వచించు WARP_కొలత 32U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#ఒకవేళ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#నిర్వచించు MAX_1024_దారాలు_ఖండాలు 2
#లేకపోతే
#నిర్వచించు MAX_1024_దారాలు_ఖండాలు 1
#అయితే_ముగింపు

// convenience macro for calculating grid/block dimensions for kernels
#నిర్వచించు CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking. Underscore added so this function can be called directly not just via macro
అంతఃవరుస శూన్య cudaCheck_(cudaError_t error, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
  ఒకవేళ (error != cudaSuccess) {
    రాయి("[CUDA ERROR] at file %s:%d:\n%s\n", file, పంక్తి, cudaGetErrorString(error));
    నిష్క్రమించు(EXIT_FAILURE);
  }
};
#నిర్వచించు cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// like cudaFree, but checks for errors _and_ resets the pointer.
మూస<class T>
అంతఃవరుస శూన్య cudaFreeCheck(T** ptr, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    cudaError_t error = cudaFree(*ptr);
    ఒకవేళ (error != cudaSuccess) {
        రాయి("[CUDA ERROR] at file %s:%d:\n%s\n", file, పంక్తి, cudaGetErrorString(error));
        నిష్క్రమించు(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#నిర్వచించు cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

గణన Precision_బహుళకం {
    PRECISION_FP32,
    PRECISION_FP16,
    PRECISION_BF16
};

// Specific configurations based on the enabled precision
#ఒకవేళ defined(ENABLE_FP32)
రకనిర్వచనం దశాంశం floatX;
#నిర్వచించు PRECISION_బహుళకం PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#లేదావేళ defined(ENABLE_FP16)
రకనిర్వచనం half floatX;
#నిర్వచించు PRECISION_బహుళకం PRECISION_FP16
#లేకపోతే // Default to bfloat16
రకనిర్వచనం __nv_bfloat16 floatX;
#నిర్వచించు PRECISION_బహుళకం PRECISION_BF16
#అయితే_ముగింపు

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#ఒకవేళ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(స్థిరరాశి floatX* చిరునామా) {
    గుర్తులేని పొట్టి bf = __ldcs(reinterpret_cast<స్థిరరాశి గుర్తులేని పొట్టి*>(చిరునామా));
    తిరిగి __nv_bfloat16_raw{bf};
}

__device__ శూన్య __stcs(floatX* చిరునామా, floatX విలువ) {
    __stcs(reinterpret_cast<గుర్తులేని పొట్టి*>(చిరునామా), ((__nv_bfloat16_raw)విలువ).x);
}
#అయితే_ముగింపు

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
 బహిరంగ:
    NvtxRange(స్థిరరాశి అక్షరం* s) { nvtx_Range_నెట్టు(s); }
    NvtxRange(స్థిరరాశి std::string& పునాది_str, పూర్ణాంకం సంఖ్య) {
        std::string range_string = పునాది_str + " " + std::గమ్యం_string(సంఖ్య);
        nvtx_Range_నెట్టు(range_string.c_str());
    }
    ~NvtxRange() { nvtx_Range_తీయి(); }
};
#నిర్వచించు NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
అంతఃవరుస శూన్య device_గమ్యం_file(FILE* dest, శూన్య* src, size_t అంకె_బైట్లు, size_t బఫరు_కొలత, cuda_ప్రవాహం_t ప్రవాహం) {
    // allocate pinned buffer for faster, async transfer
    అక్షరం* బఫరు_space;
    cudaCheck(cudaMallocHost(&బఫరు_space, 2*బఫరు_కొలత));
    // split allocation in two
    శూన్య* read_బఫరు = బఫరు_space;
    శూన్య* write_బఫరు = బఫరు_space + బఫరు_కొలత;

    // prime the read buffer; first copy means we have to wait
    అక్షరం* gpu_read_ptr = (అక్షరం*)src;
    size_t copy_amount = std::min(బఫరు_కొలత, అంకె_బైట్లు);
    cudaCheck(cudaMemcpyAsync(read_బఫరు, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_గమ్యం_Host, ప్రవాహం));
    cudaCheck(cuda_ప్రవాహం_Synchronize(ప్రవాహం));
    size_t rest_బైట్లు = అంకె_బైట్లు - copy_amount;
    size_t write_బఫరు_కొలత = copy_amount;
    gpu_read_ptr += copy_amount;

    std::మార్పిడి(read_బఫరు, write_బఫరు);
    // now the main loop; as long as there are bytes left
    అయితే(rest_బైట్లు > 0) {
        // initiate next read
        copy_amount = std::min(బఫరు_కొలత, rest_బైట్లు);
        cudaCheck(cudaMemcpyAsync(read_బఫరు, gpu_read_ptr, copy_amount, cuda_Memcpy_Device_గమ్యం_Host, ప్రవాహం));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_బఫరు, 1, write_బఫరు_కొలత, dest);
        cudaCheck(cuda_ప్రవాహం_Synchronize(ప్రవాహం));     // wait for both buffers to be ready.

        std::మార్పిడి(read_బఫరు, write_బఫరు);
        rest_బైట్లు -= copy_amount;
        write_బఫరు_కొలత = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_బఫరు, 1, write_బఫరు_కొలత, dest);
    cudaCheck(cudaFreeHost(బఫరు_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
అంతఃవరుస శూన్య file_గమ్యం_device(శూన్య* dest, FILE* src, size_t అంకె_బైట్లు, size_t బఫరు_కొలత, cuda_ప్రవాహం_t ప్రవాహం) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    అక్షరం* బఫరు_space;
    cudaCheck(cudaMallocHost(&బఫరు_space, 2*బఫరు_కొలత, cudaHostAllocWriteCombined));
    // split allocation in two
    శూన్య* read_బఫరు = బఫరు_space;
    శూన్య* write_బఫరు = బఫరు_space + బఫరు_కొలత;

    // prime the read buffer;
    అక్షరం* gpu_write_ptr = (అక్షరం*)dest;
    size_t copy_amount = std::min(బఫరు_కొలత, అంకె_బైట్లు);
    freadCheck(read_బఫరు, 1, copy_amount, src);

    size_t rest_బైట్లు = అంకె_బైట్లు - copy_amount;
    size_t write_బఫరు_కొలత = copy_amount;
    std::మార్పిడి(read_బఫరు, write_బఫరు);

    // now the main loop; as long as there are bytes left
    అయితే(rest_బైట్లు > 0) {
        // initiate next read
        copy_amount = std::min(బఫరు_కొలత, rest_బైట్లు);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_బఫరు, write_బఫరు_కొలత, cuda_Memcpy_Host_గమ్యం_Device, ప్రవాహం));
        gpu_write_ptr += write_బఫరు_కొలత;
        // while this is going on, read from disk
        freadCheck(read_బఫరు, 1, copy_amount, src);
        cudaCheck(cuda_ప్రవాహం_Synchronize(ప్రవాహం));     // wait for both buffers to be ready.

        std::మార్పిడి(read_బఫరు, write_బఫరు);
        rest_బైట్లు -= copy_amount;
        write_బఫరు_కొలత = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_బఫరు, write_బఫరు_కొలత, cuda_Memcpy_Host_గమ్యం_Device, ప్రవాహం));
    cudaCheck(cuda_ప్రవాహం_Synchronize(ప్రవాహం));
    cudaCheck(cudaFreeHost(బఫరు_space));
}

#అయితే_ముగింపు // CUDA_COMMON_H