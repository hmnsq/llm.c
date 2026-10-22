#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <cuda_runtime.h>
#अन्तर्भाव <cublas_v2.h>
#अन्तर्भाव <cublasLt.h>
#अन्तर्भाव <float.h>

#define WARP_SIZE 32U
बाह्य cudaDeviceProp deviceProp;

ढाँचा<class T>
__host__ __device__ T ceil_div(T dividend, T भाजक) {
    फर्काऊ (dividend + भाजक-1) / भाजक;
}

__device__ भिन्न warpReduceSum(भिन्न val) {
    लागि (पूर्णाङ्क offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    फर्काऊ val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = भिन्न (*) (भिन्न);

ढाँचा<reduction_func_t warp_reduction>
__device__ अन्तःस्थ भिन्न blockReduce(भिन्न val, bool final_sync, भिन्न out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ भिन्न shared_val[WARP_SIZE];
    अचर पूर्णाङ्क lane_id = threadIdx.x % WARP_SIZE;
    अचर पूर्णाङ्क warp_id = threadIdx.x / WARP_SIZE;
    अचर पूर्णाङ्क अङ्क_warps = blockDim.x / WARP_SIZE;

    भिन्न warp_val = warp_reduction(val);
    यदि (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < अङ्क_warps) ? shared_val[lane_id] : out_of_bounds;
    भिन्न block_val = warp_reduction(warp_val);

    यदि (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    फर्काऊ block_val;
}

// Helper function to call blockReduce with default arguments
ढाँचा<reduction_func_t warp_reduction>
__device__ अन्तःस्थ भिन्न blockReduce(भिन्न val) {
    फर्काऊ blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
शून्य cuda_check(cudaError_t error, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    यदि (error != cudaSuccess) {
        म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, पङ्क्ति,
               cudaGetErrorString(error));
        निकास(निकास_त्रुटि);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
शून्य cublasCheck(cublasStatus_t status, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति)
{
    यदि (status != CUBLAS_STATUS_SUCCESS) {
        म_लिखो("[cuBLAS ERROR]: %d %s %d\n", status, file, पङ्क्ति);
        निकास(निकास_त्रुटि);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
स्थिर माप_प्रकार cublaslt_workspace_size = 32 * 1024 * 1024;
स्थिर शून्य* cublaslt_workspace = रिक्त;
स्थिर cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
पूर्णाङ्क cuda_arch_major = 0;
पूर्णाङ्क cuda_arch_minor = 0;
पूर्णाङ्क cuda_अङ्क_SMs = 0; // for persistent threads where we want 1 threadblock per SM
पूर्णाङ्क cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#यदि __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#अन्यथा
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

ढाँचा<class ElementType>
संरचना alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = स्वभाव;
    __device__ explicit Packed128(int4 bits) {
        static_assert(परिमाण(bits) == परिमाण(payload), "Size mismatch.");
        स_नकल(&payload, &bits, परिमाण(bits));
    }

    __device__  स्थिर Packed128 constant(ElementType मान) {
        Packed128 result;
        लागि(पूर्णाङ्क k = 0; k < size; ++k) {
            result.payload[k] = मान;
        }
        फर्काऊ result;
    }

    __device__ स्थिर Packed128 zeros() {
        फर्काऊ constant(0);
    }

    __device__ स्थिर Packed128 ones() {
        फर्काऊ constant(1);
    }

    __device__ ElementType& चालक[](पूर्णाङ्क index) {
        फर्काऊ payload[index];
    }
    __device__ अचर ElementType& चालक[](पूर्णाङ्क index) अचर {
        फर्काऊ payload[index];
    }
    __device__ int4 get_bits() अचर {
        int4 bits;
        static_assert(परिमाण(bits) == परिमाण(payload), "Size mismatch.");
        स_नकल(&bits, &payload, परिमाण(bits));
        फर्काऊ bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    स्थिर constexpr अचर पूर्णाङ्क size = परिमाण(int4) / परिमाण(ElementType);
    ElementType payload[size];
};

// short-form typedef
प्रकारनिर्देश Packed128<भिन्न> f128;

// load a Packed128 from an aligned memory address
ढाँचा<class ElementType>
__device__ Packed128<ElementType> load128(अचर ElementType* address) {
    फर्काऊ Packed128<ElementType>{*reinterpret_cast<अचर int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
ढाँचा<class ElementType>
__device__ Packed128<ElementType> load128cs(अचर ElementType* address) {
    फर्काऊ Packed128<ElementType>{__ldcs(reinterpret_cast<अचर int4*>(address))};
}
// store a Packed128 to an aligned memory address
ढाँचा<class ElementType>
__device__ शून्य store128(ElementType* target, Packed128<ElementType> मान) {
    *reinterpret_cast<int4*>(target) = मान.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
ढाँचा<class ElementType>
__device__ शून्य store128cs(ElementType* target, Packed128<ElementType> मान) {
    __stcs(reinterpret_cast<int4*>(target), मान.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
ढाँचा<class ElementType>
__device__ शून्य store128cg(ElementType* target, Packed128<ElementType> मान) {
    __stcg(reinterpret_cast<int4*>(target), मान.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#यदि defined(ENABLE_BF16)

प्रकारनिर्देश __nv_bfloat16 floatX;
प्रकारनिर्देश __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#अथवायदि defined(ENABLE_FP16)

प्रकारनिर्देश half floatX;
प्रकारनिर्देश half floatN;

#अन्यथा

प्रकारनिर्देश भिन्न floatX;
प्रकारनिर्देश भिन्न floatN;
#endif

प्रकारनिर्देश Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#यदि defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(अचर floatX* address) {
    अचिन्हित लघु bf = __ldcs(reinterpret_cast<अचर अचिन्हित लघु*>(address));
    फर्काऊ __nv_bfloat16_raw{bf};
}

__device__ शून्य __stcs(floatX* address, floatX मान) {
    __stcs(reinterpret_cast<अचिन्हित लघु*>(address), ((__nv_bfloat16_raw)मान).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

भिन्न* make_random_float_01(माप_प्रकार N) {
    भिन्न* arr = (भिन्न*)दो_स्मृति(N * परिमाण(भिन्न));
    लागि (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((भिन्न)अक्रम() / अक्रम_उच्च); // range 0..1
    }
    फर्काऊ arr;
}

भिन्न* make_random_float(माप_प्रकार N) {
    भिन्न* arr = (भिन्न*)दो_स्मृति(N * परिमाण(भिन्न));
    लागि (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((भिन्न)अक्रम() / अक्रम_उच्च) * 2.0 - 1.0; // range -1..1
    }
    फर्काऊ arr;
}

पूर्णाङ्क* make_random_int(माप_प्रकार N, पूर्णाङ्क V) {
    पूर्णाङ्क* arr = (पूर्णाङ्क*)दो_स्मृति(N * परिमाण(पूर्णाङ्क));
    लागि (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = अक्रम() % V; // range 0..V-1
    }
    फर्काऊ arr;
}

भिन्न* make_zeros_float(माप_प्रकार N) {
    भिन्न* arr = (भिन्न*)दो_स्मृति(N * परिमाण(भिन्न));
    स_रखो(arr, 0, N * परिमाण(भिन्न)); // all zero
    फर्काऊ arr;
}

भिन्न* make_ones_float(माप_प्रकार N) {
    भिन्न* arr = (भिन्न*)दो_स्मृति(N * परिमाण(भिन्न));
    लागि (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    फर्काऊ arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

ढाँचा<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, भिन्न* h_ptr, माप_प्रकार गन्ती) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)दो_स्मृति(गन्ती * परिमाण(TargetType));
    लागि (पूर्णाङ्क i = 0; i < गन्ती; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, गन्ती * परिमाण(TargetType), cudaMemcpyHostToDevice);
    मुक्त(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    फर्काऊ status;
}

शून्य setup_main() {
    बेक्रम(0);   // determinism

    // set up the device
    पूर्णाङ्क deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_अङ्क_SMs = deviceProp.multi_Processor_गन्ती;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    पूर्णाङ्क enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

ढाँचा<class D, class T>
शून्य validate_result(D* device_result, अचर T* cpu_reference, अचर वर्ण* name, std::माप_प्रकार अङ्क_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)दो_स्मृति(अङ्क_elements * परिमाण(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, अङ्क_elements * परिमाण(D), cudaMemcpyDeviceToHost));
    पूर्णाङ्क nfaults = 0;
#ifndef ENABLE_BF16
    भिन्न epsilon = भग्न_अंतर;
#अन्यथा
    भिन्न epsilon = 0.079;
#endif
    लागि (पूर्णाङ्क i = 0; i < अङ्क_elements; i++) {
        // Skip masked elements
        यदि(!है_सीमित(cpu_reference[i]))
            जारी;

        // print the first few comparisons
        यदि (i < 5) {
            म_लिखो("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        भिन्न t_eff = tolerance + भ_असल(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        यदि (भ_असल(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            म_लिखो("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            यदि (nfaults >= 10) {
                मुक्त(out_gpu);
                निकास(निकास_त्रुटि);
            }
        }
    }

    यदि (nfaults > 0) {
        मुक्त(out_gpu);
        निकास(निकास_त्रुटि);
    }

    मुक्त(out_gpu);
}

ढाँचा<class Kernel, class... KernelArgs>
भिन्न benchmark_kernel(पूर्णाङ्क repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    पूर्णाङ्क deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    शून्य* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    भिन्न elapsed_time = 0.f;
    लागि (पूर्णाङ्क i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        भिन्न single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    फर्काऊ elapsed_time / repeats;
}