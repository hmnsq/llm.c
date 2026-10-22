#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <cuda_runtime.h>
#ಅನ್ತರ್ಭಾವ <cublas_v2.h>
#ಅನ್ತರ್ಭಾವ <cublasLt.h>
#ಅನ್ತರ್ಭಾವ <float.h>

#define WARP_SIZE 32U
ಬಾಹ್ಯ cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T ಭಾಜಕ) {
    ಪ್ರತ್ಯಾಗಮ (dividend + ಭಾಜಕ-1) / ಭಾಜಕ;
}

__device__ ಭಿನ್ನ warpReduceSum(ಭಿನ್ನ val) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    ಪ್ರತ್ಯಾಗಮ val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = ಭಿನ್ನ (*) (ಭಿನ್ನ);

template<reduction_func_t warp_reduction>
__device__ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ blockReduce(ಭಿನ್ನ val, bool final_sync, ಭಿನ್ನ out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ ಭಿನ್ನ shared_val[WARP_SIZE];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane_id = threadIdx.x % WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp_id = threadIdx.x / WARP_SIZE;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_warps = blockDim.x / WARP_SIZE;

    ಭಿನ್ನ warp_val = warp_reduction(val);
    ಯದಿ (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < ಅಂಕೆ_warps) ? shared_val[lane_id] : out_of_bounds;
    ಭಿನ್ನ block_val = warp_reduction(warp_val);

    ಯದಿ (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    ಪ್ರತ್ಯಾಗಮ block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ blockReduce(ಭಿನ್ನ val) {
    ಪ್ರತ್ಯಾಗಮ blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
ಶೂನ್ಯ cuda_check(cudaError_t error, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    ಯದಿ (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
ಶೂನ್ಯ cublasCheck(cublasStatus_t status, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು)
{
    ಯದಿ (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, ಸಾಲು);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ಸ್ಥಿರ size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ಸ್ಥಿರ ಶೂನ್ಯ* cublaslt_workspace = ರಿಕ್ತ;
ಸ್ಥಿರ cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
ಪೂರ್ಣಾಙ್ಕ cuda_arch_major = 0;
ಪೂರ್ಣಾಙ್ಕ cuda_arch_minor = 0;
ಪೂರ್ಣಾಙ್ಕ cuda_ಅಂಕೆ_SMs = 0; // for persistent threads where we want 1 threadblock per SM
ಪೂರ್ಣಾಙ್ಕ cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#ಯದಿ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#ಅನ್ಯಥಾ
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
ಸಂರಚನಾ alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = ಸ್ವಭಾವ;
    __device__ explicit Packed128(int4 bits) {
        static_assert(ಪರಿಮಾಣ(bits) == ಪರಿಮಾಣ(payload), "Size mismatch.");
        memcpy(&payload, &bits, ಪರಿಮಾಣ(bits));
    }

    __device__  ಸ್ಥಿರ Packed128 constant(ElementType ಮೌಲ್ಯ) {
        Packed128 result;
        ಪ್ರತಿ(ಪೂರ್ಣಾಙ್ಕ k = 0; k < size; ++k) {
            result.payload[k] = ಮೌಲ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ result;
    }

    __device__ ಸ್ಥಿರ Packed128 zeros() {
        ಪ್ರತ್ಯಾಗಮ constant(0);
    }

    __device__ ಸ್ಥಿರ Packed128 ones() {
        ಪ್ರತ್ಯಾಗಮ constant(1);
    }

    __device__ ElementType& operator[](ಪೂರ್ಣಾಙ್ಕ index) {
        ಪ್ರತ್ಯಾಗಮ payload[index];
    }
    __device__ ಧ್ರುವ ElementType& operator[](ಪೂರ್ಣಾಙ್ಕ index) ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ payload[index];
    }
    __device__ int4 get_bits() ಧ್ರುವ {
        int4 bits;
        static_assert(ಪರಿಮಾಣ(bits) == ಪರಿಮಾಣ(payload), "Size mismatch.");
        memcpy(&bits, &payload, ಪರಿಮಾಣ(bits));
        ಪ್ರತ್ಯಾಗಮ bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ಸ್ಥಿರ constexpr ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ size = ಪರಿಮಾಣ(int4) / ಪರಿಮಾಣ(ElementType);
    ElementType payload[size];
};

// short-form typedef
ಪ್ರಕಾರನಿರ್ದೇಶ Packed128<ಭಿನ್ನ> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(ಧ್ರುವ ElementType* address) {
    ಪ್ರತ್ಯಾಗಮ Packed128<ElementType>{*reinterpret_cast<ಧ್ರುವ int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(ಧ್ರುವ ElementType* address) {
    ಪ್ರತ್ಯಾಗಮ Packed128<ElementType>{__ldcs(reinterpret_cast<ಧ್ರುವ int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ ಶೂನ್ಯ store128(ElementType* target, Packed128<ElementType> ಮೌಲ್ಯ) {
    *reinterpret_cast<int4*>(target) = ಮೌಲ್ಯ.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ ಶೂನ್ಯ store128cs(ElementType* target, Packed128<ElementType> ಮೌಲ್ಯ) {
    __stcs(reinterpret_cast<int4*>(target), ಮೌಲ್ಯ.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ ಶೂನ್ಯ store128cg(ElementType* target, Packed128<ElementType> ಮೌಲ್ಯ) {
    __stcg(reinterpret_cast<int4*>(target), ಮೌಲ್ಯ.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#ಯದಿ defined(ENABLE_BF16)

ಪ್ರಕಾರನಿರ್ದೇಶ __nv_bfloat16 floatX;
ಪ್ರಕಾರನಿರ್ದೇಶ __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#ನೋಚೇದ್ಯದಿ defined(ENABLE_FP16)

ಪ್ರಕಾರನಿರ್ದೇಶ half floatX;
ಪ್ರಕಾರನಿರ್ದೇಶ half floatN;

#ಅನ್ಯಥಾ

ಪ್ರಕಾರನಿರ್ದೇಶ ಭಿನ್ನ floatX;
ಪ್ರಕಾರನಿರ್ದೇಶ ಭಿನ್ನ floatN;
#endif

ಪ್ರಕಾರನಿರ್ದೇಶ Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#ಯದಿ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ಧ್ರುವ floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<ಧ್ರುವ unsigned short*>(address));
    ಪ್ರತ್ಯಾಗಮ __nv_bfloat16_raw{bf};
}

__device__ ಶೂನ್ಯ __stcs(floatX* address, floatX ಮೌಲ್ಯ) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)ಮೌಲ್ಯ).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

ಭಿನ್ನ* make_random_float_01(size_t N) {
    ಭಿನ್ನ* arr = (ಭಿನ್ನ*)malloc(N * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತಿ (size_t i = 0; i < N; i++) {
        arr[i] = ((ಭಿನ್ನ)rand() / RAND_MAX); // range 0..1
    }
    ಪ್ರತ್ಯಾಗಮ arr;
}

ಭಿನ್ನ* make_random_float(size_t N) {
    ಭಿನ್ನ* arr = (ಭಿನ್ನ*)malloc(N * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತಿ (size_t i = 0; i < N; i++) {
        arr[i] = ((ಭಿನ್ನ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ಪ್ರತ್ಯಾಗಮ arr;
}

ಪೂರ್ಣಾಙ್ಕ* make_random_int(size_t N, ಪೂರ್ಣಾಙ್ಕ V) {
    ಪೂರ್ಣಾಙ್ಕ* arr = (ಪೂರ್ಣಾಙ್ಕ*)malloc(N * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪ್ರತಿ (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    ಪ್ರತ್ಯಾಗಮ arr;
}

ಭಿನ್ನ* make_zeros_float(size_t N) {
    ಭಿನ್ನ* arr = (ಭಿನ್ನ*)malloc(N * ಪರಿಮಾಣ(ಭಿನ್ನ));
    memset(arr, 0, N * ಪರಿಮಾಣ(ಭಿನ್ನ)); // all zero
    ಪ್ರತ್ಯಾಗಮ arr;
}

ಭಿನ್ನ* make_ones_float(size_t N) {
    ಭಿನ್ನ* arr = (ಭಿನ್ನ*)malloc(N * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತಿ (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    ಪ್ರತ್ಯಾಗಮ arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, ಭಿನ್ನ* h_ptr, size_t ಎಣಿಕೆ) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(ಎಣಿಕೆ * ಪರಿಮಾಣ(TargetType));
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಎಣಿಕೆ; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, ಎಣಿಕೆ * ಪರಿಮಾಣ(TargetType), cudaMemcpyHostToDevice);
    ಮುಕ್ತ(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    ಪ್ರತ್ಯಾಗಮ status;
}

ಶೂನ್ಯ setup_main() {
    srand(0);   // determinism

    // set up the device
    ಪೂರ್ಣಾಙ್ಕ deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_ಅಂಕೆ_SMs = deviceProp.multi_Processor_ಎಣಿಕೆ;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    ಪೂರ್ಣಾಙ್ಕ enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
ಶೂನ್ಯ validate_result(D* device_result, ಧ್ರುವ T* cpu_reference, ಧ್ರುವ ವರ್ಣ* name, std::size_t ಅಂಕೆ_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(ಅಂಕೆ_elements * ಪರಿಮಾಣ(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, ಅಂಕೆ_elements * ಪರಿಮಾಣ(D), cudaMemcpyDeviceToHost));
    ಪೂರ್ಣಾಙ್ಕ nfaults = 0;
#ifndef ENABLE_BF16
    ಭಿನ್ನ epsilon = FLT_EPSILON;
#ಅನ್ಯಥಾ
    ಭಿನ್ನ epsilon = 0.079;
#endif
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_elements; i++) {
        // Skip masked elements
        ಯದಿ(!isfinite(cpu_reference[i]))
            ಅನುವರ್ತ;

        // print the first few comparisons
        ಯದಿ (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        ಭಿನ್ನ t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        ಯದಿ (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            ಯದಿ (nfaults >= 10) {
                ಮುಕ್ತ(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    ಯದಿ (nfaults > 0) {
        ಮುಕ್ತ(out_gpu);
        exit(EXIT_FAILURE);
    }

    ಮುಕ್ತ(out_gpu);
}

template<class Kernel, class... KernelArgs>
ಭಿನ್ನ benchmark_kernel(ಪೂರ್ಣಾಙ್ಕ repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    ಪೂರ್ಣಾಙ್ಕ deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    ಶೂನ್ಯ* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    ಭಿನ್ನ elapsed_time = 0.f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        ಭಿನ್ನ single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    ಪ್ರತ್ಯಾಗಮ elapsed_time / repeats;
}