#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <cuda_runtime.h>
#অন্তর্ভাব <cublas_v2.h>
#অন্তর্ভাব <cublasLt.h>
#অন্তর্ভাব <float.h>

#define WARP_আকার 32U
বাহ্যিক cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T ভাজক) {
    ফেরত (dividend + ভাজক-1) / ভাজক;
}

__device__ দশমিক warp_Reduce_যোগ(দশমিক val) {
    জন্য (পূর্ণসংখ্যা offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    ফেরত val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = দশমিক (*) (দশমিক);

template<reduction_func_t warp_reduction>
__device__ অন্তঃরেখ দশমিক blockReduce(দশমিক val, bool final_sync, দশমিক out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ দশমিক shared_val[WARP_আকার];
    ধ্রুবক পূর্ণসংখ্যা lane_id = threadIdx.x % WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা warp_id = threadIdx.x / WARP_আকার;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_warps = blockDim.x / WARP_আকার;

    দশমিক warp_val = warp_reduction(val);
    যদি (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < অঙ্ক_warps) ? shared_val[lane_id] : out_of_bounds;
    দশমিক block_val = warp_reduction(warp_val);

    যদি (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    ফেরত block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ অন্তঃরেখ দশমিক blockReduce(দশমিক val) {
    ফেরত blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
শূন্য cuda_check(cudaError_t error, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    যদি (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, সারি,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
শূন্য cublasCheck(cublasStatus_t status, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি)
{
    যদি (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, সারি);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
স্থির size_t cublaslt_workspace_আকার = 32 * 1024 * 1024;
স্থির শূন্য* cublaslt_workspace = ফাঁকা;
স্থির cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
পূর্ণসংখ্যা cuda_arch_major = 0;
পূর্ণসংখ্যা cuda_arch_minor = 0;
পূর্ণসংখ্যা cuda_অঙ্ক_SMs = 0; // for persistent threads where we want 1 threadblock per SM
পূর্ণসংখ্যা cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#যদি __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#অন্যথা
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
কাঠামো alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = পূর্বনির্ধারিত;
    __device__ explicit Packed128(int4 bits) {
        static_assert(আকার(bits) == আকার(payload), "Size mismatch.");
        memcpy(&payload, &bits, আকার(bits));
    }

    __device__  স্থির Packed128 constant(ElementType মান) {
        Packed128 ফলাফল;
        জন্য(পূর্ণসংখ্যা k = 0; k < আকার_2; ++k) {
            ফলাফল.payload[k] = মান;
        }
        ফেরত ফলাফল;
    }

    __device__ স্থির Packed128 zeros() {
        ফেরত constant(0);
    }

    __device__ স্থির Packed128 ones() {
        ফেরত constant(1);
    }

    __device__ ElementType& operator[](পূর্ণসংখ্যা সূচক) {
        ফেরত payload[সূচক];
    }
    __device__ ধ্রুবক ElementType& operator[](পূর্ণসংখ্যা সূচক) ধ্রুবক {
        ফেরত payload[সূচক];
    }
    __device__ int4 get_bits() ধ্রুবক {
        int4 bits;
        static_assert(আকার(bits) == আকার(payload), "Size mismatch.");
        memcpy(&bits, &payload, আকার(bits));
        ফেরত bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    স্থির constexpr ধ্রুবক পূর্ণসংখ্যা আকার_2 = আকার(int4) / আকার(ElementType);
    ElementType payload[আকার_2];
};

// short-form typedef
প্রকারলেখ Packed128<দশমিক> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(ধ্রুবক ElementType* address) {
    ফেরত Packed128<ElementType>{*reinterpret_cast<ধ্রুবক int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(ধ্রুবক ElementType* address) {
    ফেরত Packed128<ElementType>{__ldcs(reinterpret_cast<ধ্রুবক int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ শূন্য store128(ElementType* target, Packed128<ElementType> মান) {
    *reinterpret_cast<int4*>(target) = মান.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ শূন্য store128cs(ElementType* target, Packed128<ElementType> মান) {
    __stcs(reinterpret_cast<int4*>(target), মান.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ শূন্য store128cg(ElementType* target, Packed128<ElementType> মান) {
    __stcg(reinterpret_cast<int4*>(target), মান.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#যদি defined(ENABLE_BF16)

প্রকারলেখ __nv_bfloat16 floatX;
প্রকারলেখ __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#নতুবা_যদি defined(ENABLE_FP16)

প্রকারলেখ half floatX;
প্রকারলেখ half floatN;

#অন্যথা

প্রকারলেখ দশমিক floatX;
প্রকারলেখ দশমিক floatN;
#endif

প্রকারলেখ Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
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
// random utils

দশমিক* make_random_float_01(size_t N) {
    দশমিক* arr = (দশমিক*)malloc(N * আকার(দশমিক));
    জন্য (size_t i = 0; i < N; i++) {
        arr[i] = ((দশমিক)rand() / RAND_MAX); // range 0..1
    }
    ফেরত arr;
}

দশমিক* make_random_float(size_t N) {
    দশমিক* arr = (দশমিক*)malloc(N * আকার(দশমিক));
    জন্য (size_t i = 0; i < N; i++) {
        arr[i] = ((দশমিক)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ফেরত arr;
}

পূর্ণসংখ্যা* make_random_int(size_t N, পূর্ণসংখ্যা V) {
    পূর্ণসংখ্যা* arr = (পূর্ণসংখ্যা*)malloc(N * আকার(পূর্ণসংখ্যা));
    জন্য (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    ফেরত arr;
}

দশমিক* make_zeros_float(size_t N) {
    দশমিক* arr = (দশমিক*)malloc(N * আকার(দশমিক));
    memset(arr, 0, N * আকার(দশমিক)); // all zero
    ফেরত arr;
}

দশমিক* make_ones_float(size_t N) {
    দশমিক* arr = (দশমিক*)malloc(N * আকার(দশমিক));
    জন্য (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    ফেরত arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, দশমিক* h_ptr, size_t গুনতি) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(গুনতি * আকার(TargetType));
    জন্য (পূর্ণসংখ্যা i = 0; i < গুনতি; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, গুনতি * আকার(TargetType), cudaMemcpyHostToDevice);
    মুক্ত(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    ফেরত status;
}

শূন্য setup_main() {
    srand(0);   // determinism

    // set up the device
    পূর্ণসংখ্যা deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_অঙ্ক_SMs = deviceProp.multi_Processor_গুনতি;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_আকার));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    পূর্ণসংখ্যা enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
শূন্য validate_ফলাফল(D* device_ফলাফল, ধ্রুবক T* cpu_reference, ধ্রুবক অক্ষর* name, std::size_t অঙ্ক_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(অঙ্ক_elements * আকার(D));
    cudaCheck(cudaMemcpy(out_gpu, device_ফলাফল, অঙ্ক_elements * আকার(D), cudaMemcpyDeviceToHost));
    পূর্ণসংখ্যা nfaults = 0;
#ifndef ENABLE_BF16
    দশমিক epsilon = FLT_EPSILON;
#অন্যথা
    দশমিক epsilon = 0.079;
#endif
    জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_elements; i++) {
        // Skip masked elements
        যদি(!isfinite(cpu_reference[i]))
            চালু;

        // print the first few comparisons
        যদি (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        দশমিক t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        যদি (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            যদি (nfaults >= 10) {
                মুক্ত(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    যদি (nfaults > 0) {
        মুক্ত(out_gpu);
        exit(EXIT_FAILURE);
    }

    মুক্ত(out_gpu);
}

template<class Kernel, class... KernelArgs>
দশমিক benchmark_kernel(পূর্ণসংখ্যা repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    পূর্ণসংখ্যা deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    শূন্য* flush_বাফার;
    cudaCheck(cudaMalloc(&flush_বাফার, deviceProp.l2_Cache_আকার));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    দশমিক elapsed_time = 0.f;
    জন্য (পূর্ণসংখ্যা i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_বাফার, 0, deviceProp.l2_Cache_আকার));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        দশমিক single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_বাফার));

    ফেরত elapsed_time / repeats;
}