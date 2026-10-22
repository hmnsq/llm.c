#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <cuda_runtime.h>
#অন্তর্ভাব <cublas_v2.h>
#অন্তর্ভাব <cublasLt.h>
#অন্তর্ভাব <float.h>

#define WARP_SIZE 32U
বাহ্য cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T ভাজক) {
    প্রত্যাগম (dividend + ভাজক-1) / ভাজক;
}

__device__ ভিন্ন warpReduceSum(ভিন্ন val) {
    প্রতি (পূর্ণাঙ্ক offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    প্রত্যাগম val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = ভিন্ন (*) (ভিন্ন);

template<reduction_func_t warp_reduction>
__device__ অন্তঃস্থ ভিন্ন blockReduce(ভিন্ন val, bool final_sync, ভিন্ন out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ ভিন্ন shared_val[WARP_SIZE];
    ধ্রুব পূর্ণাঙ্ক lane_id = threadIdx.x % WARP_SIZE;
    ধ্রুব পূর্ণাঙ্ক warp_id = threadIdx.x / WARP_SIZE;
    ধ্রুব পূর্ণাঙ্ক অঙ্ক_warps = blockDim.x / WARP_SIZE;

    ভিন্ন warp_val = warp_reduction(val);
    যদি (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < অঙ্ক_warps) ? shared_val[lane_id] : out_of_bounds;
    ভিন্ন block_val = warp_reduction(warp_val);

    যদি (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    প্রত্যাগম block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ অন্তঃস্থ ভিন্ন blockReduce(ভিন্ন val) {
    প্রত্যাগম blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
শূন্য cuda_check(cudaError_t error, ধ্রুব বর্ণ *file, পূর্ণাঙ্ক সারি) {
    যদি (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, সারি,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
শূন্য cublasCheck(cublasStatus_t status, ধ্রুব বর্ণ *file, পূর্ণাঙ্ক সারি)
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
স্থির size_t cublaslt_workspace_size = 32 * 1024 * 1024;
স্থির শূন্য* cublaslt_workspace = রিক্ত;
স্থির cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
পূর্ণাঙ্ক cuda_arch_major = 0;
পূর্ণাঙ্ক cuda_arch_minor = 0;
পূর্ণাঙ্ক cuda_অঙ্ক_SMs = 0; // for persistent threads where we want 1 threadblock per SM
পূর্ণাঙ্ক cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

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
সংরচনা alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = স্বভাব;
    __device__ explicit Packed128(int4 bits) {
        static_assert(পরিমাণ(bits) == পরিমাণ(payload), "Size mismatch.");
        memcpy(&payload, &bits, পরিমাণ(bits));
    }

    __device__  স্থির Packed128 constant(ElementType মান) {
        Packed128 result;
        প্রতি(পূর্ণাঙ্ক k = 0; k < size; ++k) {
            result.payload[k] = মান;
        }
        প্রত্যাগম result;
    }

    __device__ স্থির Packed128 zeros() {
        প্রত্যাগম constant(0);
    }

    __device__ স্থির Packed128 ones() {
        প্রত্যাগম constant(1);
    }

    __device__ ElementType& operator[](পূর্ণাঙ্ক index) {
        প্রত্যাগম payload[index];
    }
    __device__ ধ্রুব ElementType& operator[](পূর্ণাঙ্ক index) ধ্রুব {
        প্রত্যাগম payload[index];
    }
    __device__ int4 get_bits() ধ্রুব {
        int4 bits;
        static_assert(পরিমাণ(bits) == পরিমাণ(payload), "Size mismatch.");
        memcpy(&bits, &payload, পরিমাণ(bits));
        প্রত্যাগম bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    স্থির constexpr ধ্রুব পূর্ণাঙ্ক size = পরিমাণ(int4) / পরিমাণ(ElementType);
    ElementType payload[size];
};

// short-form typedef
প্রকারনির্দেশ Packed128<ভিন্ন> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(ধ্রুব ElementType* address) {
    প্রত্যাগম Packed128<ElementType>{*reinterpret_cast<ধ্রুব int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(ধ্রুব ElementType* address) {
    প্রত্যাগম Packed128<ElementType>{__ldcs(reinterpret_cast<ধ্রুব int4*>(address))};
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

প্রকারনির্দেশ __nv_bfloat16 floatX;
প্রকারনির্দেশ __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#নোচেদ্যদি defined(ENABLE_FP16)

প্রকারনির্দেশ half floatX;
প্রকারনির্দেশ half floatN;

#অন্যথা

প্রকারনির্দেশ ভিন্ন floatX;
প্রকারনির্দেশ ভিন্ন floatN;
#endif

প্রকারনির্দেশ Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#যদি defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ধ্রুব floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<ধ্রুব unsigned short*>(address));
    প্রত্যাগম __nv_bfloat16_raw{bf};
}

__device__ শূন্য __stcs(floatX* address, floatX মান) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)মান).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

ভিন্ন* make_random_float_01(size_t N) {
    ভিন্ন* arr = (ভিন্ন*)malloc(N * পরিমাণ(ভিন্ন));
    প্রতি (size_t i = 0; i < N; i++) {
        arr[i] = ((ভিন্ন)rand() / RAND_MAX); // range 0..1
    }
    প্রত্যাগম arr;
}

ভিন্ন* make_random_float(size_t N) {
    ভিন্ন* arr = (ভিন্ন*)malloc(N * পরিমাণ(ভিন্ন));
    প্রতি (size_t i = 0; i < N; i++) {
        arr[i] = ((ভিন্ন)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    প্রত্যাগম arr;
}

পূর্ণাঙ্ক* make_random_int(size_t N, পূর্ণাঙ্ক V) {
    পূর্ণাঙ্ক* arr = (পূর্ণাঙ্ক*)malloc(N * পরিমাণ(পূর্ণাঙ্ক));
    প্রতি (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    প্রত্যাগম arr;
}

ভিন্ন* make_zeros_float(size_t N) {
    ভিন্ন* arr = (ভিন্ন*)malloc(N * পরিমাণ(ভিন্ন));
    memset(arr, 0, N * পরিমাণ(ভিন্ন)); // all zero
    প্রত্যাগম arr;
}

ভিন্ন* make_ones_float(size_t N) {
    ভিন্ন* arr = (ভিন্ন*)malloc(N * পরিমাণ(ভিন্ন));
    প্রতি (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    প্রত্যাগম arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, ভিন্ন* h_ptr, size_t গুনতি) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(গুনতি * পরিমাণ(TargetType));
    প্রতি (পূর্ণাঙ্ক i = 0; i < গুনতি; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, গুনতি * পরিমাণ(TargetType), cudaMemcpyHostToDevice);
    মুক্ত(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    প্রত্যাগম status;
}

শূন্য setup_main() {
    srand(0);   // determinism

    // set up the device
    পূর্ণাঙ্ক deviceIdx = 0;
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
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    পূর্ণাঙ্ক enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
শূন্য validate_result(D* device_result, ধ্রুব T* cpu_reference, ধ্রুব বর্ণ* name, std::size_t অঙ্ক_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(অঙ্ক_elements * পরিমাণ(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, অঙ্ক_elements * পরিমাণ(D), cudaMemcpyDeviceToHost));
    পূর্ণাঙ্ক nfaults = 0;
#ifndef ENABLE_BF16
    ভিন্ন epsilon = FLT_EPSILON;
#অন্যথা
    ভিন্ন epsilon = 0.079;
#endif
    প্রতি (পূর্ণাঙ্ক i = 0; i < অঙ্ক_elements; i++) {
        // Skip masked elements
        যদি(!isfinite(cpu_reference[i]))
            অনুবর্ত;

        // print the first few comparisons
        যদি (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        ভিন্ন t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
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
ভিন্ন benchmark_kernel(পূর্ণাঙ্ক repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    পূর্ণাঙ্ক deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    শূন্য* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    ভিন্ন elapsed_time = 0.f;
    প্রতি (পূর্ণাঙ্ক i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        ভিন্ন single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    প্রত্যাগম elapsed_time / repeats;
}