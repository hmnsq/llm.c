#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cuda_runtime.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>
#ضمن <float.h>

#define WARP_SIZE 32U
خارجي cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    أرجع (dividend + divisor-1) / divisor;
}

__device__ عائم warpReduceSum(عائم val) {
    لكل (صحيح offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    أرجع val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = عائم (*) (عائم);

template<reduction_func_t warp_reduction>
__device__ مضمن عائم blockReduce(عائم val, bool final_sync, عائم out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ عائم shared_val[WARP_SIZE];
    ثابت صحيح lane_id = threadIdx.x % WARP_SIZE;
    ثابت صحيح warp_id = threadIdx.x / WARP_SIZE;
    ثابت صحيح num_warps = blockDim.x / WARP_SIZE;

    عائم warp_val = warp_reduction(val);
    𐤀𐤌 (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    عائم block_val = warp_reduction(warp_val);

    𐤀𐤌 (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    أرجع block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ مضمن عائم blockReduce(عائم val) {
    أرجع blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
فراغ cuda_check(cudaError_t error, ثابت حرف *file, صحيح line) {
    𐤀𐤌 (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
فراغ cublasCheck(cublasStatus_t status, ثابت حرف *file, صحيح line)
{
    𐤀𐤌 (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ساكن size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ساكن فراغ* cublaslt_workspace = عدم;
ساكن cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
صحيح cuda_arch_major = 0;
صحيح cuda_arch_minor = 0;
صحيح cuda_num_SMs = 0; // for persistent threads where we want 1 threadblock per SM
صحيح cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#𐤀𐤌 __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#وإلا
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
بنية alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = افتراضي;
    __device__ explicit Packed128(int4 bits) {
        static_assert(حجم(bits) == حجم(payload), "Size mismatch.");
        memcpy(&payload, &bits, حجم(bits));
    }

    __device__  ساكن Packed128 constant(ElementType value) {
        Packed128 result;
        لكل(صحيح k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        أرجع result;
    }

    __device__ ساكن Packed128 zeros() {
        أرجع constant(0);
    }

    __device__ ساكن Packed128 ones() {
        أرجع constant(1);
    }

    __device__ ElementType& operator[](صحيح index) {
        أرجع payload[index];
    }
    __device__ ثابت ElementType& operator[](صحيح index) ثابت {
        أرجع payload[index];
    }
    __device__ int4 get_bits() ثابت {
        int4 bits;
        static_assert(حجم(bits) == حجم(payload), "Size mismatch.");
        memcpy(&bits, &payload, حجم(bits));
        أرجع bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ساكن constexpr ثابت صحيح size = حجم(int4) / حجم(ElementType);
    ElementType payload[size];
};

// short-form typedef
تعريفنوع Packed128<عائم> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(ثابت ElementType* address) {
    أرجع Packed128<ElementType>{*reinterpret_cast<ثابت int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(ثابت ElementType* address) {
    أرجع Packed128<ElementType>{__ldcs(reinterpret_cast<ثابت int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ فراغ store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ فراغ store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ فراغ store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#𐤀𐤌 defined(ENABLE_BF16)

تعريفنوع __nv_bfloat16 floatX;
تعريفنوع __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#وإلاإذا defined(ENABLE_FP16)

تعريفنوع half floatX;
تعريفنوع half floatN;

#وإلا

تعريفنوع عائم floatX;
تعريفنوع عائم floatN;
#endif

تعريفنوع Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#𐤀𐤌 defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<ثابت unsigned short*>(address));
    أرجع __nv_bfloat16_raw{bf};
}

__device__ فراغ __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

عائم* make_random_float_01(size_t N) {
    عائم* arr = (عائم*)malloc(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX); // range 0..1
    }
    أرجع arr;
}

عائم* make_random_float(size_t N) {
    عائم* arr = (عائم*)malloc(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    أرجع arr;
}

صحيح* make_random_int(size_t N, صحيح V) {
    صحيح* arr = (صحيح*)malloc(N * حجم(صحيح));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    أرجع arr;
}

عائم* make_zeros_float(size_t N) {
    عائم* arr = (عائم*)malloc(N * حجم(عائم));
    memset(arr, 0, N * حجم(عائم)); // all zero
    أرجع arr;
}

عائم* make_ones_float(size_t N) {
    عائم* arr = (عائم*)malloc(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    أرجع arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, عائم* h_ptr, size_t count) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * حجم(TargetType));
    لكل (صحيح i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * حجم(TargetType), cudaMemcpyHostToDevice);
    حرر(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    أرجع status;
}

فراغ setup_main() {
    srand(0);   // determinism

    // set up the device
    صحيح deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    صحيح enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
فراغ validate_result(D* device_result, ثابت T* cpu_reference, ثابت حرف* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * حجم(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * حجم(D), cudaMemcpyDeviceToHost));
    صحيح nfaults = 0;
#ifndef ENABLE_BF16
    عائم epsilon = FLT_EPSILON;
#وإلا
    عائم epsilon = 0.079;
#endif
    لكل (صحيح i = 0; i < num_elements; i++) {
        // Skip masked elements
        𐤀𐤌(!isfinite(cpu_reference[i]))
            استمر;

        // print the first few comparisons
        𐤀𐤌 (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        عائم t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        𐤀𐤌 (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            𐤀𐤌 (nfaults >= 10) {
                حرر(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    𐤀𐤌 (nfaults > 0) {
        حرر(out_gpu);
        exit(EXIT_FAILURE);
    }

    حرر(out_gpu);
}

template<class Kernel, class... KernelArgs>
عائم benchmark_kernel(صحيح repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    صحيح deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    فراغ* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    عائم elapsed_time = 0.f;
    لكل (صحيح i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        عائم single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    أرجع elapsed_time / repeats;
}