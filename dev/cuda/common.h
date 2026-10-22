#شامل <stdlib.h>
#شامل <stdio.h>
#شامل <cuda_runtime.h>
#شامل <cublas_v2.h>
#شامل <cublasLt.h>
#شامل <float.h>

#define WARP_SIZE 32U
بیرونی cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T مقسوم_علیه) {
    بازگشت (dividend + مقسوم_علیه-1) / مقسوم_علیه;
}

__device__ اعشاری warpReduceSum(اعشاری val) {
    برای (صحیح offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    بازگشت val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = اعشاری (*) (اعشاری);

template<reduction_func_t warp_reduction>
__device__ درونخطی اعشاری blockReduce(اعشاری val, bool final_sync, اعشاری out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ اعشاری shared_val[WARP_SIZE];
    ثابت صحیح lane_id = threadIdx.x % WARP_SIZE;
    ثابت صحیح warp_id = threadIdx.x / WARP_SIZE;
    ثابت صحیح رقم_warps = blockDim.x / WARP_SIZE;

    اعشاری warp_val = warp_reduction(val);
    اگر (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < رقم_warps) ? shared_val[lane_id] : out_of_bounds;
    اعشاری block_val = warp_reduction(warp_val);

    اگر (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    بازگشت block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ درونخطی اعشاری blockReduce(اعشاری val) {
    بازگشت blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
تهی cuda_check(cudaError_t error, ثابت نویسه *file, صحیح سطر) {
    اگر (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
تهی cublasCheck(cublasStatus_t status, ثابت نویسه *file, صحیح سطر)
{
    اگر (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, سطر);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ایستا size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ایستا تهی* cublaslt_workspace = پوچ;
ایستا cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
صحیح cuda_arch_major = 0;
صحیح cuda_arch_minor = 0;
صحیح cuda_رقم_SMs = 0; // for persistent threads where we want 1 threadblock per SM
صحیح cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#اگر __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#وگرنه
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
ساختار alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = پیشفرض;
    __device__ explicit Packed128(int4 bits) {
        static_assert(اندازه(bits) == اندازه(payload), "Size mismatch.");
        memcpy(&payload, &bits, اندازه(bits));
    }

    __device__  ایستا Packed128 constant(ElementType مقدار) {
        Packed128 result;
        برای(صحیح k = 0; k < size; ++k) {
            result.payload[k] = مقدار;
        }
        بازگشت result;
    }

    __device__ ایستا Packed128 zeros() {
        بازگشت constant(0);
    }

    __device__ ایستا Packed128 ones() {
        بازگشت constant(1);
    }

    __device__ ElementType& operator[](صحیح index) {
        بازگشت payload[index];
    }
    __device__ ثابت ElementType& operator[](صحیح index) ثابت {
        بازگشت payload[index];
    }
    __device__ int4 get_bits() ثابت {
        int4 bits;
        static_assert(اندازه(bits) == اندازه(payload), "Size mismatch.");
        memcpy(&bits, &payload, اندازه(bits));
        بازگشت bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ایستا constexpr ثابت صحیح size = اندازه(int4) / اندازه(ElementType);
    ElementType payload[size];
};

// short-form typedef
تعریفنوع Packed128<اعشاری> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(ثابت ElementType* address) {
    بازگشت Packed128<ElementType>{*reinterpret_cast<ثابت int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(ثابت ElementType* address) {
    بازگشت Packed128<ElementType>{__ldcs(reinterpret_cast<ثابت int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ تهی store128(ElementType* target, Packed128<ElementType> مقدار) {
    *reinterpret_cast<int4*>(target) = مقدار.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ تهی store128cs(ElementType* target, Packed128<ElementType> مقدار) {
    __stcs(reinterpret_cast<int4*>(target), مقدار.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ تهی store128cg(ElementType* target, Packed128<ElementType> مقدار) {
    __stcg(reinterpret_cast<int4*>(target), مقدار.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#اگر defined(ENABLE_BF16)

تعریفنوع __nv_bfloat16 floatX;
تعریفنوع __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#وگرنهاگر defined(ENABLE_FP16)

تعریفنوع half floatX;
تعریفنوع half floatN;

#وگرنه

تعریفنوع اعشاری floatX;
تعریفنوع اعشاری floatN;
#endif

تعریفنوع Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#اگر defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<ثابت unsigned short*>(address));
    بازگشت __nv_bfloat16_raw{bf};
}

__device__ تهی __stcs(floatX* address, floatX مقدار) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)مقدار).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

اعشاری* make_random_float_01(size_t N) {
    اعشاری* arr = (اعشاری*)malloc(N * اندازه(اعشاری));
    برای (size_t i = 0; i < N; i++) {
        arr[i] = ((اعشاری)rand() / RAND_MAX); // range 0..1
    }
    بازگشت arr;
}

اعشاری* make_random_float(size_t N) {
    اعشاری* arr = (اعشاری*)malloc(N * اندازه(اعشاری));
    برای (size_t i = 0; i < N; i++) {
        arr[i] = ((اعشاری)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    بازگشت arr;
}

صحیح* make_random_int(size_t N, صحیح V) {
    صحیح* arr = (صحیح*)malloc(N * اندازه(صحیح));
    برای (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    بازگشت arr;
}

اعشاری* make_zeros_float(size_t N) {
    اعشاری* arr = (اعشاری*)malloc(N * اندازه(اعشاری));
    memset(arr, 0, N * اندازه(اعشاری)); // all zero
    بازگشت arr;
}

اعشاری* make_ones_float(size_t N) {
    اعشاری* arr = (اعشاری*)malloc(N * اندازه(اعشاری));
    برای (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    بازگشت arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, اعشاری* h_ptr, size_t شمار) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(شمار * اندازه(TargetType));
    برای (صحیح i = 0; i < شمار; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, شمار * اندازه(TargetType), cudaMemcpyHostToDevice);
    آزاد(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    بازگشت status;
}

تهی setup_main() {
    srand(0);   // determinism

    // set up the device
    صحیح deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_رقم_SMs = deviceProp.multi_Processor_شمار;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    صحیح enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
تهی validate_result(D* device_result, ثابت T* cpu_reference, ثابت نویسه* name, std::size_t رقم_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(رقم_elements * اندازه(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, رقم_elements * اندازه(D), cudaMemcpyDeviceToHost));
    صحیح nfaults = 0;
#ifndef ENABLE_BF16
    اعشاری epsilon = FLT_EPSILON;
#وگرنه
    اعشاری epsilon = 0.079;
#endif
    برای (صحیح i = 0; i < رقم_elements; i++) {
        // Skip masked elements
        اگر(!isfinite(cpu_reference[i]))
            ادامه;

        // print the first few comparisons
        اگر (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        اعشاری t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        اگر (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            اگر (nfaults >= 10) {
                آزاد(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    اگر (nfaults > 0) {
        آزاد(out_gpu);
        exit(EXIT_FAILURE);
    }

    آزاد(out_gpu);
}

template<class Kernel, class... KernelArgs>
اعشاری benchmark_kernel(صحیح repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    صحیح deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    تهی* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    اعشاری elapsed_time = 0.f;
    برای (صحیح i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        اعشاری single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    بازگشت elapsed_time / repeats;
}