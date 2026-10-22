#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <cuda_runtime.h>
#அந்தர்பாவ <cublas_v2.h>
#அந்தர்பாவ <cublasLt.h>
#அந்தர்பாவ <float.h>

#define WARP_SIZE 32U
வெளி cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T வகுஎண்) {
    திரும்பு (dividend + வகுஎண்-1) / வகுஎண்;
}

__device__ தசம warpReduceSum(தசம val) {
    க்கு (முழு offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    திரும்பு val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = தசம (*) (தசம);

template<reduction_func_t warp_reduction>
__device__ உள்வரி தசம blockReduce(தசம val, bool final_sync, தசம out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ தசம shared_val[WARP_SIZE];
    மாறிலி முழு lane_id = threadIdx.x % WARP_SIZE;
    மாறிலி முழு warp_id = threadIdx.x / WARP_SIZE;
    மாறிலி முழு இலக்கம்_warps = blockDim.x / WARP_SIZE;

    தசம warp_val = warp_reduction(val);
    எனில் (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < இலக்கம்_warps) ? shared_val[lane_id] : out_of_bounds;
    தசம block_val = warp_reduction(warp_val);

    எனில் (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    திரும்பு block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ உள்வரி தசம blockReduce(தசம val) {
    திரும்பு blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
வெறுமை cuda_check(cudaError_t error, மாறிலி எழுத்து *file, முழு வரி) {
    எனில் (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, வரி,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
வெறுமை cublasCheck(cublasStatus_t status, மாறிலி எழுத்து *file, முழு வரி)
{
    எனில் (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, வரி);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
நிலையான size_t cublaslt_workspace_size = 32 * 1024 * 1024;
நிலையான வெறுமை* cublaslt_workspace = வெற்று;
நிலையான cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
முழு cuda_arch_major = 0;
முழு cuda_arch_minor = 0;
முழு cuda_இலக்கம்_SMs = 0; // for persistent threads where we want 1 threadblock per SM
முழு cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#எனில் __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#இல்லையெனில்
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
கட்டமைப்பு alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = இயல்பு;
    __device__ explicit Packed128(int4 bits) {
        static_assert(அளவு(bits) == அளவு(payload), "Size mismatch.");
        memcpy(&payload, &bits, அளவு(bits));
    }

    __device__  நிலையான Packed128 constant(ElementType மதிப்பு) {
        Packed128 result;
        க்கு(முழு k = 0; k < size; ++k) {
            result.payload[k] = மதிப்பு;
        }
        திரும்பு result;
    }

    __device__ நிலையான Packed128 zeros() {
        திரும்பு constant(0);
    }

    __device__ நிலையான Packed128 ones() {
        திரும்பு constant(1);
    }

    __device__ ElementType& operator[](முழு index) {
        திரும்பு payload[index];
    }
    __device__ மாறிலி ElementType& operator[](முழு index) மாறிலி {
        திரும்பு payload[index];
    }
    __device__ int4 get_bits() மாறிலி {
        int4 bits;
        static_assert(அளவு(bits) == அளவு(payload), "Size mismatch.");
        memcpy(&bits, &payload, அளவு(bits));
        திரும்பு bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    நிலையான constexpr மாறிலி முழு size = அளவு(int4) / அளவு(ElementType);
    ElementType payload[size];
};

// short-form typedef
வகைவரை Packed128<தசம> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(மாறிலி ElementType* address) {
    திரும்பு Packed128<ElementType>{*reinterpret_cast<மாறிலி int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(மாறிலி ElementType* address) {
    திரும்பு Packed128<ElementType>{__ldcs(reinterpret_cast<மாறிலி int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ வெறுமை store128(ElementType* target, Packed128<ElementType> மதிப்பு) {
    *reinterpret_cast<int4*>(target) = மதிப்பு.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ வெறுமை store128cs(ElementType* target, Packed128<ElementType> மதிப்பு) {
    __stcs(reinterpret_cast<int4*>(target), மதிப்பு.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ வெறுமை store128cg(ElementType* target, Packed128<ElementType> மதிப்பு) {
    __stcg(reinterpret_cast<int4*>(target), மதிப்பு.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#எனில் defined(ENABLE_BF16)

வகைவரை __nv_bfloat16 floatX;
வகைவரை __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#அல்லதெனில் defined(ENABLE_FP16)

வகைவரை half floatX;
வகைவரை half floatN;

#இல்லையெனில்

வகைவரை தசம floatX;
வகைவரை தசம floatN;
#endif

வகைவரை Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#எனில் defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(மாறிலி floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<மாறிலி unsigned short*>(address));
    திரும்பு __nv_bfloat16_raw{bf};
}

__device__ வெறுமை __stcs(floatX* address, floatX மதிப்பு) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)மதிப்பு).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

தசம* make_random_float_01(size_t N) {
    தசம* arr = (தசம*)malloc(N * அளவு(தசம));
    க்கு (size_t i = 0; i < N; i++) {
        arr[i] = ((தசம)rand() / RAND_MAX); // range 0..1
    }
    திரும்பு arr;
}

தசம* make_random_float(size_t N) {
    தசம* arr = (தசம*)malloc(N * அளவு(தசம));
    க்கு (size_t i = 0; i < N; i++) {
        arr[i] = ((தசம)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    திரும்பு arr;
}

முழு* make_random_int(size_t N, முழு V) {
    முழு* arr = (முழு*)malloc(N * அளவு(முழு));
    க்கு (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    திரும்பு arr;
}

தசம* make_zeros_float(size_t N) {
    தசம* arr = (தசம*)malloc(N * அளவு(தசம));
    memset(arr, 0, N * அளவு(தசம)); // all zero
    திரும்பு arr;
}

தசம* make_ones_float(size_t N) {
    தசம* arr = (தசம*)malloc(N * அளவு(தசம));
    க்கு (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    திரும்பு arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, தசம* h_ptr, size_t மொத்தம்) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(மொத்தம் * அளவு(TargetType));
    க்கு (முழு i = 0; i < மொத்தம்; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, மொத்தம் * அளவு(TargetType), cudaMemcpyHostToDevice);
    விடுவி(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    திரும்பு status;
}

வெறுமை setup_main() {
    srand(0);   // determinism

    // set up the device
    முழு deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_இலக்கம்_SMs = deviceProp.multi_Processor_மொத்தம்;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    முழு enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
வெறுமை validate_result(D* device_result, மாறிலி T* cpu_reference, மாறிலி எழுத்து* name, std::size_t இலக்கம்_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(இலக்கம்_elements * அளவு(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, இலக்கம்_elements * அளவு(D), cudaMemcpyDeviceToHost));
    முழு nfaults = 0;
#ifndef ENABLE_BF16
    தசம epsilon = FLT_EPSILON;
#இல்லையெனில்
    தசம epsilon = 0.079;
#endif
    க்கு (முழு i = 0; i < இலக்கம்_elements; i++) {
        // Skip masked elements
        எனில்(!isfinite(cpu_reference[i]))
            தொடர்;

        // print the first few comparisons
        எனில் (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        தசம t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        எனில் (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            எனில் (nfaults >= 10) {
                விடுவி(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    எனில் (nfaults > 0) {
        விடுவி(out_gpu);
        exit(EXIT_FAILURE);
    }

    விடுவி(out_gpu);
}

template<class Kernel, class... KernelArgs>
தசம benchmark_kernel(முழு repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    முழு deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    வெறுமை* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    தசம elapsed_time = 0.f;
    க்கு (முழு i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        தசம single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    திரும்பு elapsed_time / repeats;
}