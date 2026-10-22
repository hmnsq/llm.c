#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <cuda_runtime.h>
#ଅନ୍ତର୍ଭାଵ <cublas_v2.h>
#ଅନ୍ତର୍ଭାଵ <cublasLt.h>
#ଅନ୍ତର୍ଭାଵ <float.h>

#define WARP_SIZE 32U
ବାହ୍ୟ cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T ଭାଜକ) {
    ଫେରନ୍ତୁ (dividend + ଭାଜକ-1) / ଭାଜକ;
}

__device__ ଦଶମିକ warpReduceSum(ଦଶମିକ val) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    ଫେରନ୍ତୁ val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = ଦଶମିକ (*) (ଦଶମିକ);

template<reduction_func_t warp_reduction>
__device__ ଧାଡ଼ିଭିତରେ ଦଶମିକ blockReduce(ଦଶମିକ val, bool final_sync, ଦଶମିକ out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ ଦଶମିକ shared_val[WARP_SIZE];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane_id = threadIdx.x % WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_id = threadIdx.x / WARP_SIZE;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_warps = blockDim.x / WARP_SIZE;

    ଦଶମିକ warp_val = warp_reduction(val);
    ଯଦି (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < ଅଙ୍କ_warps) ? shared_val[lane_id] : out_of_bounds;
    ଦଶମିକ block_val = warp_reduction(warp_val);

    ଯଦି (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    ଫେରନ୍ତୁ block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ ଧାଡ଼ିଭିତରେ ଦଶମିକ blockReduce(ଦଶମିକ val) {
    ଫେରନ୍ତୁ blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
ଶୂନ୍ୟ cuda_check(cudaError_t error, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    ଯଦି (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, ଧାଡ଼ି,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
ଶୂନ୍ୟ cublasCheck(cublasStatus_t status, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି)
{
    ଯଦି (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, ଧାଡ଼ି);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ସ୍ଥିର size_t cublaslt_workspace_size = 32 * 1024 * 1024;
ସ୍ଥିର ଶୂନ୍ୟ* cublaslt_workspace = ଖାଲି;
ସ୍ଥିର cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା cuda_arch_major = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା cuda_arch_minor = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା cuda_ଅଙ୍କ_SMs = 0; // for persistent threads where we want 1 threadblock per SM
ପୂର୍ଣ୍ଣସଂଖ୍ୟା cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#ଯଦି __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#ଅନ୍ୟଥା
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
ସଂରଚନା alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;
    __device__ explicit Packed128(int4 bits) {
        static_assert(ଆକାର(bits) == ଆକାର(payload), "Size mismatch.");
        memcpy(&payload, &bits, ଆକାର(bits));
    }

    __device__  ସ୍ଥିର Packed128 constant(ElementType ମୂଲ୍ୟ) {
        Packed128 result;
        ପାଇଁ(ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < size; ++k) {
            result.payload[k] = ମୂଲ୍ୟ;
        }
        ଫେରନ୍ତୁ result;
    }

    __device__ ସ୍ଥିର Packed128 zeros() {
        ଫେରନ୍ତୁ constant(0);
    }

    __device__ ସ୍ଥିର Packed128 ones() {
        ଫେରନ୍ତୁ constant(1);
    }

    __device__ ElementType& operator[](ପୂର୍ଣ୍ଣସଂଖ୍ୟା index) {
        ଫେରନ୍ତୁ payload[index];
    }
    __device__ ଅଚଳ ElementType& operator[](ପୂର୍ଣ୍ଣସଂଖ୍ୟା index) ଅଚଳ {
        ଫେରନ୍ତୁ payload[index];
    }
    __device__ int4 get_bits() ଅଚଳ {
        int4 bits;
        static_assert(ଆକାର(bits) == ଆକାର(payload), "Size mismatch.");
        memcpy(&bits, &payload, ଆକାର(bits));
        ଫେରନ୍ତୁ bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ସ୍ଥିର constexpr ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା size = ଆକାର(int4) / ଆକାର(ElementType);
    ElementType payload[size];
};

// short-form typedef
ପ୍ରକାରବ୍ୟାଖ୍ୟା Packed128<ଦଶମିକ> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(ଅଚଳ ElementType* address) {
    ଫେରନ୍ତୁ Packed128<ElementType>{*reinterpret_cast<ଅଚଳ int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(ଅଚଳ ElementType* address) {
    ଫେରନ୍ତୁ Packed128<ElementType>{__ldcs(reinterpret_cast<ଅଚଳ int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ ଶୂନ୍ୟ store128(ElementType* target, Packed128<ElementType> ମୂଲ୍ୟ) {
    *reinterpret_cast<int4*>(target) = ମୂଲ୍ୟ.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ ଶୂନ୍ୟ store128cs(ElementType* target, Packed128<ElementType> ମୂଲ୍ୟ) {
    __stcs(reinterpret_cast<int4*>(target), ମୂଲ୍ୟ.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ ଶୂନ୍ୟ store128cg(ElementType* target, Packed128<ElementType> ମୂଲ୍ୟ) {
    __stcg(reinterpret_cast<int4*>(target), ମୂଲ୍ୟ.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#ଯଦି defined(ENABLE_BF16)

ପ୍ରକାରବ୍ୟାଖ୍ୟା __nv_bfloat16 floatX;
ପ୍ରକାରବ୍ୟାଖ୍ୟା __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#ନଚେତଯଦି defined(ENABLE_FP16)

ପ୍ରକାରବ୍ୟାଖ୍ୟା half floatX;
ପ୍ରକାରବ୍ୟାଖ୍ୟା half floatN;

#ଅନ୍ୟଥା

ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦଶମିକ floatX;
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦଶମିକ floatN;
#endif

ପ୍ରକାରବ୍ୟାଖ୍ୟା Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#ଯଦି defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ଅଚଳ floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<ଅଚଳ unsigned short*>(address));
    ଫେରନ୍ତୁ __nv_bfloat16_raw{bf};
}

__device__ ଶୂନ୍ୟ __stcs(floatX* address, floatX ମୂଲ୍ୟ) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)ମୂଲ୍ୟ).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

ଦଶମିକ* make_random_float_01(size_t N) {
    ଦଶମିକ* arr = (ଦଶମିକ*)malloc(N * ଆକାର(ଦଶମିକ));
    ପାଇଁ (size_t i = 0; i < N; i++) {
        arr[i] = ((ଦଶମିକ)rand() / RAND_MAX); // range 0..1
    }
    ଫେରନ୍ତୁ arr;
}

ଦଶମିକ* make_random_float(size_t N) {
    ଦଶମିକ* arr = (ଦଶମିକ*)malloc(N * ଆକାର(ଦଶମିକ));
    ପାଇଁ (size_t i = 0; i < N; i++) {
        arr[i] = ((ଦଶମିକ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ଫେରନ୍ତୁ arr;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା* make_random_int(size_t N, ପୂର୍ଣ୍ଣସଂଖ୍ୟା V) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା* arr = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା*)malloc(N * ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ପାଇଁ (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    ଫେରନ୍ତୁ arr;
}

ଦଶମିକ* make_zeros_float(size_t N) {
    ଦଶମିକ* arr = (ଦଶମିକ*)malloc(N * ଆକାର(ଦଶମିକ));
    memset(arr, 0, N * ଆକାର(ଦଶମିକ)); // all zero
    ଫେରନ୍ତୁ arr;
}

ଦଶମିକ* make_ones_float(size_t N) {
    ଦଶମିକ* arr = (ଦଶମିକ*)malloc(N * ଆକାର(ଦଶମିକ));
    ପାଇଁ (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    ଫେରନ୍ତୁ arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, ଦଶମିକ* h_ptr, size_t ଗଣନା_2) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(ଗଣନା_2 * ଆକାର(TargetType));
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଗଣନା_2; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, ଗଣନା_2 * ଆକାର(TargetType), cudaMemcpyHostToDevice);
    ମୁକ୍ତ(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    ଫେରନ୍ତୁ status;
}

ଶୂନ୍ୟ setup_main() {
    srand(0);   // determinism

    // set up the device
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_ଅଙ୍କ_SMs = deviceProp.multi_Processor_ଗଣନା;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
ଶୂନ୍ୟ validate_result(D* device_result, ଅଚଳ T* cpu_reference, ଅଚଳ ଅକ୍ଷର* name, std::size_t ଅଙ୍କ_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(ଅଙ୍କ_elements * ଆକାର(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, ଅଙ୍କ_elements * ଆକାର(D), cudaMemcpyDeviceToHost));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nfaults = 0;
#ifndef ENABLE_BF16
    ଦଶମିକ epsilon = FLT_EPSILON;
#ଅନ୍ୟଥା
    ଦଶମିକ epsilon = 0.079;
#endif
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_elements; i++) {
        // Skip masked elements
        ଯଦି(!isfinite(cpu_reference[i]))
            ଜାରି;

        // print the first few comparisons
        ଯଦି (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        ଦଶମିକ t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        ଯଦି (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            ଯଦି (nfaults >= 10) {
                ମୁକ୍ତ(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    ଯଦି (nfaults > 0) {
        ମୁକ୍ତ(out_gpu);
        exit(EXIT_FAILURE);
    }

    ମୁକ୍ତ(out_gpu);
}

template<class Kernel, class... KernelArgs>
ଦଶମିକ benchmark_kernel(ପୂର୍ଣ୍ଣସଂଖ୍ୟା repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    ଶୂନ୍ୟ* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    ଦଶମିକ elapsed_time = 0.f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        ଦଶମିକ single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    ଫେରନ୍ତୁ elapsed_time / repeats;
}