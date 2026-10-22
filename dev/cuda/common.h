#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <cuda_runtime.h>
#અન્તર્ભાવ <cublas_v2.h>
#અન્તર્ભાવ <cublasLt.h>
#અન્તર્ભાવ <float.h>

#define WARP_SIZE 32U
બાહ્ય cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T ભાજક) {
    પાછા (dividend + ભાજક-1) / ભાજક;
}

__device__ દશાંશ warpReduceSum(દશાંશ val) {
    માટે (પૂર્ણાંક offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    પાછા val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = દશાંશ (*) (દશાંશ);

template<reduction_func_t warp_reduction>
__device__ અંતઃરેખ દશાંશ blockReduce(દશાંશ val, bool final_sync, દશાંશ out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ દશાંશ shared_val[WARP_SIZE];
    અચલ પૂર્ણાંક lane_id = threadIdx.x % WARP_SIZE;
    અચલ પૂર્ણાંક warp_id = threadIdx.x / WARP_SIZE;
    અચલ પૂર્ણાંક અંક_warps = blockDim.x / WARP_SIZE;

    દશાંશ warp_val = warp_reduction(val);
    જો (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < અંક_warps) ? shared_val[lane_id] : out_of_bounds;
    દશાંશ block_val = warp_reduction(warp_val);

    જો (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    પાછા block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ અંતઃરેખ દશાંશ blockReduce(દશાંશ val) {
    પાછા blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
શૂન્ય cuda_check(cudaError_t error, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, લીટી,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
શૂન્ય cublasCheck(cublasStatus_t status, અચલ અક્ષર *file, પૂર્ણાંક લીટી)
{
    જો (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, લીટી);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
સ્થિર size_t cublaslt_workspace_size = 32 * 1024 * 1024;
સ્થિર શૂન્ય* cublaslt_workspace = ખાલી;
સ્થિર cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
પૂર્ણાંક cuda_arch_major = 0;
પૂર્ણાંક cuda_arch_minor = 0;
પૂર્ણાંક cuda_અંક_SMs = 0; // for persistent threads where we want 1 threadblock per SM
પૂર્ણાંક cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#જો __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#નહીંતર
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
સંરચના alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = મૂળભૂત;
    __device__ explicit Packed128(int4 bits) {
        static_assert(કદ(bits) == કદ(payload), "Size mismatch.");
        memcpy(&payload, &bits, કદ(bits));
    }

    __device__  સ્થિર Packed128 constant(ElementType કિંમત) {
        Packed128 result;
        માટે(પૂર્ણાંક k = 0; k < size; ++k) {
            result.payload[k] = કિંમત;
        }
        પાછા result;
    }

    __device__ સ્થિર Packed128 zeros() {
        પાછા constant(0);
    }

    __device__ સ્થિર Packed128 ones() {
        પાછા constant(1);
    }

    __device__ ElementType& operator[](પૂર્ણાંક index) {
        પાછા payload[index];
    }
    __device__ અચલ ElementType& operator[](પૂર્ણાંક index) અચલ {
        પાછા payload[index];
    }
    __device__ int4 get_bits() અચલ {
        int4 bits;
        static_assert(કદ(bits) == કદ(payload), "Size mismatch.");
        memcpy(&bits, &payload, કદ(bits));
        પાછા bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    સ્થિર constexpr અચલ પૂર્ણાંક size = કદ(int4) / કદ(ElementType);
    ElementType payload[size];
};

// short-form typedef
પ્રકારવ્યાખ્યા Packed128<દશાંશ> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(અચલ ElementType* address) {
    પાછા Packed128<ElementType>{*reinterpret_cast<અચલ int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(અચલ ElementType* address) {
    પાછા Packed128<ElementType>{__ldcs(reinterpret_cast<અચલ int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ શૂન્ય store128(ElementType* target, Packed128<ElementType> કિંમત) {
    *reinterpret_cast<int4*>(target) = કિંમત.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ શૂન્ય store128cs(ElementType* target, Packed128<ElementType> કિંમત) {
    __stcs(reinterpret_cast<int4*>(target), કિંમત.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ શૂન્ય store128cg(ElementType* target, Packed128<ElementType> કિંમત) {
    __stcg(reinterpret_cast<int4*>(target), કિંમત.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#જો defined(ENABLE_BF16)

પ્રકારવ્યાખ્યા __nv_bfloat16 floatX;
પ્રકારવ્યાખ્યા __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#અથવાજો defined(ENABLE_FP16)

પ્રકારવ્યાખ્યા half floatX;
પ્રકારવ્યાખ્યા half floatN;

#નહીંતર

પ્રકારવ્યાખ્યા દશાંશ floatX;
પ્રકારવ્યાખ્યા દશાંશ floatN;
#endif

પ્રકારવ્યાખ્યા Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#જો defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(અચલ floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<અચલ unsigned short*>(address));
    પાછા __nv_bfloat16_raw{bf};
}

__device__ શૂન્ય __stcs(floatX* address, floatX કિંમત) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)કિંમત).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

દશાંશ* make_random_float_01(size_t N) {
    દશાંશ* arr = (દશાંશ*)malloc(N * કદ(દશાંશ));
    માટે (size_t i = 0; i < N; i++) {
        arr[i] = ((દશાંશ)rand() / RAND_MAX); // range 0..1
    }
    પાછા arr;
}

દશાંશ* make_random_float(size_t N) {
    દશાંશ* arr = (દશાંશ*)malloc(N * કદ(દશાંશ));
    માટે (size_t i = 0; i < N; i++) {
        arr[i] = ((દશાંશ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    પાછા arr;
}

પૂર્ણાંક* make_random_int(size_t N, પૂર્ણાંક V) {
    પૂર્ણાંક* arr = (પૂર્ણાંક*)malloc(N * કદ(પૂર્ણાંક));
    માટે (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    પાછા arr;
}

દશાંશ* make_zeros_float(size_t N) {
    દશાંશ* arr = (દશાંશ*)malloc(N * કદ(દશાંશ));
    memset(arr, 0, N * કદ(દશાંશ)); // all zero
    પાછા arr;
}

દશાંશ* make_ones_float(size_t N) {
    દશાંશ* arr = (દશાંશ*)malloc(N * કદ(દશાંશ));
    માટે (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    પાછા arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, દશાંશ* h_ptr, size_t ગણતરી) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(ગણતરી * કદ(TargetType));
    માટે (પૂર્ણાંક i = 0; i < ગણતરી; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, ગણતરી * કદ(TargetType), cudaMemcpyHostToDevice);
    મુક્ત(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    પાછા status;
}

શૂન્ય setup_main() {
    srand(0);   // determinism

    // set up the device
    પૂર્ણાંક deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_અંક_SMs = deviceProp.multi_Processor_ગણતરી;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    પૂર્ણાંક enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
શૂન્ય validate_result(D* device_result, અચલ T* cpu_reference, અચલ અક્ષર* name, std::size_t અંક_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(અંક_elements * કદ(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, અંક_elements * કદ(D), cudaMemcpyDeviceToHost));
    પૂર્ણાંક nfaults = 0;
#ifndef ENABLE_BF16
    દશાંશ epsilon = FLT_EPSILON;
#નહીંતર
    દશાંશ epsilon = 0.079;
#endif
    માટે (પૂર્ણાંક i = 0; i < અંક_elements; i++) {
        // Skip masked elements
        જો(!isfinite(cpu_reference[i]))
            ચાલુ;

        // print the first few comparisons
        જો (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        દશાંશ t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        જો (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            જો (nfaults >= 10) {
                મુક્ત(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    જો (nfaults > 0) {
        મુક્ત(out_gpu);
        exit(EXIT_FAILURE);
    }

    મુક્ત(out_gpu);
}

template<class Kernel, class... KernelArgs>
દશાંશ benchmark_kernel(પૂર્ણાંક repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    પૂર્ણાંક deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    શૂન્ય* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    દશાંશ elapsed_time = 0.f;
    માટે (પૂર્ણાંક i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        દશાંશ single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    પાછા elapsed_time / repeats;
}