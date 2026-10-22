#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <cuda_runtime.h>
#അന്തര്ഭാവ <cublas_v2.h>
#അന്തര്ഭാവ <cublasLt.h>
#അന്തര്ഭാവ <float.h>

#define WARP_SIZE 32U
ബാഹ്യ cudaDeviceProp deviceProp;

template<class T>
__host__ __device__ T ceil_div(T dividend, T ഹാരകം) {
    തിരികെ (dividend + ഹാരകം-1) / ഹാരകം;
}

__device__ ദശാംശം warpReduceSum(ദശാംശം val) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    തിരികെ val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = ദശാംശം (*) (ദശാംശം);

template<reduction_func_t warp_reduction>
__device__ വരിയിൽ ദശാംശം blockReduce(ദശാംശം val, bool final_sync, ദശാംശം out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ ദശാംശം shared_val[WARP_SIZE];
    സ്ഥിരം പൂർണ്ണസംഖ്യ lane_id = threadIdx.x % WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ warp_id = threadIdx.x / WARP_SIZE;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_warps = blockDim.x / WARP_SIZE;

    ദശാംശം warp_val = warp_reduction(val);
    എങ്കിൽ (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < അക്കം_warps) ? shared_val[lane_id] : out_of_bounds;
    ദശാംശം block_val = warp_reduction(warp_val);

    എങ്കിൽ (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    തിരികെ block_val;
}

// Helper function to call blockReduce with default arguments
template<reduction_func_t warp_reduction>
__device__ വരിയിൽ ദശാംശം blockReduce(ദശാംശം val) {
    തിരികെ blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
ശൂന്യം cuda_check(cudaError_t error, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    എങ്കിൽ (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, വരി,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
ശൂന്യം cublasCheck(cublasStatus_t status, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി)
{
    എങ്കിൽ (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, വരി);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
സ്ഥിതം size_t cublaslt_workspace_size = 32 * 1024 * 1024;
സ്ഥിതം ശൂന്യം* cublaslt_workspace = ശൂന്യ;
സ്ഥിതം cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
പൂർണ്ണസംഖ്യ cuda_arch_major = 0;
പൂർണ്ണസംഖ്യ cuda_arch_minor = 0;
പൂർണ്ണസംഖ്യ cuda_അക്കം_SMs = 0; // for persistent threads where we want 1 threadblock per SM
പൂർണ്ണസംഖ്യ cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#എങ്കിൽ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#അല്ലെങ്കിൽ
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template<class ElementType>
ഘടന alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = സ്ഥിരസ്ഥിതി;
    __device__ explicit Packed128(int4 bits) {
        static_assert(വലുപ്പം(bits) == വലുപ്പം(payload), "Size mismatch.");
        memcpy(&payload, &bits, വലുപ്പം(bits));
    }

    __device__  സ്ഥിതം Packed128 constant(ElementType മൂല്യം) {
        Packed128 result;
        വേണ്ടി(പൂർണ്ണസംഖ്യ k = 0; k < size; ++k) {
            result.payload[k] = മൂല്യം;
        }
        തിരികെ result;
    }

    __device__ സ്ഥിതം Packed128 zeros() {
        തിരികെ constant(0);
    }

    __device__ സ്ഥിതം Packed128 ones() {
        തിരികെ constant(1);
    }

    __device__ ElementType& operator[](പൂർണ്ണസംഖ്യ index) {
        തിരികെ payload[index];
    }
    __device__ സ്ഥിരം ElementType& operator[](പൂർണ്ണസംഖ്യ index) സ്ഥിരം {
        തിരികെ payload[index];
    }
    __device__ int4 get_bits() സ്ഥിരം {
        int4 bits;
        static_assert(വലുപ്പം(bits) == വലുപ്പം(payload), "Size mismatch.");
        memcpy(&bits, &payload, വലുപ്പം(bits));
        തിരികെ bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    സ്ഥിതം constexpr സ്ഥിരം പൂർണ്ണസംഖ്യ size = വലുപ്പം(int4) / വലുപ്പം(ElementType);
    ElementType payload[size];
};

// short-form typedef
തരനിർവചനം Packed128<ദശാംശം> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(സ്ഥിരം ElementType* address) {
    തിരികെ Packed128<ElementType>{*reinterpret_cast<സ്ഥിരം int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(സ്ഥിരം ElementType* address) {
    തിരികെ Packed128<ElementType>{__ldcs(reinterpret_cast<സ്ഥിരം int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ ശൂന്യം store128(ElementType* target, Packed128<ElementType> മൂല്യം) {
    *reinterpret_cast<int4*>(target) = മൂല്യം.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ ശൂന്യം store128cs(ElementType* target, Packed128<ElementType> മൂല്യം) {
    __stcs(reinterpret_cast<int4*>(target), മൂല്യം.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ ശൂന്യം store128cg(ElementType* target, Packed128<ElementType> മൂല്യം) {
    __stcg(reinterpret_cast<int4*>(target), മൂല്യം.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#എങ്കിൽ defined(ENABLE_BF16)

തരനിർവചനം __nv_bfloat16 floatX;
തരനിർവചനം __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#അല്ലെങ്കിലെങ്കിൽ defined(ENABLE_FP16)

തരനിർവചനം half floatX;
തരനിർവചനം half floatN;

#അല്ലെങ്കിൽ

തരനിർവചനം ദശാംശം floatX;
തരനിർവചനം ദശാംശം floatN;
#endif

തരനിർവചനം Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#എങ്കിൽ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(സ്ഥിരം floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<സ്ഥിരം unsigned short*>(address));
    തിരികെ __nv_bfloat16_raw{bf};
}

__device__ ശൂന്യം __stcs(floatX* address, floatX മൂല്യം) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)മൂല്യം).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

ദശാംശം* make_random_float_01(size_t N) {
    ദശാംശം* arr = (ദശാംശം*)malloc(N * വലുപ്പം(ദശാംശം));
    വേണ്ടി (size_t i = 0; i < N; i++) {
        arr[i] = ((ദശാംശം)rand() / RAND_MAX); // range 0..1
    }
    തിരികെ arr;
}

ദശാംശം* make_random_float(size_t N) {
    ദശാംശം* arr = (ദശാംശം*)malloc(N * വലുപ്പം(ദശാംശം));
    വേണ്ടി (size_t i = 0; i < N; i++) {
        arr[i] = ((ദശാംശം)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    തിരികെ arr;
}

പൂർണ്ണസംഖ്യ* make_random_int(size_t N, പൂർണ്ണസംഖ്യ V) {
    പൂർണ്ണസംഖ്യ* arr = (പൂർണ്ണസംഖ്യ*)malloc(N * വലുപ്പം(പൂർണ്ണസംഖ്യ));
    വേണ്ടി (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    തിരികെ arr;
}

ദശാംശം* make_zeros_float(size_t N) {
    ദശാംശം* arr = (ദശാംശം*)malloc(N * വലുപ്പം(ദശാംശം));
    memset(arr, 0, N * വലുപ്പം(ദശാംശം)); // all zero
    തിരികെ arr;
}

ദശാംശം* make_ones_float(size_t N) {
    ദശാംശം* arr = (ദശാംശം*)malloc(N * വലുപ്പം(ദശാംശം));
    വേണ്ടി (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    തിരികെ arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, ദശാംശം* h_ptr, size_t എണ്ണം) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(എണ്ണം * വലുപ്പം(TargetType));
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < എണ്ണം; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, എണ്ണം * വലുപ്പം(TargetType), cudaMemcpyHostToDevice);
    സ്വതന്ത്രം(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    തിരികെ status;
}

ശൂന്യം setup_main() {
    srand(0);   // determinism

    // set up the device
    പൂർണ്ണസംഖ്യ deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_അക്കം_SMs = deviceProp.multi_Processor_എണ്ണം;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    പൂർണ്ണസംഖ്യ enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
ശൂന്യം validate_result(D* device_result, സ്ഥിരം T* cpu_reference, സ്ഥിരം അക്ഷരം* name, std::size_t അക്കം_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(അക്കം_elements * വലുപ്പം(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, അക്കം_elements * വലുപ്പം(D), cudaMemcpyDeviceToHost));
    പൂർണ്ണസംഖ്യ nfaults = 0;
#ifndef ENABLE_BF16
    ദശാംശം epsilon = FLT_EPSILON;
#അല്ലെങ്കിൽ
    ദശാംശം epsilon = 0.079;
#endif
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_elements; i++) {
        // Skip masked elements
        എങ്കിൽ(!isfinite(cpu_reference[i]))
            തുടരുക;

        // print the first few comparisons
        എങ്കിൽ (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        ദശാംശം t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        എങ്കിൽ (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            എങ്കിൽ (nfaults >= 10) {
                സ്വതന്ത്രം(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    എങ്കിൽ (nfaults > 0) {
        സ്വതന്ത്രം(out_gpu);
        exit(EXIT_FAILURE);
    }

    സ്വതന്ത്രം(out_gpu);
}

template<class Kernel, class... KernelArgs>
ദശാംശം benchmark_kernel(പൂർണ്ണസംഖ്യ repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    പൂർണ്ണസംഖ്യ deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    ശൂന്യം* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    ദശാംശം elapsed_time = 0.f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        ദശാംശം single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    തിരികെ elapsed_time / repeats;
}