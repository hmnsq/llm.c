#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_runtime.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cublas_v2.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cublasLt.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WARP_ਮਾਪ 32U
ਬਾਹਰੀ cudaDeviceProp deviceProp;

ਸਾਂਚਾ<class T>
__host__ __device__ T ceil_div(T dividend, T ਭਾਜਕ) {
    ਵਾਪਸ (dividend + ਭਾਜਕ-1) / ਭਾਜਕ;
}

__device__ ਦਸ਼ਮਲਵ warp_Reduce_ਜੋੜ(ਦਸ਼ਮਲਵ val) {
    ਲਈ (ਪੂਰਨ ਵਿੱਥ = 16; ਵਿੱਥ > 0; ਵਿੱਥ /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, ਵਿੱਥ);
    }
    ਵਾਪਸ val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = ਦਸ਼ਮਲਵ (*) (ਦਸ਼ਮਲਵ);

ਸਾਂਚਾ<reduction_func_t warp_reduction>
__device__ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ਖੰਡ_Reduce(ਦਸ਼ਮਲਵ val, bool final_sync, ਦਸ਼ਮਲਵ ਬਾਹਰਲਾ_ਦਾ_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ ਦਸ਼ਮਲਵ shared_val[WARP_ਮਾਪ];
    ਅਚਲ ਪੂਰਨ lane_id = threadIdx.x % WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ warp_id = threadIdx.x / WARP_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_warps = ਖੰਡ_Dim.x / WARP_ਮਾਪ;

    ਦਸ਼ਮਲਵ warp_val = warp_reduction(val);
    ਜੇ (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < ਅੰਕ_warps) ? shared_val[lane_id] : ਬਾਹਰਲਾ_ਦਾ_bounds;
    ਦਸ਼ਮਲਵ ਖੰਡ_val = warp_reduction(warp_val);

    ਜੇ (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    ਵਾਪਸ ਖੰਡ_val;
}

// Helper function to call blockReduce with default arguments
ਸਾਂਚਾ<reduction_func_t warp_reduction>
__device__ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ਖੰਡ_Reduce(ਦਸ਼ਮਲਵ val) {
    ਵਾਪਸ ਖੰਡ_Reduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
ਖਾਲੀ cuda_check(cudaError_t error, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (error != cudaSuccess) {
        ਲਿਖੋ("[CUDA ERROR] at file %s:%d:\n%s\n", file, ਸਤਰ,
               cudaGetErrorString(error));
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
};
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
ਖਾਲੀ cublasCheck(cublas_ਸਥਿਤੀ_t ਸਥਿਤੀ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ)
{
    ਜੇ (ਸਥਿਤੀ != CUBLAS_ਸਥਿਤੀ_SUCCESS) {
        ਲਿਖੋ("[cuBLAS ERROR]: %d %s %d\n", ਸਥਿਤੀ, file, ਸਤਰ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasCheck(ਸਥਿਤੀ) { cublasCheck((ਸਥਿਤੀ), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ਸਥਿਰ size_t cublaslt_workspace_ਮਾਪ = 32 * 1024 * 1024;
ਸਥਿਰ ਖਾਲੀ* cublaslt_workspace = ਨਿਰਾ;
ਸਥਿਰ cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
ਪੂਰਨ cuda_arch_major = 0;
ਪੂਰਨ cuda_arch_minor = 0;
ਪੂਰਨ cuda_ਅੰਕ_SMs = 0; // for persistent threads where we want 1 threadblock per SM
ਪੂਰਨ cuda_ਧਾਗੇ_ਪ੍ਰਤੀ_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#ਜੇ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_1024_ਧਾਗੇ_ਖੰਡਾਂ 2
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_1024_ਧਾਗੇ_ਖੰਡਾਂ 1
#ਜੇ_ਅੰਤ

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

ਸਾਂਚਾ<class ElementType>
ਬਣਤਰ alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = ਮੂਲ;
    __device__ explicit Packed128(int4 ਬਿੱਟ) {
        static_assert(ਆਕਾਰ(ਬਿੱਟ) == ਆਕਾਰ(payload), "Size mismatch.");
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&payload, &ਬਿੱਟ, ਆਕਾਰ(ਬਿੱਟ));
    }

    __device__  ਸਥਿਰ Packed128 constant(ElementType ਮੁੱਲ) {
        Packed128 ਨਤੀਜਾ;
        ਲਈ(ਪੂਰਨ k = 0; k < ਮਾਪ; ++k) {
            ਨਤੀਜਾ.payload[k] = ਮੁੱਲ;
        }
        ਵਾਪਸ ਨਤੀਜਾ;
    }

    __device__ ਸਥਿਰ Packed128 zeros() {
        ਵਾਪਸ constant(0);
    }

    __device__ ਸਥਿਰ Packed128 ones() {
        ਵਾਪਸ constant(1);
    }

    __device__ ElementType& ਸੰਚਾਲਕ[](ਪੂਰਨ ਸੂਚਕਾਂਕ) {
        ਵਾਪਸ payload[ਸੂਚਕਾਂਕ];
    }
    __device__ ਅਚਲ ElementType& ਸੰਚਾਲਕ[](ਪੂਰਨ ਸੂਚਕਾਂਕ) ਅਚਲ {
        ਵਾਪਸ payload[ਸੂਚਕਾਂਕ];
    }
    __device__ int4 get_ਬਿੱਟ() ਅਚਲ {
        int4 ਬਿੱਟ;
        static_assert(ਆਕਾਰ(ਬਿੱਟ) == ਆਕਾਰ(payload), "Size mismatch.");
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਿੱਟ, &payload, ਆਕਾਰ(ਬਿੱਟ));
        ਵਾਪਸ ਬਿੱਟ;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ਸਥਿਰ constexpr ਅਚਲ ਪੂਰਨ ਮਾਪ = ਆਕਾਰ(int4) / ਆਕਾਰ(ElementType);
    ElementType payload[ਮਾਪ];
};

// short-form typedef
ਕਿਸਮਪਰਿਭਾਸ਼ਾ Packed128<ਦਸ਼ਮਲਵ> f128;

// load a Packed128 from an aligned memory address
ਸਾਂਚਾ<class ElementType>
__device__ Packed128<ElementType> load128(ਅਚਲ ElementType* ਪਤਾ) {
    ਵਾਪਸ Packed128<ElementType>{*reinterpret_cast<ਅਚਲ int4*>(ਪਤਾ)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
ਸਾਂਚਾ<class ElementType>
__device__ Packed128<ElementType> load128cs(ਅਚਲ ElementType* ਪਤਾ) {
    ਵਾਪਸ Packed128<ElementType>{__ldcs(reinterpret_cast<ਅਚਲ int4*>(ਪਤਾ))};
}
// store a Packed128 to an aligned memory address
ਸਾਂਚਾ<class ElementType>
__device__ ਖਾਲੀ store128(ElementType* ਨਿਸ਼ਾਨਾ, Packed128<ElementType> ਮੁੱਲ) {
    *reinterpret_cast<int4*>(ਨਿਸ਼ਾਨਾ) = ਮੁੱਲ.get_ਬਿੱਟ();
}
// store a Packed128 to an aligned memory address with streaming cache hint
ਸਾਂਚਾ<class ElementType>
__device__ ਖਾਲੀ store128cs(ElementType* ਨਿਸ਼ਾਨਾ, Packed128<ElementType> ਮੁੱਲ) {
    __stcs(reinterpret_cast<int4*>(ਨਿਸ਼ਾਨਾ), ਮੁੱਲ.get_ਬਿੱਟ());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
ਸਾਂਚਾ<class ElementType>
__device__ ਖਾਲੀ store128cg(ElementType* ਨਿਸ਼ਾਨਾ, Packed128<ElementType> ਮੁੱਲ) {
    __stcg(reinterpret_cast<int4*>(ਨਿਸ਼ਾਨਾ), ਮੁੱਲ.get_ਬਿੱਟ());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#ਜੇ defined(ENABLE_BF16)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ __nv_bfloat16 floatX;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ __nv_bfloat16 floatN;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#ਜਾਂਜੇ defined(ENABLE_FP16)

ਕਿਸਮਪਰਿਭਾਸ਼ਾ half floatX;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ half floatN;

#ਨਹੀਂਤਾਂ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ floatX;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ floatN;
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#ਜੇ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ਅਚਲ floatX* ਪਤਾ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ bf = __ldcs(reinterpret_cast<ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ*>(ਪਤਾ));
    ਵਾਪਸ __nv_bfloat16_raw{bf};
}

__device__ ਖਾਲੀ __stcs(floatX* ਪਤਾ, floatX ਮੁੱਲ) {
    __stcs(reinterpret_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ ਛੋਟਾ*>(ਪਤਾ), ((__nv_bfloat16_raw)ਮੁੱਲ).x);
}
#ਜੇ_ਅੰਤ


// ----------------------------------------------------------------------------
// random utils

ਦਸ਼ਮਲਵ* make_random_float_01(size_t N) {
    ਦਸ਼ਮਲਵ* arr = (ਦਸ਼ਮਲਵ*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(N * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਲਈ (size_t i = 0; i < N; i++) {
        arr[i] = ((ਦਸ਼ਮਲਵ)rand() / RAND_MAX); // range 0..1
    }
    ਵਾਪਸ arr;
}

ਦਸ਼ਮਲਵ* make_random_float(size_t N) {
    ਦਸ਼ਮਲਵ* arr = (ਦਸ਼ਮਲਵ*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(N * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਲਈ (size_t i = 0; i < N; i++) {
        arr[i] = ((ਦਸ਼ਮਲਵ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ਵਾਪਸ arr;
}

ਪੂਰਨ* make_random_int(size_t N, ਪੂਰਨ V) {
    ਪੂਰਨ* arr = (ਪੂਰਨ*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(N * ਆਕਾਰ(ਪੂਰਨ));
    ਲਈ (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    ਵਾਪਸ arr;
}

ਦਸ਼ਮਲਵ* make_zeros_float(size_t N) {
    ਦਸ਼ਮਲਵ* arr = (ਦਸ਼ਮਲਵ*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(N * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    memset(arr, 0, N * ਆਕਾਰ(ਦਸ਼ਮਲਵ)); // all zero
    ਵਾਪਸ arr;
}

ਦਸ਼ਮਲਵ* make_ones_float(size_t N) {
    ਦਸ਼ਮਲਵ* arr = (ਦਸ਼ਮਲਵ*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(N * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਲਈ (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    ਵਾਪਸ arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

ਸਾਂਚਾ<class ਨਿਸ਼ਾਨਾ_Type>
[[nodiscard]] cudaError_t memcpy_convert(ਨਿਸ਼ਾਨਾ_Type* d_ptr, ਦਸ਼ਮਲਵ* h_ptr, size_t ਗਿਣਤੀ_2) {
    // copy from host to device with data type conversion.
    ਨਿਸ਼ਾਨਾ_Type* converted = (ਨਿਸ਼ਾਨਾ_Type*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਗਿਣਤੀ_2 * ਆਕਾਰ(ਨਿਸ਼ਾਨਾ_Type));
    ਲਈ (ਪੂਰਨ i = 0; i < ਗਿਣਤੀ_2; i++) {
        converted[i] = (ਨਿਸ਼ਾਨਾ_Type)h_ptr[i];
    }

    cudaError_t ਸਥਿਤੀ = cudaMemcpy(d_ptr, converted, ਗਿਣਤੀ_2 * ਆਕਾਰ(ਨਿਸ਼ਾਨਾ_Type), cuda_Memcpy_Host_ਮੰਜ਼ਿਲ_Device);
    ਮੁਕਤ(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    ਵਾਪਸ ਸਥਿਤੀ;
}

ਖਾਲੀ ਤਿਆਰੀ_main() {
    srand(0);   // determinism

    // set up the device
    ਪੂਰਨ deviceIdx = 0;
    cudaCheck(cuda_ਸੈੱਟ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_ਅੰਕ_SMs = deviceProp.multi_Processor_ਗਿਣਤੀ;
    cuda_ਧਾਗੇ_ਪ੍ਰਤੀ_SM = deviceProp.max_ਧਾਗੇ_ਪ੍ਰਤੀ_Multi_Processor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_ਮਾਪ));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    ਪੂਰਨ enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_ਬਹੁਲਕ = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublas_ਸੈੱਟ_Math_ਬਹੁਲਕ(cublas_handle, cublas_math_ਬਹੁਲਕ));
}

ਸਾਂਚਾ<class D, class T>
ਖਾਲੀ validate_ਨਤੀਜਾ(D* device_ਨਤੀਜਾ, ਅਚਲ T* cpu_reference, ਅਚਲ ਅੱਖਰ* name, std::size_t ਅੰਕ_elements, T tolerance=1e-4) {
    D* ਬਾਹਰਲਾ_gpu = (D*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਅੰਕ_elements * ਆਕਾਰ(D));
    cudaCheck(cudaMemcpy(ਬਾਹਰਲਾ_gpu, device_ਨਤੀਜਾ, ਅੰਕ_elements * ਆਕਾਰ(D), cuda_Memcpy_Device_ਮੰਜ਼ਿਲ_Host));
    ਪੂਰਨ nfaults = 0;
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ENABLE_BF16
    ਦਸ਼ਮਲਵ epsilon = FLT_EPSILON;
#ਨਹੀਂਤਾਂ
    ਦਸ਼ਮਲਵ epsilon = 0.079;
#ਜੇ_ਅੰਤ
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_elements; i++) {
        // Skip masked elements
        ਜੇ(!isfinite(cpu_reference[i]))
            ਜਾਰੀ;

        // print the first few comparisons
        ਜੇ (i < 5) {
            ਲਿਖੋ("%f %f\n", cpu_reference[i], (T)ਬਾਹਰਲਾ_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        ਦਸ਼ਮਲਵ t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        ਜੇ (fabs(cpu_reference[i] - (T)ਬਾਹਰਲਾ_gpu[i]) > t_eff) {
            ਲਿਖੋ("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)ਬਾਹਰਲਾ_gpu[i]);
            nfaults ++;
            ਜੇ (nfaults >= 10) {
                ਮੁਕਤ(ਬਾਹਰਲਾ_gpu);
                ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
            }
        }
    }

    ਜੇ (nfaults > 0) {
        ਮੁਕਤ(ਬਾਹਰਲਾ_gpu);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }

    ਮੁਕਤ(ਬਾਹਰਲਾ_gpu);
}

ਸਾਂਚਾ<class Kernel, class... Kernel_ਦਲੀਲਾਂ>
ਦਸ਼ਮਲਵ benchmark_kernel(ਪੂਰਨ repeats, Kernel kernel, Kernel_ਦਲੀਲਾਂ&&... kernel_ਦਲੀਲਾਂ) {
    cudaEvent_t ਸ਼ੁਰੂ, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    ਪੂਰਨ deviceIdx = 0;
    cudaCheck(cuda_ਸੈੱਟ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    ਖਾਲੀ* flush_ਬਫਰ;
    cudaCheck(cudaMalloc(&flush_ਬਫਰ, deviceProp.l2_Cache_ਮਾਪ));

    cudaCheck(cudaEventCreate(&ਸ਼ੁਰੂ));
    cudaCheck(cudaEventCreate(&stop));
    ਦਸ਼ਮਲਵ elapsed_ਸਮਾਂ = 0.f;
    ਲਈ (ਪੂਰਨ i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_ਬਫਰ, 0, deviceProp.l2_Cache_ਮਾਪ));
        // now we can start recording the timing of the kernel
        cudaCheck(cuda_Event_ਰਿਕਾਰਡ(ਸ਼ੁਰੂ, nullptr));
        kernel(std::forward<Kernel_ਦਲੀਲਾਂ>(kernel_ਦਲੀਲਾਂ)...);
        cudaCheck(cuda_Event_ਰਿਕਾਰਡ(stop, nullptr));
        cudaCheck(cudaEventSynchronize(ਸ਼ੁਰੂ));
        cudaCheck(cudaEventSynchronize(stop));
        ਦਸ਼ਮਲਵ single_call;
        cudaCheck(cuda_Event_Elapsed_ਸਮਾਂ(&single_call, ਸ਼ੁਰੂ, stop));
        elapsed_ਸਮਾਂ += single_call;
    }

    cudaCheck(cudaFree(flush_ਬਫਰ));

    ਵਾਪਸ elapsed_ਸਮਾਂ / repeats;
}