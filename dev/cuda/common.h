#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <cuda_runtime.h>
#ಅನ್ತರ್ಭಾವ <cublas_v2.h>
#ಅನ್ತರ್ಭಾವ <cublasLt.h>
#ಅನ್ತರ್ಭಾವ <float.h>

#ವ್ಯಾಖ್ಯಾನಿಸು WARP_ಗಾತ್ರ 32U
ಬಾಹ್ಯ cudaDeviceProp deviceProp;

ಮಾದರಿ<class T>
__host__ __device__ T ceil_div(T dividend, T ಭಾಜಕ) {
    ಹಿಂತಿರುಗಿ (dividend + ಭಾಜಕ-1) / ಭಾಜಕ;
}

__device__ ದಶಮಾಂಶ warp_Reduce_ಮೊತ್ತ(ದಶಮಾಂಶ val) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = 16; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, ಸ್ಥಾನಾಂತರ);
    }
    ಹಿಂತಿರುಗಿ val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = ದಶಮಾಂಶ (*) (ದಶಮಾಂಶ);

ಮಾದರಿ<reduction_func_t warp_reduction>
__device__ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ಖಂಡ_Reduce(ದಶಮಾಂಶ val, bool final_sync, ದಶಮಾಂಶ ಹೊರಹೋಗು_ದ_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ ದಶಮಾಂಶ shared_val[WARP_ಗಾತ್ರ];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane_id = threadIdx.x % WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp_id = threadIdx.x / WARP_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_warps = ಖಂಡ_Dim.x / WARP_ಗಾತ್ರ;

    ದಶಮಾಂಶ warp_val = warp_reduction(val);
    ಆದರೆ (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < ಅಂಕೆ_warps) ? shared_val[lane_id] : ಹೊರಹೋಗು_ದ_bounds;
    ದಶಮಾಂಶ ಖಂಡ_val = warp_reduction(warp_val);

    ಆದರೆ (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    ಹಿಂತಿರುಗಿ ಖಂಡ_val;
}

// Helper function to call blockReduce with default arguments
ಮಾದರಿ<reduction_func_t warp_reduction>
__device__ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ಖಂಡ_Reduce(ದಶಮಾಂಶ val) {
    ಹಿಂತಿರುಗಿ ಖಂಡ_Reduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
ಶೂನ್ಯ cuda_check(cudaError_t error, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (error != cudaSuccess) {
        ಬರೆ("[CUDA ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು,
               cudaGetErrorString(error));
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
};
#ವ್ಯಾಖ್ಯಾನಿಸು cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
ಶೂನ್ಯ cublasCheck(cublas_ಸ್ಥಿತಿ_t ಸ್ಥಿತಿ_2, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು)
{
    ಆದರೆ (ಸ್ಥಿತಿ_2 != CUBLAS_ಸ್ಥಿತಿ_SUCCESS) {
        ಬರೆ("[cuBLAS ERROR]: %d %s %d\n", ಸ್ಥಿತಿ_2, file, ಸಾಲು);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}
#ವ್ಯಾಖ್ಯಾನಿಸು cublasCheck(ಸ್ಥಿತಿ_2) { cublasCheck((ಸ್ಥಿತಿ_2), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ಸ್ಥಿರ size_t cublaslt_workspace_ಗಾತ್ರ = 32 * 1024 * 1024;
ಸ್ಥಿರ ಶೂನ್ಯ* cublaslt_workspace = ಖಾಲಿ;
ಸ್ಥಿರ cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
ಪೂರ್ಣಾಂಕ cuda_arch_major = 0;
ಪೂರ್ಣಾಂಕ cuda_arch_minor = 0;
ಪೂರ್ಣಾಂಕ cuda_ಅಂಕೆ_SMs = 0; // for persistent threads where we want 1 threadblock per SM
ಪೂರ್ಣಾಂಕ cuda_ಎಳೆಗಳು_ಪ್ರತಿ_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#ಆದರೆ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_1024_ಎಳೆಗಳು_ಖಂಡಗಳು 2
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_1024_ಎಳೆಗಳು_ಖಂಡಗಳು 1
#ಆದರೆ_ಅಂತ್ಯ

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

ಮಾದರಿ<class ElementType>
ರಚನೆ alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = ಪೂರ್ವನಿಯೋಜಿತ;
    __device__ explicit Packed128(int4 ಬಿಟ್ಟುಗಳು) {
        static_assert(ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು) == ಗಾತ್ರ(payload), "Size mismatch.");
        ಸ್ಮರಣೆ_ನಕಲು(&payload, &ಬಿಟ್ಟುಗಳು, ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು));
    }

    __device__  ಸ್ಥಿರ Packed128 constant(ElementType ಮೌಲ್ಯ) {
        Packed128 ಫಲಿತಾಂಶ;
        ಗಾಗಿ(ಪೂರ್ಣಾಂಕ k = 0; k < ಗಾತ್ರ_2; ++k) {
            ಫಲಿತಾಂಶ.payload[k] = ಮೌಲ್ಯ;
        }
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    __device__ ಸ್ಥಿರ Packed128 zeros() {
        ಹಿಂತಿರುಗಿ constant(0);
    }

    __device__ ಸ್ಥಿರ Packed128 ones() {
        ಹಿಂತಿರುಗಿ constant(1);
    }

    __device__ ElementType& ಕ್ರಿಯಾಕಾರಕ[](ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ) {
        ಹಿಂತಿರುಗಿ payload[ಸೂಚ್ಯಂಕ];
    }
    __device__ ಸ್ಥಿರಾಂಕ ElementType& ಕ್ರಿಯಾಕಾರಕ[](ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ payload[ಸೂಚ್ಯಂಕ];
    }
    __device__ int4 get_ಬಿಟ್ಟುಗಳು() ಸ್ಥಿರಾಂಕ {
        int4 ಬಿಟ್ಟುಗಳು;
        static_assert(ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು) == ಗಾತ್ರ(payload), "Size mismatch.");
        ಸ್ಮರಣೆ_ನಕಲು(&ಬಿಟ್ಟುಗಳು, &payload, ಗಾತ್ರ(ಬಿಟ್ಟುಗಳು));
        ಹಿಂತಿರುಗಿ ಬಿಟ್ಟುಗಳು;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ಸ್ಥಿರ constexpr ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಗಾತ್ರ_2 = ಗಾತ್ರ(int4) / ಗಾತ್ರ(ElementType);
    ElementType payload[ಗಾತ್ರ_2];
};

// short-form typedef
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ Packed128<ದಶಮಾಂಶ> f128;

// load a Packed128 from an aligned memory address
ಮಾದರಿ<class ElementType>
__device__ Packed128<ElementType> load128(ಸ್ಥಿರಾಂಕ ElementType* ವಿಳಾಸ) {
    ಹಿಂತಿರುಗಿ Packed128<ElementType>{*reinterpret_cast<ಸ್ಥಿರಾಂಕ int4*>(ವಿಳಾಸ)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
ಮಾದರಿ<class ElementType>
__device__ Packed128<ElementType> load128cs(ಸ್ಥಿರಾಂಕ ElementType* ವಿಳಾಸ) {
    ಹಿಂತಿರುಗಿ Packed128<ElementType>{__ldcs(reinterpret_cast<ಸ್ಥಿರಾಂಕ int4*>(ವಿಳಾಸ))};
}
// store a Packed128 to an aligned memory address
ಮಾದರಿ<class ElementType>
__device__ ಶೂನ್ಯ store128(ElementType* ಗುರಿ, Packed128<ElementType> ಮೌಲ್ಯ) {
    *reinterpret_cast<int4*>(ಗುರಿ) = ಮೌಲ್ಯ.get_ಬಿಟ್ಟುಗಳು();
}
// store a Packed128 to an aligned memory address with streaming cache hint
ಮಾದರಿ<class ElementType>
__device__ ಶೂನ್ಯ store128cs(ElementType* ಗುರಿ, Packed128<ElementType> ಮೌಲ್ಯ) {
    __stcs(reinterpret_cast<int4*>(ಗುರಿ), ಮೌಲ್ಯ.get_ಬಿಟ್ಟುಗಳು());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
ಮಾದರಿ<class ElementType>
__device__ ಶೂನ್ಯ store128cg(ElementType* ಗುರಿ, Packed128<ElementType> ಮೌಲ್ಯ) {
    __stcg(reinterpret_cast<int4*>(ಗುರಿ), ಮೌಲ್ಯ.get_ಬಿಟ್ಟುಗಳು());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#ಆದರೆ defined(ENABLE_BF16)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __nv_bfloat16 floatX;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __nv_bfloat16 floatN;
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#ಅಥವಾಆದರೆ defined(ENABLE_FP16)

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ half floatX;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ half floatN;

#ಇಲ್ಲದಿದ್ದರೆ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ floatX;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ floatN;
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#ಆದರೆ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ಸ್ಥಿರಾಂಕ floatX* ವಿಳಾಸ) {
    ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ bf = __ldcs(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ*>(ವಿಳಾಸ));
    ಹಿಂತಿರುಗಿ __nv_bfloat16_raw{bf};
}

__device__ ಶೂನ್ಯ __stcs(floatX* ವಿಳಾಸ, floatX ಮೌಲ್ಯ) {
    __stcs(reinterpret_cast<ಚಿಹ್ನೆರಹಿತ ಗಿಡ್ಡ*>(ವಿಳಾಸ), ((__nv_bfloat16_raw)ಮೌಲ್ಯ).x);
}
#ಆದರೆ_ಅಂತ್ಯ


// ----------------------------------------------------------------------------
// random utils

ದಶಮಾಂಶ* make_random_float_01(size_t N) {
    ದಶಮಾಂಶ* arr = (ದಶಮಾಂಶ*)ಸ್ಮರಣೆ_ಕೊಡು(N * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಗಾಗಿ (size_t i = 0; i < N; i++) {
        arr[i] = ((ದಶಮಾಂಶ)rand() / RAND_MAX); // range 0..1
    }
    ಹಿಂತಿರುಗಿ arr;
}

ದಶಮಾಂಶ* make_random_float(size_t N) {
    ದಶಮಾಂಶ* arr = (ದಶಮಾಂಶ*)ಸ್ಮರಣೆ_ಕೊಡು(N * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಗಾಗಿ (size_t i = 0; i < N; i++) {
        arr[i] = ((ದಶಮಾಂಶ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ಹಿಂತಿರುಗಿ arr;
}

ಪೂರ್ಣಾಂಕ* make_random_int(size_t N, ಪೂರ್ಣಾಂಕ V) {
    ಪೂರ್ಣಾಂಕ* arr = (ಪೂರ್ಣಾಂಕ*)ಸ್ಮರಣೆ_ಕೊಡು(N * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಗಾಗಿ (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    ಹಿಂತಿರುಗಿ arr;
}

ದಶಮಾಂಶ* make_zeros_float(size_t N) {
    ದಶಮಾಂಶ* arr = (ದಶಮಾಂಶ*)ಸ್ಮರಣೆ_ಕೊಡು(N * ಗಾತ್ರ(ದಶಮಾಂಶ));
    memset(arr, 0, N * ಗಾತ್ರ(ದಶಮಾಂಶ)); // all zero
    ಹಿಂತಿರುಗಿ arr;
}

ದಶಮಾಂಶ* make_ones_float(size_t N) {
    ದಶಮಾಂಶ* arr = (ದಶಮಾಂಶ*)ಸ್ಮರಣೆ_ಕೊಡು(N * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಗಾಗಿ (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    ಹಿಂತಿರುಗಿ arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

ಮಾದರಿ<class ಗುರಿ_Type>
[[nodiscard]] cudaError_t memcpy_convert(ಗುರಿ_Type* d_ptr, ದಶಮಾಂಶ* h_ptr, size_t ಎಣಿಕೆ_2) {
    // copy from host to device with data type conversion.
    ಗುರಿ_Type* converted = (ಗುರಿ_Type*)ಸ್ಮರಣೆ_ಕೊಡು(ಎಣಿಕೆ_2 * ಗಾತ್ರ(ಗುರಿ_Type));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಎಣಿಕೆ_2; i++) {
        converted[i] = (ಗುರಿ_Type)h_ptr[i];
    }

    cudaError_t ಸ್ಥಿತಿ_2 = cudaMemcpy(d_ptr, converted, ಎಣಿಕೆ_2 * ಗಾತ್ರ(ಗುರಿ_Type), cuda_Memcpy_Host_ಗಮ್ಯ_Device);
    ಬಿಡುಗಡೆ(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
}

ಶೂನ್ಯ ಸಿದ್ಧತೆ_main() {
    srand(0);   // determinism

    // set up the device
    ಪೂರ್ಣಾಂಕ deviceIdx = 0;
    cudaCheck(cuda_ಗಣ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_ಅಂಕೆ_SMs = deviceProp.multi_Processor_ಎಣಿಕೆ;
    cuda_ಎಳೆಗಳು_ಪ್ರತಿ_SM = deviceProp.max_ಎಳೆಗಳು_ಪ್ರತಿ_Multi_Processor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_ಗಾತ್ರ));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    ಪೂರ್ಣಾಂಕ enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_ಬಹುಲಕ = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublas_ಗಣ_Math_ಬಹುಲಕ(cublas_handle, cublas_math_ಬಹುಲಕ));
}

ಮಾದರಿ<class D, class T>
ಶೂನ್ಯ validate_ಫಲಿತಾಂಶ(D* device_ಫಲಿತಾಂಶ, ಸ್ಥಿರಾಂಕ T* cpu_reference, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* name, std::size_t ಅಂಕೆ_elements, T tolerance=1e-4) {
    D* ಹೊರಹೋಗು_gpu = (D*)ಸ್ಮರಣೆ_ಕೊಡು(ಅಂಕೆ_elements * ಗಾತ್ರ(D));
    cudaCheck(cudaMemcpy(ಹೊರಹೋಗು_gpu, device_ಫಲಿತಾಂಶ, ಅಂಕೆ_elements * ಗಾತ್ರ(D), cuda_Memcpy_Device_ಗಮ್ಯ_Host));
    ಪೂರ್ಣಾಂಕ nfaults = 0;
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ ENABLE_BF16
    ದಶಮಾಂಶ epsilon = FLT_EPSILON;
#ಇಲ್ಲದಿದ್ದರೆ
    ದಶಮಾಂಶ epsilon = 0.079;
#ಆದರೆ_ಅಂತ್ಯ
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_elements; i++) {
        // Skip masked elements
        ಆದರೆ(!isfinite(cpu_reference[i]))
            ಮುಂದುವರಿ;

        // print the first few comparisons
        ಆದರೆ (i < 5) {
            ಬರೆ("%f %f\n", cpu_reference[i], (T)ಹೊರಹೋಗು_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        ದಶಮಾಂಶ t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        ಆದರೆ (fabs(cpu_reference[i] - (T)ಹೊರಹೋಗು_gpu[i]) > t_eff) {
            ಬರೆ("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)ಹೊರಹೋಗು_gpu[i]);
            nfaults ++;
            ಆದರೆ (nfaults >= 10) {
                ಬಿಡುಗಡೆ(ಹೊರಹೋಗು_gpu);
                ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
            }
        }
    }

    ಆದರೆ (nfaults > 0) {
        ಬಿಡುಗಡೆ(ಹೊರಹೋಗು_gpu);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }

    ಬಿಡುಗಡೆ(ಹೊರಹೋಗು_gpu);
}

ಮಾದರಿ<class Kernel, class... Kernel_ವಾದಗಳು>
ದಶಮಾಂಶ benchmark_kernel(ಪೂರ್ಣಾಂಕ repeats, Kernel kernel, Kernel_ವಾದಗಳು&&... kernel_ವಾದಗಳು) {
    cudaEvent_t ಆರಂಭ, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    ಪೂರ್ಣಾಂಕ deviceIdx = 0;
    cudaCheck(cuda_ಗಣ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    ಶೂನ್ಯ* flush_ಬಫರ್;
    cudaCheck(cudaMalloc(&flush_ಬಫರ್, deviceProp.l2_Cache_ಗಾತ್ರ));

    cudaCheck(cudaEventCreate(&ಆರಂಭ));
    cudaCheck(cudaEventCreate(&stop));
    ದಶಮಾಂಶ elapsed_ಸಮಯ = 0.f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_ಬಫರ್, 0, deviceProp.l2_Cache_ಗಾತ್ರ));
        // now we can start recording the timing of the kernel
        cudaCheck(cuda_Event_ದಾಖಲೆ(ಆರಂಭ, nullptr));
        kernel(std::forward<Kernel_ವಾದಗಳು>(kernel_ವಾದಗಳು)...);
        cudaCheck(cuda_Event_ದಾಖಲೆ(stop, nullptr));
        cudaCheck(cudaEventSynchronize(ಆರಂಭ));
        cudaCheck(cudaEventSynchronize(stop));
        ದಶಮಾಂಶ single_call;
        cudaCheck(cuda_Event_Elapsed_ಸಮಯ(&single_call, ಆರಂಭ, stop));
        elapsed_ಸಮಯ += single_call;
    }

    cudaCheck(cudaFree(flush_ಬಫರ್));

    ಹಿಂತಿರುಗಿ elapsed_ಸಮಯ / repeats;
}