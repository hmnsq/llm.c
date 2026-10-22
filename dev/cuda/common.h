#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cuda_runtime.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>
#ضمن <float.h>

#تعریف_کرو WARP_ناپ 32U
خارجي cudaDeviceProp deviceProp;

سانچا<class T>
__host__ __device__ T ceil_div(T dividend, T مقسوم_علیہ) {
    أرجع (dividend + مقسوم_علیہ-1) / مقسوم_علیہ;
}

__device__ عائم warp_Reduce_جوڑ(عائم val) {
    لكل (صحيح وتھ = 16; وتھ > 0; وتھ /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, وتھ);
    }
    أرجع val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = عائم (*) (عائم);

سانچا<reduction_func_t warp_reduction>
__device__ مضمن عائم بلاک_Reduce(عائم val, bool final_sync, عائم باہرلا_دا_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ عائم shared_val[WARP_ناپ];
    ثابت صحيح lane_id = threadIdx.x % WARP_ناپ;
    ثابت صحيح warp_id = threadIdx.x / WARP_ناپ;
    ثابت صحيح عدد_warps = بلاک_Dim.x / WARP_ناپ;

    عائم warp_val = warp_reduction(val);
    إذا (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < عدد_warps) ? shared_val[lane_id] : باہرلا_دا_bounds;
    عائم بلاک_val = warp_reduction(warp_val);

    إذا (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    أرجع بلاک_val;
}

// Helper function to call blockReduce with default arguments
سانچا<reduction_func_t warp_reduction>
__device__ مضمن عائم بلاک_Reduce(عائم val) {
    أرجع بلاک_Reduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
فراغ cuda_check(cudaError_t error, ثابت حرف *file, صحيح سطر) {
    إذا (error != cudaSuccess) {
        لکھو("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر,
               cudaGetErrorString(error));
        باہر_نکلو(EXIT_FAILURE);
    }
};
#تعریف_کرو cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
فراغ cublasCheck(cublas_حالت_t حالت, ثابت حرف *file, صحيح سطر)
{
    إذا (حالت != CUBLAS_حالت_SUCCESS) {
        لکھو("[cuBLAS ERROR]: %d %s %d\n", حالت, file, سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}
#تعریف_کرو cublasCheck(حالت) { cublasCheck((حالت), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ساكن size_t cublaslt_workspace_ناپ = 32 * 1024 * 1024;
ساكن فراغ* cublaslt_workspace = عدم;
ساكن cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
صحيح cuda_arch_major = 0;
صحيح cuda_arch_minor = 0;
صحيح cuda_عدد_SMs = 0; // for persistent threads where we want 1 threadblock per SM
صحيح cuda_دھاگے_فی_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#إذا __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#تعریف_کرو MAX_1024_دھاگے_بلاکاں 2
#وإلا
#تعریف_کرو MAX_1024_دھاگے_بلاکاں 1
#جے_انت

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

سانچا<class ElementType>
بنية alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = افتراضي;
    __device__ explicit Packed128(int4 بٹ) {
        static_assert(حجم(بٹ) == حجم(payload), "Size mismatch.");
        حافظہ_نقل(&payload, &بٹ, حجم(بٹ));
    }

    __device__  ساكن Packed128 constant(ElementType قدر) {
        Packed128 نتیجہ;
        لكل(صحيح k = 0; k < ناپ; ++k) {
            نتیجہ.payload[k] = قدر;
        }
        أرجع نتیجہ;
    }

    __device__ ساكن Packed128 zeros() {
        أرجع constant(0);
    }

    __device__ ساكن Packed128 ones() {
        أرجع constant(1);
    }

    __device__ ElementType& عامل[](صحيح اشاریہ) {
        أرجع payload[اشاریہ];
    }
    __device__ ثابت ElementType& عامل[](صحيح اشاریہ) ثابت {
        أرجع payload[اشاریہ];
    }
    __device__ int4 get_بٹ() ثابت {
        int4 بٹ;
        static_assert(حجم(بٹ) == حجم(payload), "Size mismatch.");
        حافظہ_نقل(&بٹ, &payload, حجم(بٹ));
        أرجع بٹ;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ساكن constexpr ثابت صحيح ناپ = حجم(int4) / حجم(ElementType);
    ElementType payload[ناپ];
};

// short-form typedef
تعريفنوع Packed128<عائم> f128;

// load a Packed128 from an aligned memory address
سانچا<class ElementType>
__device__ Packed128<ElementType> load128(ثابت ElementType* پتہ) {
    أرجع Packed128<ElementType>{*reinterpret_cast<ثابت int4*>(پتہ)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
سانچا<class ElementType>
__device__ Packed128<ElementType> load128cs(ثابت ElementType* پتہ) {
    أرجع Packed128<ElementType>{__ldcs(reinterpret_cast<ثابت int4*>(پتہ))};
}
// store a Packed128 to an aligned memory address
سانچا<class ElementType>
__device__ فراغ store128(ElementType* نشانہ, Packed128<ElementType> قدر) {
    *reinterpret_cast<int4*>(نشانہ) = قدر.get_بٹ();
}
// store a Packed128 to an aligned memory address with streaming cache hint
سانچا<class ElementType>
__device__ فراغ store128cs(ElementType* نشانہ, Packed128<ElementType> قدر) {
    __stcs(reinterpret_cast<int4*>(نشانہ), قدر.get_بٹ());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
سانچا<class ElementType>
__device__ فراغ store128cg(ElementType* نشانہ, Packed128<ElementType> قدر) {
    __stcg(reinterpret_cast<int4*>(نشانہ), قدر.get_بٹ());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#إذا defined(ENABLE_BF16)

تعريفنوع __nv_bfloat16 floatX;
تعريفنوع __nv_bfloat16 floatN;
#تعریف_کرو CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#تعریف_کرو CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#وإلاإذا defined(ENABLE_FP16)

تعريفنوع half floatX;
تعريفنوع half floatN;

#وإلا

تعريفنوع عائم floatX;
تعريفنوع عائم floatN;
#جے_انت

تعريفنوع Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#إذا defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* پتہ) {
    بنانشانی چھوٹا bf = __ldcs(reinterpret_cast<ثابت بنانشانی چھوٹا*>(پتہ));
    أرجع __nv_bfloat16_raw{bf};
}

__device__ فراغ __stcs(floatX* پتہ, floatX قدر) {
    __stcs(reinterpret_cast<بنانشانی چھوٹا*>(پتہ), ((__nv_bfloat16_raw)قدر).x);
}
#جے_انت


// ----------------------------------------------------------------------------
// random utils

عائم* make_random_float_01(size_t N) {
    عائم* arr = (عائم*)حافظہ_دیو(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX); // range 0..1
    }
    أرجع arr;
}

عائم* make_random_float(size_t N) {
    عائم* arr = (عائم*)حافظہ_دیو(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    أرجع arr;
}

صحيح* make_random_int(size_t N, صحيح V) {
    صحيح* arr = (صحيح*)حافظہ_دیو(N * حجم(صحيح));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    أرجع arr;
}

عائم* make_zeros_float(size_t N) {
    عائم* arr = (عائم*)حافظہ_دیو(N * حجم(عائم));
    memset(arr, 0, N * حجم(عائم)); // all zero
    أرجع arr;
}

عائم* make_ones_float(size_t N) {
    عائم* arr = (عائم*)حافظہ_دیو(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    أرجع arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

سانچا<class نشانہ_Type>
[[nodiscard]] cudaError_t memcpy_convert(نشانہ_Type* d_ptr, عائم* h_ptr, size_t گنتی) {
    // copy from host to device with data type conversion.
    نشانہ_Type* converted = (نشانہ_Type*)حافظہ_دیو(گنتی * حجم(نشانہ_Type));
    لكل (صحيح i = 0; i < گنتی; i++) {
        converted[i] = (نشانہ_Type)h_ptr[i];
    }

    cudaError_t حالت = cudaMemcpy(d_ptr, converted, گنتی * حجم(نشانہ_Type), cuda_Memcpy_Host_منزل_Device);
    حرر(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    أرجع حالت;
}

فراغ تیاری_main() {
    srand(0);   // determinism

    // set up the device
    صحيح deviceIdx = 0;
    cudaCheck(cuda_سیٹ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_عدد_SMs = deviceProp.multi_Processor_گنتی;
    cuda_دھاگے_فی_SM = deviceProp.max_دھاگے_فی_Multi_Processor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_ناپ));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    صحيح enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_منوال = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublas_سیٹ_Math_منوال(cublas_handle, cublas_math_منوال));
}

سانچا<class D, class T>
فراغ validate_نتیجہ(D* device_نتیجہ, ثابت T* cpu_reference, ثابت حرف* name, std::size_t عدد_elements, T tolerance=1e-4) {
    D* باہرلا_gpu = (D*)حافظہ_دیو(عدد_elements * حجم(D));
    cudaCheck(cudaMemcpy(باہرلا_gpu, device_نتیجہ, عدد_elements * حجم(D), cuda_Memcpy_Device_منزل_Host));
    صحيح nfaults = 0;
#جے_بنا_تعریف ENABLE_BF16
    عائم epsilon = FLT_EPSILON;
#وإلا
    عائم epsilon = 0.079;
#جے_انت
    لكل (صحيح i = 0; i < عدد_elements; i++) {
        // Skip masked elements
        إذا(!isfinite(cpu_reference[i]))
            استمر;

        // print the first few comparisons
        إذا (i < 5) {
            لکھو("%f %f\n", cpu_reference[i], (T)باہرلا_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        عائم t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        إذا (fabs(cpu_reference[i] - (T)باہرلا_gpu[i]) > t_eff) {
            لکھو("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)باہرلا_gpu[i]);
            nfaults ++;
            إذا (nfaults >= 10) {
                حرر(باہرلا_gpu);
                باہر_نکلو(EXIT_FAILURE);
            }
        }
    }

    إذا (nfaults > 0) {
        حرر(باہرلا_gpu);
        باہر_نکلو(EXIT_FAILURE);
    }

    حرر(باہرلا_gpu);
}

سانچا<class Kernel, class... Kernel_دلیلاں>
عائم benchmark_kernel(صحيح repeats, Kernel kernel, Kernel_دلیلاں&&... kernel_دلیلاں) {
    cudaEvent_t شروع, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    صحيح deviceIdx = 0;
    cudaCheck(cuda_سیٹ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    فراغ* flush_بفر;
    cudaCheck(cudaMalloc(&flush_بفر, deviceProp.l2_Cache_ناپ));

    cudaCheck(cudaEventCreate(&شروع));
    cudaCheck(cudaEventCreate(&stop));
    عائم elapsed_ویلا = 0.f;
    لكل (صحيح i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_بفر, 0, deviceProp.l2_Cache_ناپ));
        // now we can start recording the timing of the kernel
        cudaCheck(cuda_Event_ریکارڈ(شروع, nullptr));
        kernel(std::forward<Kernel_دلیلاں>(kernel_دلیلاں)...);
        cudaCheck(cuda_Event_ریکارڈ(stop, nullptr));
        cudaCheck(cudaEventSynchronize(شروع));
        cudaCheck(cudaEventSynchronize(stop));
        عائم single_call;
        cudaCheck(cuda_Event_Elapsed_ویلا(&single_call, شروع, stop));
        elapsed_ویلا += single_call;
    }

    cudaCheck(cudaFree(flush_بفر));

    أرجع elapsed_ویلا / repeats;
}