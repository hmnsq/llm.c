#ضمن <stdlib.h>
#ضمن <stdio.h>
#ضمن <cuda_runtime.h>
#ضمن <cublas_v2.h>
#ضمن <cublasLt.h>
#ضمن <float.h>

#تعریف_کن WARP_اندازه 32U
خارجي cudaDeviceProp deviceProp;

قالب<class T>
__host__ __device__ T ceil_div(T dividend, T مقسوم_علیه) {
    أرجع (dividend + مقسوم_علیه-1) / مقسوم_علیه;
}

__device__ عائم warp_Reduce_جمع(عائم val) {
    لكل (صحيح جابه_جایی = 16; جابه_جایی > 0; جابه_جایی /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, جابه_جایی);
    }
    أرجع val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = عائم (*) (عائم);

قالب<reduction_func_t warp_reduction>
__device__ مضمن عائم بلوک_Reduce(عائم val, bool final_sync, عائم بیرون_از_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ عائم shared_val[WARP_اندازه];
    ثابت صحيح lane_id = threadIdx.x % WARP_اندازه;
    ثابت صحيح warp_id = threadIdx.x / WARP_اندازه;
    ثابت صحيح رقم_warps = بلوک_Dim.x / WARP_اندازه;

    عائم warp_val = warp_reduction(val);
    إذا (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < رقم_warps) ? shared_val[lane_id] : بیرون_از_bounds;
    عائم بلوک_val = warp_reduction(warp_val);

    إذا (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    أرجع بلوک_val;
}

// Helper function to call blockReduce with default arguments
قالب<reduction_func_t warp_reduction>
__device__ مضمن عائم بلوک_Reduce(عائم val) {
    أرجع بلوک_Reduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
فراغ cuda_check(cudaError_t error, ثابت حرف *file, صحيح سطر) {
    إذا (error != cudaSuccess) {
        چاپ_کن("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر,
               cudaGetErrorString(error));
        خروج(EXIT_FAILURE);
    }
};
#تعریف_کن cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
فراغ cublasCheck(cublas_وضعیت_t وضعیت, ثابت حرف *file, صحيح سطر)
{
    إذا (وضعیت != CUBLAS_وضعیت_SUCCESS) {
        چاپ_کن("[cuBLAS ERROR]: %d %s %d\n", وضعیت, file, سطر);
        خروج(EXIT_FAILURE);
    }
}
#تعریف_کن cublasCheck(وضعیت) { cublasCheck((وضعیت), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ساكن size_t cublaslt_workspace_اندازه = 32 * 1024 * 1024;
ساكن فراغ* cublaslt_workspace = عدم;
ساكن cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
صحيح cuda_arch_major = 0;
صحيح cuda_arch_minor = 0;
صحيح cuda_رقم_SMs = 0; // for persistent threads where we want 1 threadblock per SM
صحيح cuda_نخها_در_هر_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#إذا __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#تعریف_کن MAX_1024_نخها_بلوکها 2
#وإلا
#تعریف_کن MAX_1024_نخها_بلوکها 1
#پایان_اگر

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

قالب<class ElementType>
بنية alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = افتراضي;
    __device__ explicit Packed128(int4 بیتها) {
        static_assert(حجم(بیتها) == حجم(payload), "Size mismatch.");
        کپی_حافظه(&payload, &بیتها, حجم(بیتها));
    }

    __device__  ساكن Packed128 constant(ElementType مقدار) {
        Packed128 نتیجه;
        لكل(صحيح k = 0; k < اندازه; ++k) {
            نتیجه.payload[k] = مقدار;
        }
        أرجع نتیجه;
    }

    __device__ ساكن Packed128 zeros() {
        أرجع constant(0);
    }

    __device__ ساكن Packed128 ones() {
        أرجع constant(1);
    }

    __device__ ElementType& عملگر[](صحيح شاخص) {
        أرجع payload[شاخص];
    }
    __device__ ثابت ElementType& عملگر[](صحيح شاخص) ثابت {
        أرجع payload[شاخص];
    }
    __device__ int4 get_بیتها() ثابت {
        int4 بیتها;
        static_assert(حجم(بیتها) == حجم(payload), "Size mismatch.");
        کپی_حافظه(&بیتها, &payload, حجم(بیتها));
        أرجع بیتها;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ساكن constexpr ثابت صحيح اندازه = حجم(int4) / حجم(ElementType);
    ElementType payload[اندازه];
};

// short-form typedef
تعريفنوع Packed128<عائم> f128;

// load a Packed128 from an aligned memory address
قالب<class ElementType>
__device__ Packed128<ElementType> load128(ثابت ElementType* آدرس) {
    أرجع Packed128<ElementType>{*reinterpret_cast<ثابت int4*>(آدرس)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
قالب<class ElementType>
__device__ Packed128<ElementType> load128cs(ثابت ElementType* آدرس) {
    أرجع Packed128<ElementType>{__ldcs(reinterpret_cast<ثابت int4*>(آدرس))};
}
// store a Packed128 to an aligned memory address
قالب<class ElementType>
__device__ فراغ store128(ElementType* هدف, Packed128<ElementType> مقدار) {
    *reinterpret_cast<int4*>(هدف) = مقدار.get_بیتها();
}
// store a Packed128 to an aligned memory address with streaming cache hint
قالب<class ElementType>
__device__ فراغ store128cs(ElementType* هدف, Packed128<ElementType> مقدار) {
    __stcs(reinterpret_cast<int4*>(هدف), مقدار.get_بیتها());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
قالب<class ElementType>
__device__ فراغ store128cg(ElementType* هدف, Packed128<ElementType> مقدار) {
    __stcg(reinterpret_cast<int4*>(هدف), مقدار.get_بیتها());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#إذا defined(ENABLE_BF16)

تعريفنوع __nv_bfloat16 floatX;
تعريفنوع __nv_bfloat16 floatN;
#تعریف_کن CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#تعریف_کن CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#وإلاإذا defined(ENABLE_FP16)

تعريفنوع half floatX;
تعريفنوع half floatN;

#وإلا

تعريفنوع عائم floatX;
تعريفنوع عائم floatN;
#پایان_اگر

تعريفنوع Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#إذا defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* آدرس) {
    بیعلامت کوتاه bf = __ldcs(reinterpret_cast<ثابت بیعلامت کوتاه*>(آدرس));
    أرجع __nv_bfloat16_raw{bf};
}

__device__ فراغ __stcs(floatX* آدرس, floatX مقدار) {
    __stcs(reinterpret_cast<بیعلامت کوتاه*>(آدرس), ((__nv_bfloat16_raw)مقدار).x);
}
#پایان_اگر


// ----------------------------------------------------------------------------
// random utils

عائم* make_random_float_01(size_t N) {
    عائم* arr = (عائم*)حافظه_بده(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX); // range 0..1
    }
    أرجع arr;
}

عائم* make_random_float(size_t N) {
    عائم* arr = (عائم*)حافظه_بده(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    أرجع arr;
}

صحيح* make_random_int(size_t N, صحيح V) {
    صحيح* arr = (صحيح*)حافظه_بده(N * حجم(صحيح));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    أرجع arr;
}

عائم* make_zeros_float(size_t N) {
    عائم* arr = (عائم*)حافظه_بده(N * حجم(عائم));
    memset(arr, 0, N * حجم(عائم)); // all zero
    أرجع arr;
}

عائم* make_ones_float(size_t N) {
    عائم* arr = (عائم*)حافظه_بده(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    أرجع arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

قالب<class هدف_Type>
[[nodiscard]] cudaError_t memcpy_convert(هدف_Type* d_ptr, عائم* h_ptr, size_t شمار) {
    // copy from host to device with data type conversion.
    هدف_Type* converted = (هدف_Type*)حافظه_بده(شمار * حجم(هدف_Type));
    لكل (صحيح i = 0; i < شمار; i++) {
        converted[i] = (هدف_Type)h_ptr[i];
    }

    cudaError_t وضعیت = cudaMemcpy(d_ptr, converted, شمار * حجم(هدف_Type), cuda_Memcpy_Host_مقصد_Device);
    حرر(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    أرجع وضعیت;
}

فراغ راهاندازی_main() {
    srand(0);   // determinism

    // set up the device
    صحيح deviceIdx = 0;
    cudaCheck(cuda_مجموعه_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_رقم_SMs = deviceProp.multi_Processor_شمار;
    cuda_نخها_در_هر_SM = deviceProp.max_نخها_در_هر_Multi_Processor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_اندازه));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    صحيح enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_نما = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublas_مجموعه_Math_نما(cublas_handle, cublas_math_نما));
}

قالب<class D, class T>
فراغ validate_نتیجه(D* device_نتیجه, ثابت T* cpu_reference, ثابت حرف* name, std::size_t رقم_elements, T tolerance=1e-4) {
    D* بیرون_gpu = (D*)حافظه_بده(رقم_elements * حجم(D));
    cudaCheck(cudaMemcpy(بیرون_gpu, device_نتیجه, رقم_elements * حجم(D), cuda_Memcpy_Device_مقصد_Host));
    صحيح nfaults = 0;
#اگر_تعریف_نشده ENABLE_BF16
    عائم epsilon = FLT_EPSILON;
#وإلا
    عائم epsilon = 0.079;
#پایان_اگر
    لكل (صحيح i = 0; i < رقم_elements; i++) {
        // Skip masked elements
        إذا(!isfinite(cpu_reference[i]))
            استمر;

        // print the first few comparisons
        إذا (i < 5) {
            چاپ_کن("%f %f\n", cpu_reference[i], (T)بیرون_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        عائم t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        إذا (fabs(cpu_reference[i] - (T)بیرون_gpu[i]) > t_eff) {
            چاپ_کن("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)بیرون_gpu[i]);
            nfaults ++;
            إذا (nfaults >= 10) {
                حرر(بیرون_gpu);
                خروج(EXIT_FAILURE);
            }
        }
    }

    إذا (nfaults > 0) {
        حرر(بیرون_gpu);
        خروج(EXIT_FAILURE);
    }

    حرر(بیرون_gpu);
}

قالب<class Kernel, class... Kernel_آرگومانها>
عائم benchmark_kernel(صحيح repeats, Kernel kernel, Kernel_آرگومانها&&... kernel_آرگومانها) {
    cudaEvent_t شروع, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    صحيح deviceIdx = 0;
    cudaCheck(cuda_مجموعه_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    فراغ* flush_میانگیر;
    cudaCheck(cudaMalloc(&flush_میانگیر, deviceProp.l2_Cache_اندازه));

    cudaCheck(cudaEventCreate(&شروع));
    cudaCheck(cudaEventCreate(&stop));
    عائم elapsed_زمان = 0.f;
    لكل (صحيح i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_میانگیر, 0, deviceProp.l2_Cache_اندازه));
        // now we can start recording the timing of the kernel
        cudaCheck(cuda_Event_رکورد(شروع, nullptr));
        kernel(std::forward<Kernel_آرگومانها>(kernel_آرگومانها)...);
        cudaCheck(cuda_Event_رکورد(stop, nullptr));
        cudaCheck(cudaEventSynchronize(شروع));
        cudaCheck(cudaEventSynchronize(stop));
        عائم single_call;
        cudaCheck(cuda_Event_Elapsed_زمان(&single_call, شروع, stop));
        elapsed_زمان += single_call;
    }

    cudaCheck(cudaFree(flush_میانگیر));

    أرجع elapsed_زمان / repeats;
}