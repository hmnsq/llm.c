#شامل <stdlib.h>
#شامل <stdio.h>
#شامل <cuda_runtime.h>
#شامل <cublas_v2.h>
#شامل <cublasLt.h>
#شامل <float.h>

#تعریف_کرو WARP_سائز 32U
خارجی cudaDeviceProp deviceProp;

سانچہ<class T>
__host__ __device__ T ceil_div(T dividend, T مقسوم_علیہ) {
    واپس (dividend + مقسوم_علیہ-1) / مقسوم_علیہ;
}

__device__ اعشاری warp_Reduce_مجموع(اعشاری val) {
    برائے (صحیح انحراف = 16; انحراف > 0; انحراف /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, انحراف);
    }
    واپس val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = اعشاری (*) (اعشاری);

سانچہ<reduction_func_t warp_reduction>
__device__ اندرونی اعشاری بلاک_Reduce(اعشاری val, bool final_sync, اعشاری برآمد_کا_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ اعشاری shared_val[WARP_سائز];
    ثابت صحیح lane_id = threadIdx.x % WARP_سائز;
    ثابت صحیح warp_id = threadIdx.x / WARP_سائز;
    ثابت صحیح عدد_warps = بلاک_Dim.x / WARP_سائز;

    اعشاری warp_val = warp_reduction(val);
    اگر (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < عدد_warps) ? shared_val[lane_id] : برآمد_کا_bounds;
    اعشاری بلاک_val = warp_reduction(warp_val);

    اگر (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    واپس بلاک_val;
}

// Helper function to call blockReduce with default arguments
سانچہ<reduction_func_t warp_reduction>
__device__ اندرونی اعشاری بلاک_Reduce(اعشاری val) {
    واپس بلاک_Reduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
خالی cuda_check(cudaError_t error, ثابت حرف *file, صحیح سطر) {
    اگر (error != cudaSuccess) {
        لکھو("[CUDA ERROR] at file %s:%d:\n%s\n", file, سطر,
               cudaGetErrorString(error));
        باہر_نکلو(EXIT_FAILURE);
    }
};
#تعریف_کرو cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
خالی cublasCheck(cublas_حالت_t حالت_2, ثابت حرف *file, صحیح سطر)
{
    اگر (حالت_2 != CUBLAS_حالت_SUCCESS) {
        لکھو("[cuBLAS ERROR]: %d %s %d\n", حالت_2, file, سطر);
        باہر_نکلو(EXIT_FAILURE);
    }
}
#تعریف_کرو cublasCheck(حالت_2) { cublasCheck((حالت_2), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
ساکن size_t cublaslt_workspace_سائز = 32 * 1024 * 1024;
ساکن خالی* cublaslt_workspace = تہی;
ساکن cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
صحیح cuda_arch_major = 0;
صحیح cuda_arch_minor = 0;
صحیح cuda_عدد_SMs = 0; // for persistent threads where we want 1 threadblock per SM
صحیح cuda_دھاگے_فی_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#اگر __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#تعریف_کرو MAX_1024_دھاگے_بلاکیں 2
#ورنہ
#تعریف_کرو MAX_1024_دھاگے_بلاکیں 1
#اگر_ختم

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

سانچہ<class ElementType>
ساخت alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = طےشدہ;
    __device__ explicit Packed128(int4 بٹ) {
        static_assert(حجم(بٹ) == حجم(payload), "Size mismatch.");
        حافظہ_نقل(&payload, &بٹ, حجم(بٹ));
    }

    __device__  ساکن Packed128 constant(ElementType قدر) {
        Packed128 نتیجہ;
        برائے(صحیح k = 0; k < سائز; ++k) {
            نتیجہ.payload[k] = قدر;
        }
        واپس نتیجہ;
    }

    __device__ ساکن Packed128 zeros() {
        واپس constant(0);
    }

    __device__ ساکن Packed128 ones() {
        واپس constant(1);
    }

    __device__ ElementType& عامل[](صحیح اشاریہ) {
        واپس payload[اشاریہ];
    }
    __device__ ثابت ElementType& عامل[](صحیح اشاریہ) ثابت {
        واپس payload[اشاریہ];
    }
    __device__ int4 get_بٹ() ثابت {
        int4 بٹ;
        static_assert(حجم(بٹ) == حجم(payload), "Size mismatch.");
        حافظہ_نقل(&بٹ, &payload, حجم(بٹ));
        واپس بٹ;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    ساکن constexpr ثابت صحیح سائز = حجم(int4) / حجم(ElementType);
    ElementType payload[سائز];
};

// short-form typedef
قسمتعریف Packed128<اعشاری> f128;

// load a Packed128 from an aligned memory address
سانچہ<class ElementType>
__device__ Packed128<ElementType> load128(ثابت ElementType* پتہ) {
    واپس Packed128<ElementType>{*reinterpret_cast<ثابت int4*>(پتہ)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
سانچہ<class ElementType>
__device__ Packed128<ElementType> load128cs(ثابت ElementType* پتہ) {
    واپس Packed128<ElementType>{__ldcs(reinterpret_cast<ثابت int4*>(پتہ))};
}
// store a Packed128 to an aligned memory address
سانچہ<class ElementType>
__device__ خالی store128(ElementType* ہدف, Packed128<ElementType> قدر) {
    *reinterpret_cast<int4*>(ہدف) = قدر.get_بٹ();
}
// store a Packed128 to an aligned memory address with streaming cache hint
سانچہ<class ElementType>
__device__ خالی store128cs(ElementType* ہدف, Packed128<ElementType> قدر) {
    __stcs(reinterpret_cast<int4*>(ہدف), قدر.get_بٹ());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
سانچہ<class ElementType>
__device__ خالی store128cg(ElementType* ہدف, Packed128<ElementType> قدر) {
    __stcg(reinterpret_cast<int4*>(ہدف), قدر.get_بٹ());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#اگر defined(ENABLE_BF16)

قسمتعریف __nv_bfloat16 floatX;
قسمتعریف __nv_bfloat16 floatN;
#تعریف_کرو CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#تعریف_کرو CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#وگرنہ_اگر defined(ENABLE_FP16)

قسمتعریف half floatX;
قسمتعریف half floatN;

#ورنہ

قسمتعریف اعشاری floatX;
قسمتعریف اعشاری floatN;
#اگر_ختم

قسمتعریف Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#اگر defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(ثابت floatX* پتہ) {
    غیرعلامتی مختصر bf = __ldcs(reinterpret_cast<ثابت غیرعلامتی مختصر*>(پتہ));
    واپس __nv_bfloat16_raw{bf};
}

__device__ خالی __stcs(floatX* پتہ, floatX قدر) {
    __stcs(reinterpret_cast<غیرعلامتی مختصر*>(پتہ), ((__nv_bfloat16_raw)قدر).x);
}
#اگر_ختم


// ----------------------------------------------------------------------------
// random utils

اعشاری* make_random_float_01(size_t N) {
    اعشاری* arr = (اعشاری*)حافظہ_دو(N * حجم(اعشاری));
    برائے (size_t i = 0; i < N; i++) {
        arr[i] = ((اعشاری)rand() / RAND_MAX); // range 0..1
    }
    واپس arr;
}

اعشاری* make_random_float(size_t N) {
    اعشاری* arr = (اعشاری*)حافظہ_دو(N * حجم(اعشاری));
    برائے (size_t i = 0; i < N; i++) {
        arr[i] = ((اعشاری)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    واپس arr;
}

صحیح* make_random_int(size_t N, صحیح V) {
    صحیح* arr = (صحیح*)حافظہ_دو(N * حجم(صحیح));
    برائے (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    واپس arr;
}

اعشاری* make_zeros_float(size_t N) {
    اعشاری* arr = (اعشاری*)حافظہ_دو(N * حجم(اعشاری));
    memset(arr, 0, N * حجم(اعشاری)); // all zero
    واپس arr;
}

اعشاری* make_ones_float(size_t N) {
    اعشاری* arr = (اعشاری*)حافظہ_دو(N * حجم(اعشاری));
    برائے (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    واپس arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

سانچہ<class ہدف_Type>
[[nodiscard]] cudaError_t memcpy_convert(ہدف_Type* d_ptr, اعشاری* h_ptr, size_t گنتی) {
    // copy from host to device with data type conversion.
    ہدف_Type* converted = (ہدف_Type*)حافظہ_دو(گنتی * حجم(ہدف_Type));
    برائے (صحیح i = 0; i < گنتی; i++) {
        converted[i] = (ہدف_Type)h_ptr[i];
    }

    cudaError_t حالت_2 = cudaMemcpy(d_ptr, converted, گنتی * حجم(ہدف_Type), cuda_Memcpy_Host_مقصد_Device);
    آزاد(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    واپس حالت_2;
}

خالی تیاری_main() {
    srand(0);   // determinism

    // set up the device
    صحیح deviceIdx = 0;
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
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_سائز));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    صحیح enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_منوال = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublas_سیٹ_Math_منوال(cublas_handle, cublas_math_منوال));
}

سانچہ<class D, class T>
خالی validate_نتیجہ(D* device_نتیجہ, ثابت T* cpu_reference, ثابت حرف* name, std::size_t عدد_elements, T tolerance=1e-4) {
    D* برآمد_gpu = (D*)حافظہ_دو(عدد_elements * حجم(D));
    cudaCheck(cudaMemcpy(برآمد_gpu, device_نتیجہ, عدد_elements * حجم(D), cuda_Memcpy_Device_مقصد_Host));
    صحیح nfaults = 0;
#اگر_غیر_تعریف_شدہ ENABLE_BF16
    اعشاری epsilon = FLT_EPSILON;
#ورنہ
    اعشاری epsilon = 0.079;
#اگر_ختم
    برائے (صحیح i = 0; i < عدد_elements; i++) {
        // Skip masked elements
        اگر(!isfinite(cpu_reference[i]))
            جاری;

        // print the first few comparisons
        اگر (i < 5) {
            لکھو("%f %f\n", cpu_reference[i], (T)برآمد_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        اعشاری t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        اگر (fabs(cpu_reference[i] - (T)برآمد_gpu[i]) > t_eff) {
            لکھو("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)برآمد_gpu[i]);
            nfaults ++;
            اگر (nfaults >= 10) {
                آزاد(برآمد_gpu);
                باہر_نکلو(EXIT_FAILURE);
            }
        }
    }

    اگر (nfaults > 0) {
        آزاد(برآمد_gpu);
        باہر_نکلو(EXIT_FAILURE);
    }

    آزاد(برآمد_gpu);
}

سانچہ<class Kernel, class... Kernel_دلائل>
اعشاری benchmark_kernel(صحیح repeats, Kernel kernel, Kernel_دلائل&&... kernel_دلائل) {
    cudaEvent_t آغاز, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    صحیح deviceIdx = 0;
    cudaCheck(cuda_سیٹ_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    خالی* flush_بفر;
    cudaCheck(cudaMalloc(&flush_بفر, deviceProp.l2_Cache_سائز));

    cudaCheck(cudaEventCreate(&آغاز));
    cudaCheck(cudaEventCreate(&stop));
    اعشاری elapsed_وقت = 0.f;
    برائے (صحیح i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_بفر, 0, deviceProp.l2_Cache_سائز));
        // now we can start recording the timing of the kernel
        cudaCheck(cuda_Event_ریکارڈ(آغاز, nullptr));
        kernel(std::forward<Kernel_دلائل>(kernel_دلائل)...);
        cudaCheck(cuda_Event_ریکارڈ(stop, nullptr));
        cudaCheck(cudaEventSynchronize(آغاز));
        cudaCheck(cudaEventSynchronize(stop));
        اعشاری single_call;
        cudaCheck(cuda_Event_Elapsed_وقت(&single_call, آغاز, stop));
        elapsed_وقت += single_call;
    }

    cudaCheck(cudaFree(flush_بفر));

    واپس elapsed_وقت / repeats;
}