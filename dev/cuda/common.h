#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <cuda_runtime.h>
#अन्तर्भाव <cublas_v2.h>
#अन्तर्भाव <cublasLt.h>
#अन्तर्भाव <float.h>

#define WARP_SIZE 32U
बाह्य cudaDeviceProp deviceProp;

ढाँचा<class T>
__host__ __device__ T ceil_div(T dividend, T भाजक) {
    परत (dividend + भाजक-1) / भाजक;
}

__device__ दशांश warpReduceSum(दशांश val) {
    साठी (पूर्णांक offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    परत val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = दशांश (*) (दशांश);

ढाँचा<reduction_func_t warp_reduction>
__device__ ओळीत दशांश blockReduce(दशांश val, bool final_sync, दशांश out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ दशांश shared_val[WARP_SIZE];
    स्थिरांक पूर्णांक lane_id = threadIdx.x % WARP_SIZE;
    स्थिरांक पूर्णांक warp_id = threadIdx.x / WARP_SIZE;
    स्थिरांक पूर्णांक अंक_warps = blockDim.x / WARP_SIZE;

    दशांश warp_val = warp_reduction(val);
    जर (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < अंक_warps) ? shared_val[lane_id] : out_of_bounds;
    दशांश block_val = warp_reduction(warp_val);

    जर (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    परत block_val;
}

// Helper function to call blockReduce with default arguments
ढाँचा<reduction_func_t warp_reduction>
__device__ ओळीत दशांश blockReduce(दशांश val) {
    परत blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
शून्य cuda_check(cudaError_t error, स्थिरांक अक्षर *file, पूर्णांक ओळ) {
    जर (error != cudaSuccess) {
        म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, ओळ,
               cudaGetErrorString(error));
        निकास(निकास_त्रुटि);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
शून्य cublasCheck(cublasStatus_t status, स्थिरांक अक्षर *file, पूर्णांक ओळ)
{
    जर (status != CUBLAS_STATUS_SUCCESS) {
        म_लिखो("[cuBLAS ERROR]: %d %s %d\n", status, file, ओळ);
        निकास(निकास_त्रुटि);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
स्थिर माप_प्रकार cublaslt_workspace_size = 32 * 1024 * 1024;
स्थिर शून्य* cublaslt_workspace = रिक्त;
स्थिर cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
पूर्णांक cuda_arch_major = 0;
पूर्णांक cuda_arch_minor = 0;
पूर्णांक cuda_अंक_SMs = 0; // for persistent threads where we want 1 threadblock per SM
पूर्णांक cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#जर __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#अन्यथा
#define MAX_1024_THREADS_BLOCKS 1
#endif

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

ढाँचा<class ElementType>
संरचना alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = पूर्वनिर्धारित;
    __device__ explicit Packed128(int4 bits) {
        static_assert(आकार(bits) == आकार(payload), "Size mismatch.");
        स_नकल(&payload, &bits, आकार(bits));
    }

    __device__  स्थिर Packed128 constant(ElementType मूल्य) {
        Packed128 result;
        साठी(पूर्णांक k = 0; k < size; ++k) {
            result.payload[k] = मूल्य;
        }
        परत result;
    }

    __device__ स्थिर Packed128 zeros() {
        परत constant(0);
    }

    __device__ स्थिर Packed128 ones() {
        परत constant(1);
    }

    __device__ ElementType& चालक[](पूर्णांक index) {
        परत payload[index];
    }
    __device__ स्थिरांक ElementType& चालक[](पूर्णांक index) स्थिरांक {
        परत payload[index];
    }
    __device__ int4 get_bits() स्थिरांक {
        int4 bits;
        static_assert(आकार(bits) == आकार(payload), "Size mismatch.");
        स_नकल(&bits, &payload, आकार(bits));
        परत bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    स्थिर constexpr स्थिरांक पूर्णांक size = आकार(int4) / आकार(ElementType);
    ElementType payload[size];
};

// short-form typedef
प्रकारव्याख्या Packed128<दशांश> f128;

// load a Packed128 from an aligned memory address
ढाँचा<class ElementType>
__device__ Packed128<ElementType> load128(स्थिरांक ElementType* address) {
    परत Packed128<ElementType>{*reinterpret_cast<स्थिरांक int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
ढाँचा<class ElementType>
__device__ Packed128<ElementType> load128cs(स्थिरांक ElementType* address) {
    परत Packed128<ElementType>{__ldcs(reinterpret_cast<स्थिरांक int4*>(address))};
}
// store a Packed128 to an aligned memory address
ढाँचा<class ElementType>
__device__ शून्य store128(ElementType* target, Packed128<ElementType> मूल्य) {
    *reinterpret_cast<int4*>(target) = मूल्य.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
ढाँचा<class ElementType>
__device__ शून्य store128cs(ElementType* target, Packed128<ElementType> मूल्य) {
    __stcs(reinterpret_cast<int4*>(target), मूल्य.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
ढाँचा<class ElementType>
__device__ शून्य store128cg(ElementType* target, Packed128<ElementType> मूल्य) {
    __stcg(reinterpret_cast<int4*>(target), मूल्य.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#जर defined(ENABLE_BF16)

प्रकारव्याख्या __nv_bfloat16 floatX;
प्रकारव्याख्या __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#किंवाजर defined(ENABLE_FP16)

प्रकारव्याख्या half floatX;
प्रकारव्याख्या half floatN;

#अन्यथा

प्रकारव्याख्या दशांश floatX;
प्रकारव्याख्या दशांश floatN;
#endif

प्रकारव्याख्या Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#जर defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(स्थिरांक floatX* address) {
    अचिन्हित लघु bf = __ldcs(reinterpret_cast<स्थिरांक अचिन्हित लघु*>(address));
    परत __nv_bfloat16_raw{bf};
}

__device__ शून्य __stcs(floatX* address, floatX मूल्य) {
    __stcs(reinterpret_cast<अचिन्हित लघु*>(address), ((__nv_bfloat16_raw)मूल्य).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

दशांश* make_random_float_01(माप_प्रकार N) {
    दशांश* arr = (दशांश*)दो_स्मृति(N * आकार(दशांश));
    साठी (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((दशांश)अक्रम() / अक्रम_उच्च); // range 0..1
    }
    परत arr;
}

दशांश* make_random_float(माप_प्रकार N) {
    दशांश* arr = (दशांश*)दो_स्मृति(N * आकार(दशांश));
    साठी (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((दशांश)अक्रम() / अक्रम_उच्च) * 2.0 - 1.0; // range -1..1
    }
    परत arr;
}

पूर्णांक* make_random_int(माप_प्रकार N, पूर्णांक V) {
    पूर्णांक* arr = (पूर्णांक*)दो_स्मृति(N * आकार(पूर्णांक));
    साठी (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = अक्रम() % V; // range 0..V-1
    }
    परत arr;
}

दशांश* make_zeros_float(माप_प्रकार N) {
    दशांश* arr = (दशांश*)दो_स्मृति(N * आकार(दशांश));
    स_रखो(arr, 0, N * आकार(दशांश)); // all zero
    परत arr;
}

दशांश* make_ones_float(माप_प्रकार N) {
    दशांश* arr = (दशांश*)दो_स्मृति(N * आकार(दशांश));
    साठी (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    परत arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

ढाँचा<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, दशांश* h_ptr, माप_प्रकार मोजणी) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)दो_स्मृति(मोजणी * आकार(TargetType));
    साठी (पूर्णांक i = 0; i < मोजणी; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, मोजणी * आकार(TargetType), cudaMemcpyHostToDevice);
    मुक्त(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    परत status;
}

शून्य setup_main() {
    बेक्रम(0);   // determinism

    // set up the device
    पूर्णांक deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_अंक_SMs = deviceProp.multi_Processor_मोजणी;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    पूर्णांक enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

ढाँचा<class D, class T>
शून्य validate_result(D* device_result, स्थिरांक T* cpu_reference, स्थिरांक अक्षर* name, std::माप_प्रकार अंक_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)दो_स्मृति(अंक_elements * आकार(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, अंक_elements * आकार(D), cudaMemcpyDeviceToHost));
    पूर्णांक nfaults = 0;
#ifndef ENABLE_BF16
    दशांश epsilon = भग्न_अंतर;
#अन्यथा
    दशांश epsilon = 0.079;
#endif
    साठी (पूर्णांक i = 0; i < अंक_elements; i++) {
        // Skip masked elements
        जर(!है_सीमित(cpu_reference[i]))
            सुरूठेवा;

        // print the first few comparisons
        जर (i < 5) {
            म_लिखो("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        दशांश t_eff = tolerance + भ_असल(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        जर (भ_असल(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            म_लिखो("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            जर (nfaults >= 10) {
                मुक्त(out_gpu);
                निकास(निकास_त्रुटि);
            }
        }
    }

    जर (nfaults > 0) {
        मुक्त(out_gpu);
        निकास(निकास_त्रुटि);
    }

    मुक्त(out_gpu);
}

ढाँचा<class Kernel, class... KernelArgs>
दशांश benchmark_kernel(पूर्णांक repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    पूर्णांक deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    शून्य* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    दशांश elapsed_time = 0.f;
    साठी (पूर्णांक i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        दशांश single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    परत elapsed_time / repeats;
}