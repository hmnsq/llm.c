#समावेश <stdlib.h>
#समावेश <stdio.h>
#समावेश <cuda_runtime.h>
#समावेश <cublas_v2.h>
#समावेश <cublasLt.h>
#समावेश <float.h>

#घोषणा WARP_आकार 32U
बाह्य cudaDeviceProp deviceProp;

ढाँचा<class T>
__host__ __device__ T ceil_div(T dividend, T भाजक) {
    वापस (dividend + भाजक-1) / भाजक;
}

__device__ भग्न warp_Reduce_योग(भग्न val) {
    क्रम (पूर्णांक offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    वापस val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = भग्न (*) (भग्न);

ढाँचा<reduction_func_t warp_reduction>
__device__ अंतरभूत भग्न blockReduce(भग्न val, bool final_sync, भग्न out_of_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ भग्न shared_val[WARP_आकार];
    स्थिर पूर्णांक lane_id = threadIdx.x % WARP_आकार;
    स्थिर पूर्णांक warp_id = threadIdx.x / WARP_आकार;
    स्थिर पूर्णांक अंक_warps = blockDim.x / WARP_आकार;

    भग्न warp_val = warp_reduction(val);
    अगर (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < अंक_warps) ? shared_val[lane_id] : out_of_bounds;
    भग्न block_val = warp_reduction(warp_val);

    अगर (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    वापस block_val;
}

// Helper function to call blockReduce with default arguments
ढाँचा<reduction_func_t warp_reduction>
__device__ अंतरभूत भग्न blockReduce(भग्न val) {
    वापस blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
व्योम cuda_check(cudaError_t error, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (error != cudaSuccess) {
        म_लिखो("[CUDA ERROR] at file %s:%d:\n%s\n", file, पंक्ति,
               cudaGetErrorString(error));
        निकास(निकास_त्रुटि);
    }
};
#घोषणा cudaCheck(err) (cuda_check(err, __खाता__, __LINE__))

// cuBLAS error checking
व्योम cublasCheck(cublasStatus_t status, स्थिर अक्षर *file, पूर्णांक पंक्ति)
{
    अगर (status != CUBLAS_STATUS_SUCCESS) {
        म_लिखो("[cuBLAS ERROR]: %d %s %d\n", status, file, पंक्ति);
        निकास(निकास_त्रुटि);
    }
}
#घोषणा cublasCheck(status) { cublasCheck((status), __खाता__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
अटल माप_प्रकार cublaslt_workspace_आकार = 32 * 1024 * 1024;
अटल व्योम* cublaslt_workspace = शून्य;
अटल cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
पूर्णांक cuda_arch_major = 0;
पूर्णांक cuda_arch_minor = 0;
पूर्णांक cuda_अंक_SMs = 0; // for persistent threads where we want 1 threadblock per SM
पूर्णांक cuda_threads_per_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#अगर __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#घोषणा MAX_1024_THREADS_BLOCKS 2
#अन्यथा
#घोषणा MAX_1024_THREADS_BLOCKS 1
#पूर्ण_अगर

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

ढाँचा<class ElementType>
काष्ठा alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = शेष;
    __device__ explicit Packed128(int4 bits) {
        static_assert(माप(bits) == माप(payload), "Size mismatch.");
        स_नकल(&payload, &bits, माप(bits));
    }

    __device__  अटल Packed128 constant(ElementType मूल्य) {
        Packed128 परिणाम;
        क्रम(पूर्णांक k = 0; k < आकार; ++k) {
            परिणाम.payload[k] = मूल्य;
        }
        वापस परिणाम;
    }

    __device__ अटल Packed128 zeros() {
        वापस constant(0);
    }

    __device__ अटल Packed128 ones() {
        वापस constant(1);
    }

    __device__ ElementType& चालक[](पूर्णांक सूचकांक_2) {
        वापस payload[सूचकांक_2];
    }
    __device__ स्थिर ElementType& चालक[](पूर्णांक सूचकांक_2) स्थिर {
        वापस payload[सूचकांक_2];
    }
    __device__ int4 get_bits() स्थिर {
        int4 bits;
        static_assert(माप(bits) == माप(payload), "Size mismatch.");
        स_नकल(&bits, &payload, माप(bits));
        वापस bits;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    अटल constexpr स्थिर पूर्णांक आकार = माप(int4) / माप(ElementType);
    ElementType payload[आकार];
};

// short-form typedef
प्रकार Packed128<भग्न> f128;

// load a Packed128 from an aligned memory address
ढाँचा<class ElementType>
__device__ Packed128<ElementType> load128(स्थिर ElementType* address) {
    वापस Packed128<ElementType>{*reinterpret_cast<स्थिर int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
ढाँचा<class ElementType>
__device__ Packed128<ElementType> load128cs(स्थिर ElementType* address) {
    वापस Packed128<ElementType>{__ldcs(reinterpret_cast<स्थिर int4*>(address))};
}
// store a Packed128 to an aligned memory address
ढाँचा<class ElementType>
__device__ व्योम store128(ElementType* target, Packed128<ElementType> मूल्य) {
    *reinterpret_cast<int4*>(target) = मूल्य.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
ढाँचा<class ElementType>
__device__ व्योम store128cs(ElementType* target, Packed128<ElementType> मूल्य) {
    __stcs(reinterpret_cast<int4*>(target), मूल्य.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
ढाँचा<class ElementType>
__device__ व्योम store128cg(ElementType* target, Packed128<ElementType> मूल्य) {
    __stcg(reinterpret_cast<int4*>(target), मूल्य.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#अगर defined(ENABLE_BF16)

प्रकार __nv_bfloat16 floatX;
प्रकार __nv_bfloat16 floatN;
#घोषणा CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#घोषणा CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#या_अगर defined(ENABLE_FP16)

प्रकार half floatX;
प्रकार half floatN;

#अन्यथा

प्रकार भग्न floatX;
प्रकार भग्न floatN;
#पूर्ण_अगर

प्रकार Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#अगर defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(स्थिर floatX* address) {
    अचिन्हित लघु bf = __ldcs(reinterpret_cast<स्थिर अचिन्हित लघु*>(address));
    वापस __nv_bfloat16_raw{bf};
}

__device__ व्योम __stcs(floatX* address, floatX मूल्य) {
    __stcs(reinterpret_cast<अचिन्हित लघु*>(address), ((__nv_bfloat16_raw)मूल्य).x);
}
#पूर्ण_अगर


// ----------------------------------------------------------------------------
// random utils

भग्न* make_random_float_01(माप_प्रकार N) {
    भग्न* arr = (भग्न*)दो_स्मृति(N * माप(भग्न));
    क्रम (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((भग्न)अक्रम() / अक्रम_उच्च); // range 0..1
    }
    वापस arr;
}

भग्न* make_random_float(माप_प्रकार N) {
    भग्न* arr = (भग्न*)दो_स्मृति(N * माप(भग्न));
    क्रम (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((भग्न)अक्रम() / अक्रम_उच्च) * 2.0 - 1.0; // range -1..1
    }
    वापस arr;
}

पूर्णांक* make_random_int(माप_प्रकार N, पूर्णांक V) {
    पूर्णांक* arr = (पूर्णांक*)दो_स्मृति(N * माप(पूर्णांक));
    क्रम (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = अक्रम() % V; // range 0..V-1
    }
    वापस arr;
}

भग्न* make_zeros_float(माप_प्रकार N) {
    भग्न* arr = (भग्न*)दो_स्मृति(N * माप(भग्न));
    स_रखो(arr, 0, N * माप(भग्न)); // all zero
    वापस arr;
}

भग्न* make_ones_float(माप_प्रकार N) {
    भग्न* arr = (भग्न*)दो_स्मृति(N * माप(भग्न));
    क्रम (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    वापस arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

ढाँचा<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, भग्न* h_ptr, माप_प्रकार गिनती) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)दो_स्मृति(गिनती * माप(TargetType));
    क्रम (पूर्णांक i = 0; i < गिनती; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, गिनती * माप(TargetType), cudaMemcpyHostToDevice);
    मुक्त(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    वापस status;
}

व्योम setup_main() {
    बेक्रम(0);   // determinism

    // set up the device
    पूर्णांक deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_अंक_SMs = deviceProp.multi_Processor_गिनती;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_आकार));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    पूर्णांक enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

ढाँचा<class D, class T>
व्योम validate_परिणाम(D* device_परिणाम, स्थिर T* cpu_reference, स्थिर अक्षर* name, std::माप_प्रकार अंक_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)दो_स्मृति(अंक_elements * माप(D));
    cudaCheck(cudaMemcpy(out_gpu, device_परिणाम, अंक_elements * माप(D), cudaMemcpyDeviceToHost));
    पूर्णांक nfaults = 0;
#अगर_अघोषित ENABLE_BF16
    भग्न epsilon = भग्न_अंतर;
#अन्यथा
    भग्न epsilon = 0.079;
#पूर्ण_अगर
    क्रम (पूर्णांक i = 0; i < अंक_elements; i++) {
        // Skip masked elements
        अगर(!है_सीमित(cpu_reference[i]))
            जारी;

        // print the first few comparisons
        अगर (i < 5) {
            म_लिखो("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        भग्न t_eff = tolerance + भ_असल(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        अगर (भ_असल(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            म_लिखो("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            अगर (nfaults >= 10) {
                मुक्त(out_gpu);
                निकास(निकास_त्रुटि);
            }
        }
    }

    अगर (nfaults > 0) {
        मुक्त(out_gpu);
        निकास(निकास_त्रुटि);
    }

    मुक्त(out_gpu);
}

ढाँचा<class Kernel, class... KernelArgs>
भग्न benchmark_kernel(पूर्णांक repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    पूर्णांक deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    व्योम* flush_बफर;
    cudaCheck(cudaMalloc(&flush_बफर, deviceProp.l2_Cache_आकार));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    भग्न elapsed_time = 0.f;
    क्रम (पूर्णांक i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_बफर, 0, deviceProp.l2_Cache_आकार));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        भग्न single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_बफर));

    वापस elapsed_time / repeats;
}