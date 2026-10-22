#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <cuda_runtime.h>
#అన్తర్భావ <cublas_v2.h>
#అన్తర్భావ <cublasLt.h>
#అన్తర్భావ <float.h>

#నిర్వచించు WARP_కొలత 32U
బాహ్య cudaDeviceProp deviceProp;

మూస<class T>
__host__ __device__ T ceil_div(T dividend, T భాజకం) {
    తిరిగి (dividend + భాజకం-1) / భాజకం;
}

__device__ దశాంశం warp_Reduce_కూడిక(దశాంశం val) {
    కొరకు (పూర్ణాంకం స్థానభ్రంశం = 16; స్థానభ్రంశం > 0; స్థానభ్రంశం /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, స్థానభ్రంశం);
    }
    తిరిగి val;
}

// requires all 32 threads in the warp to be active, but should work for any block size
// uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
// the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
// but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
using reduction_func_t = దశాంశం (*) (దశాంశం);

మూస<reduction_func_t warp_reduction>
__device__ అంతఃవరుస దశాంశం ఖండం_Reduce(దశాంశం val, bool final_sync, దశాంశం బహిర్గతం_యొక్క_bounds) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ దశాంశం shared_val[WARP_కొలత];
    స్థిరరాశి పూర్ణాంకం lane_id = threadIdx.x % WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం warp_id = threadIdx.x / WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం అంకె_warps = ఖండం_Dim.x / WARP_కొలత;

    దశాంశం warp_val = warp_reduction(val);
    ఒకవేళ (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < అంకె_warps) ? shared_val[lane_id] : బహిర్గతం_యొక్క_bounds;
    దశాంశం ఖండం_val = warp_reduction(warp_val);

    ఒకవేళ (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    తిరిగి ఖండం_val;
}

// Helper function to call blockReduce with default arguments
మూస<reduction_func_t warp_reduction>
__device__ అంతఃవరుస దశాంశం ఖండం_Reduce(దశాంశం val) {
    తిరిగి ఖండం_Reduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
శూన్య cuda_check(cudaError_t error, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (error != cudaSuccess) {
        రాయి("[CUDA ERROR] at file %s:%d:\n%s\n", file, పంక్తి,
               cudaGetErrorString(error));
        నిష్క్రమించు(EXIT_FAILURE);
    }
};
#నిర్వచించు cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
శూన్య cublasCheck(cublas_స్థితి_t స్థితి_2, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి)
{
    ఒకవేళ (స్థితి_2 != CUBLAS_స్థితి_SUCCESS) {
        రాయి("[cuBLAS ERROR]: %d %s %d\n", స్థితి_2, file, పంక్తి);
        నిష్క్రమించు(EXIT_FAILURE);
    }
}
#నిర్వచించు cublasCheck(స్థితి_2) { cublasCheck((స్థితి_2), __FILE__, __LINE__); }

// ----------------------------------------------------------------------------
// cuBLAS setup
// these will be initialized by setup_main

// cuBLAS workspace. Hardcoding to 32MiB but only Hopper needs 32, for others 4 is OK
స్థిర size_t cublaslt_workspace_కొలత = 32 * 1024 * 1024;
స్థిర శూన్య* cublaslt_workspace = ఖాళీ;
స్థిర cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
పూర్ణాంకం cuda_arch_major = 0;
పూర్ణాంకం cuda_arch_minor = 0;
పూర్ణాంకం cuda_అంకె_SMs = 0; // for persistent threads where we want 1 threadblock per SM
పూర్ణాంకం cuda_దారాలు_చొప్పున_SM = 0;    // needed to calculate how many blocks to launch to fill up the GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#ఒకవేళ __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#నిర్వచించు MAX_1024_దారాలు_ఖండాలు 2
#లేకపోతే
#నిర్వచించు MAX_1024_దారాలు_ఖండాలు 1
#అయితే_ముగింపు

// ----------------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

మూస<class ElementType>
నిర్మాణం alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
    Packed128() = అప్రమేయ;
    __device__ explicit Packed128(int4 బిట్లు) {
        static_assert(పరిమాణం(బిట్లు) == పరిమాణం(payload), "Size mismatch.");
        జ్ఞాపకం_నకలు(&payload, &బిట్లు, పరిమాణం(బిట్లు));
    }

    __device__  స్థిర Packed128 constant(ElementType విలువ) {
        Packed128 ఫలితం;
        కొరకు(పూర్ణాంకం k = 0; k < కొలత; ++k) {
            ఫలితం.payload[k] = విలువ;
        }
        తిరిగి ఫలితం;
    }

    __device__ స్థిర Packed128 zeros() {
        తిరిగి constant(0);
    }

    __device__ స్థిర Packed128 ones() {
        తిరిగి constant(1);
    }

    __device__ ElementType& క్రియాకారకం[](పూర్ణాంకం సూచీ) {
        తిరిగి payload[సూచీ];
    }
    __device__ స్థిరరాశి ElementType& క్రియాకారకం[](పూర్ణాంకం సూచీ) స్థిరరాశి {
        తిరిగి payload[సూచీ];
    }
    __device__ int4 get_బిట్లు() స్థిరరాశి {
        int4 బిట్లు;
        static_assert(పరిమాణం(బిట్లు) == పరిమాణం(payload), "Size mismatch.");
        జ్ఞాపకం_నకలు(&బిట్లు, &payload, పరిమాణం(బిట్లు));
        తిరిగి బిట్లు;
    }
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    స్థిర constexpr స్థిరరాశి పూర్ణాంకం కొలత = పరిమాణం(int4) / పరిమాణం(ElementType);
    ElementType payload[కొలత];
};

// short-form typedef
రకనిర్వచనం Packed128<దశాంశం> f128;

// load a Packed128 from an aligned memory address
మూస<class ElementType>
__device__ Packed128<ElementType> load128(స్థిరరాశి ElementType* చిరునామా) {
    తిరిగి Packed128<ElementType>{*reinterpret_cast<స్థిరరాశి int4*>(చిరునామా)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
మూస<class ElementType>
__device__ Packed128<ElementType> load128cs(స్థిరరాశి ElementType* చిరునామా) {
    తిరిగి Packed128<ElementType>{__ldcs(reinterpret_cast<స్థిరరాశి int4*>(చిరునామా))};
}
// store a Packed128 to an aligned memory address
మూస<class ElementType>
__device__ శూన్య store128(ElementType* లక్ష్యం, Packed128<ElementType> విలువ) {
    *reinterpret_cast<int4*>(లక్ష్యం) = విలువ.get_బిట్లు();
}
// store a Packed128 to an aligned memory address with streaming cache hint
మూస<class ElementType>
__device__ శూన్య store128cs(ElementType* లక్ష్యం, Packed128<ElementType> విలువ) {
    __stcs(reinterpret_cast<int4*>(లక్ష్యం), విలువ.get_బిట్లు());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
మూస<class ElementType>
__device__ శూన్య store128cg(ElementType* లక్ష్యం, Packed128<ElementType> విలువ) {
    __stcg(reinterpret_cast<int4*>(లక్ష్యం), విలువ.get_బిట్లు());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#ఒకవేళ defined(ENABLE_BF16)

రకనిర్వచనం __nv_bfloat16 floatX;
రకనిర్వచనం __nv_bfloat16 floatN;
#నిర్వచించు CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#నిర్వచించు CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#లేదావేళ defined(ENABLE_FP16)

రకనిర్వచనం half floatX;
రకనిర్వచనం half floatN;

#లేకపోతే

రకనిర్వచనం దశాంశం floatX;
రకనిర్వచనం దశాంశం floatN;
#అయితే_ముగింపు

రకనిర్వచనం Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#ఒకవేళ defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(స్థిరరాశి floatX* చిరునామా) {
    గుర్తులేని పొట్టి bf = __ldcs(reinterpret_cast<స్థిరరాశి గుర్తులేని పొట్టి*>(చిరునామా));
    తిరిగి __nv_bfloat16_raw{bf};
}

__device__ శూన్య __stcs(floatX* చిరునామా, floatX విలువ) {
    __stcs(reinterpret_cast<గుర్తులేని పొట్టి*>(చిరునామా), ((__nv_bfloat16_raw)విలువ).x);
}
#అయితే_ముగింపు


// ----------------------------------------------------------------------------
// random utils

దశాంశం* make_random_float_01(size_t N) {
    దశాంశం* arr = (దశాంశం*)జ్ఞాపకం_ఇవ్వు(N * పరిమాణం(దశాంశం));
    కొరకు (size_t i = 0; i < N; i++) {
        arr[i] = ((దశాంశం)rand() / RAND_MAX); // range 0..1
    }
    తిరిగి arr;
}

దశాంశం* make_random_float(size_t N) {
    దశాంశం* arr = (దశాంశం*)జ్ఞాపకం_ఇవ్వు(N * పరిమాణం(దశాంశం));
    కొరకు (size_t i = 0; i < N; i++) {
        arr[i] = ((దశాంశం)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    తిరిగి arr;
}

పూర్ణాంకం* make_random_int(size_t N, పూర్ణాంకం V) {
    పూర్ణాంకం* arr = (పూర్ణాంకం*)జ్ఞాపకం_ఇవ్వు(N * పరిమాణం(పూర్ణాంకం));
    కొరకు (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    తిరిగి arr;
}

దశాంశం* make_zeros_float(size_t N) {
    దశాంశం* arr = (దశాంశం*)జ్ఞాపకం_ఇవ్వు(N * పరిమాణం(దశాంశం));
    memset(arr, 0, N * పరిమాణం(దశాంశం)); // all zero
    తిరిగి arr;
}

దశాంశం* make_ones_float(size_t N) {
    దశాంశం* arr = (దశాంశం*)జ్ఞాపకం_ఇవ్వు(N * పరిమాణం(దశాంశం));
    కొరకు (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    తిరిగి arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

మూస<class లక్ష్యం_Type>
[[nodiscard]] cudaError_t memcpy_convert(లక్ష్యం_Type* d_ptr, దశాంశం* h_ptr, size_t లెక్క) {
    // copy from host to device with data type conversion.
    లక్ష్యం_Type* converted = (లక్ష్యం_Type*)జ్ఞాపకం_ఇవ్వు(లెక్క * పరిమాణం(లక్ష్యం_Type));
    కొరకు (పూర్ణాంకం i = 0; i < లెక్క; i++) {
        converted[i] = (లక్ష్యం_Type)h_ptr[i];
    }

    cudaError_t స్థితి_2 = cudaMemcpy(d_ptr, converted, లెక్క * పరిమాణం(లక్ష్యం_Type), cuda_Memcpy_Host_గమ్యం_Device);
    విడుదల(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    తిరిగి స్థితి_2;
}

శూన్య అమరిక_main() {
    srand(0);   // determinism

    // set up the device
    పూర్ణాంకం deviceIdx = 0;
    cudaCheck(cuda_సమితి_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_అంకె_SMs = deviceProp.multi_Processor_లెక్క;
    cuda_దారాలు_చొప్పున_SM = deviceProp.max_దారాలు_చొప్పున_Multi_Processor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_కొలత));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    పూర్ణాంకం enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_బహుళకం = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublas_సమితి_Math_బహుళకం(cublas_handle, cublas_math_బహుళకం));
}

మూస<class D, class T>
శూన్య validate_ఫలితం(D* device_ఫలితం, స్థిరరాశి T* cpu_reference, స్థిరరాశి అక్షరం* name, std::size_t అంకె_elements, T tolerance=1e-4) {
    D* బహిర్గతం_gpu = (D*)జ్ఞాపకం_ఇవ్వు(అంకె_elements * పరిమాణం(D));
    cudaCheck(cudaMemcpy(బహిర్గతం_gpu, device_ఫలితం, అంకె_elements * పరిమాణం(D), cuda_Memcpy_Device_గమ్యం_Host));
    పూర్ణాంకం nfaults = 0;
#నిర్వచితం_కాకపోతే ENABLE_BF16
    దశాంశం epsilon = FLT_EPSILON;
#లేకపోతే
    దశాంశం epsilon = 0.079;
#అయితే_ముగింపు
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_elements; i++) {
        // Skip masked elements
        ఒకవేళ(!isfinite(cpu_reference[i]))
            కొనసాగు;

        // print the first few comparisons
        ఒకవేళ (i < 5) {
            రాయి("%f %f\n", cpu_reference[i], (T)బహిర్గతం_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        దశాంశం t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        ఒకవేళ (fabs(cpu_reference[i] - (T)బహిర్గతం_gpu[i]) > t_eff) {
            రాయి("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)బహిర్గతం_gpu[i]);
            nfaults ++;
            ఒకవేళ (nfaults >= 10) {
                విడుదల(బహిర్గతం_gpu);
                నిష్క్రమించు(EXIT_FAILURE);
            }
        }
    }

    ఒకవేళ (nfaults > 0) {
        విడుదల(బహిర్గతం_gpu);
        నిష్క్రమించు(EXIT_FAILURE);
    }

    విడుదల(బహిర్గతం_gpu);
}

మూస<class Kernel, class... Kernel_వాదనలు>
దశాంశం benchmark_kernel(పూర్ణాంకం repeats, Kernel kernel, Kernel_వాదనలు&&... kernel_వాదనలు) {
    cudaEvent_t ప్రారంభం, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    పూర్ణాంకం deviceIdx = 0;
    cudaCheck(cuda_సమితి_Device(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    శూన్య* flush_బఫరు;
    cudaCheck(cudaMalloc(&flush_బఫరు, deviceProp.l2_Cache_కొలత));

    cudaCheck(cudaEventCreate(&ప్రారంభం));
    cudaCheck(cudaEventCreate(&stop));
    దశాంశం elapsed_సమయం = 0.f;
    కొరకు (పూర్ణాంకం i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_బఫరు, 0, deviceProp.l2_Cache_కొలత));
        // now we can start recording the timing of the kernel
        cudaCheck(cuda_Event_రికార్డు(ప్రారంభం, nullptr));
        kernel(std::forward<Kernel_వాదనలు>(kernel_వాదనలు)...);
        cudaCheck(cuda_Event_రికార్డు(stop, nullptr));
        cudaCheck(cudaEventSynchronize(ప్రారంభం));
        cudaCheck(cudaEventSynchronize(stop));
        దశాంశం single_call;
        cudaCheck(cuda_Event_Elapsed_సమయం(&single_call, ప్రారంభం, stop));
        elapsed_సమయం += single_call;
    }

    cudaCheck(cudaFree(flush_బఫరు));

    తిరిగి elapsed_సమయం / repeats;
}