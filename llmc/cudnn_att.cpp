// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#تعریف_کن NOMINMAX
#ضمن <unistd.h>
#ضمن "cudnn_att.h"
#ضمن <cudnn_frontend.h>

نطاق fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
static_assert(خطأ, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
#تعریف_کن CUDNN_16_بیت fe::داده_Type_t::HALF
#وإلا // Default to bfloat16
#تعریف_کن CUDNN_16_بیت fe::داده_Type_t::BFLOAT16
#پایان_اگر

ساكن cudnnHandle_t cudnn_handle;
ساكن size_t cudnn_workspace_اندازه = 0; // dynamically allocated as needed (up to 256MiB!)
ساكن فراغ* cudnn_workspace = عدم;

ساكن فراغ cuDNNCheck(cudnn_وضعیت_t error, ثابت حرف *file, صحيح سطر) {
    إذا (error != CUDNN_وضعیت_SUCCESS) {
        چاپ_کن("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سطر, cudnnGetErrorString(error));
        خروج(EXIT_FAILURE);
    }
};
#تعریف_کن cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ساكن فراغ checkCudnnFE(ثابت fe::error_object& e, ثابت حرف *file, صحيح سطر) {
    إذا(!e.is_good()) {
        چاپ_کن("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سطر, e.err_پیام.c_str());
        خروج(EXIT_FAILURE);
    }
}
#تعریف_کن checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

تعداد UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
استفاده cache_type_fwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح, صحيح>, std::shared_ptr<fe::گراف::گراف_2>>;
استفاده cache_type_bwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح>, std::shared_ptr<fe::گراف::گراف_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
خودکار lookup_cache_or_build_گراف_fwd(صحيح B,صحيح H,صحيح T,صحيح HS, صحيح is_inference_only) {

    ساكن cache_type_fwd user_maintained_cache_fwd;

    خودکار کلید = std::make_tuple(B, H, T, HS, is_inference_only);

    خودکار it = user_maintained_cache_fwd.find(کلید);
    إذا (it != user_maintained_cache_fwd.end()) {
        أرجع it->second;
    }

    خودکار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->مجموعه_io_داده_type(CUDNN_16_بیت)
          .مجموعه_intermediate_داده_type(fe::داده_Type_t::FLOAT)
          .مجموعه_compute_داده_type(fe::داده_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    خودکار Q = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("Q")
                               .مجموعه_dim({B, H, T, HS})
                               .مجموعه_uid(Q_UID)
                               .مجموعه_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    خودکار K = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("K")
                               .مجموعه_dim({B, H, T, HS})
                               .مجموعه_uid(K_UID)
                               .مجموعه_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودکار V = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("V")
                               .مجموعه_dim({B, H, T, HS})
                               .مجموعه_uid(V_UID)
                               .مجموعه_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودکار attn_scale = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("attn_scale")
                               .مجموعه_dim({1, 1, 1, 1})
                               .مجموعه_stride({1, 1, 1, 1})
                               .مجموعه_uid(Attn_scale_UID)
                               .مجموعه_is_pass_by_مقدار(درست)
                               .مجموعه_داده_type(fe::داده_Type_t::FLOAT));

    خودکار sdpa_options = fe::گراف::SDPA_صفات().مجموعه_name("flash_attention");
    sdpa_options.مجموعه_is_inference(is_inference_only);
    sdpa_options.مجموعه_attn_scale(attn_scale);
    sdpa_options.مجموعه_causal_ماسک(درست);

    // Create the graph operation and get the output tensors back
    خودکار [O, stats] = گراف->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->مجموعه_output(درست).مجموعه_dim({B, H, T, HS}).مجموعه_stride({H * HS * T, HS, H * HS, 1}).مجموعه_uid(O_UID);

    أكد(stats == اشاره_تهی || is_inference_only == خطأ);
    إذا (is_inference_only == خطأ) {
        stats->مجموعه_output(درست).مجموعه_داده_type(fe::داده_Type_t::FLOAT)
                               .مجموعه_dim({B, H, T, 1})
                               .مجموعه_stride({H * T, T, 1, 1})
                               .مجموعه_uid(Stats_UID);
    }

    checkCudnnFE(گراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(گراف->build_operation_گراف(cudnn_handle));
    خودکار plans = گراف->create_execution_plans({fe::Heur_نما_t::A});
    checkCudnnFE(گراف->check_support(cudnn_handle));
    checkCudnnFE(گراف->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    إذا (گراف->get_workspace_اندازه() > cudnn_workspace_اندازه) {
        إذا (cudnn_workspace_اندازه > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_اندازه = گراف->get_workspace_اندازه();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_اندازه));
    }

    user_maintained_cache_fwd.insert({کلید, گراف});

    أرجع گراف;
}

خودکار lookup_cache_or_build_گراف_bwd(صحيح B, صحيح NH, صحيح T, صحيح HS) {
    ساكن cache_type_bwd user_maintained_cache_bwd;

    خودکار کلید = std::make_tuple(B, NH, T, HS);

    خودکار it = user_maintained_cache_bwd.find(کلید);
    إذا (it != user_maintained_cache_bwd.end()) {
        أرجع it->second;
    }

    خودکار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->مجموعه_io_داده_type(CUDNN_16_بیت)
          .مجموعه_intermediate_داده_type(fe::داده_Type_t::FLOAT)
          .مجموعه_compute_داده_type(fe::داده_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    خودکار Q = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("Q")
                            .مجموعه_dim({B, NH, T, HS})
                            .مجموعه_uid(Q_UID)
                            .مجموعه_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار K = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("K")
                            .مجموعه_dim({B, NH, T, HS})
                            .مجموعه_uid(K_UID)
                            .مجموعه_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار V = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("V")
                            .مجموعه_dim({B, NH, T, HS})
                            .مجموعه_uid(V_UID)
                            .مجموعه_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار O = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("O")
                            .مجموعه_dim({B, NH, T, HS})
                            .مجموعه_uid(O_UID)
                            .مجموعه_stride({NH * HS * T, HS, NH * HS, 1}));
    خودکار dO = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("dO")
                            .مجموعه_dim({B, NH, T, HS})
                            .مجموعه_uid(dO_UID)
                            .مجموعه_stride({NH * HS * T, HS, NH * HS, 1}));

    خودکار stats = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("stats")
                            .مجموعه_dim({B, NH, T, 1})
                            .مجموعه_uid(Stats_UID)
                            .مجموعه_stride({NH * T, T, 1, 1})
                            .مجموعه_داده_type(fe::داده_Type_t::FLOAT));
    خودکار attn_scale = گراف->tensor(fe::گراف::Tensor_صفات().مجموعه_name("attn_scale")
                            .مجموعه_dim({1, 1, 1, 1})
                            .مجموعه_stride({1, 1, 1, 1})
                            .مجموعه_is_pass_by_مقدار(درست)
                            .مجموعه_uid(Attn_scale_UID)
                            .مجموعه_داده_type(fe::داده_Type_t::FLOAT));
    خودکار sdpa_backward_options = fe::گراف::SDPA_backward_صفات().مجموعه_name("flash_attention_backward")
#إذا CUDNN_FRONTEND_MAJOR_نسخه > 1 || CUDNN_FRONTEND_MINOR_نسخه >= 5
                            .مجموعه_deterministic_algorithm(درست) // 1.5+ needs this for determinism
#پایان_اگر
                            .مجموعه_causal_ماسک(درست)
                            .مجموعه_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    خودکار [dQ, dK, dV] = گراف->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->مجموعه_output(درست).مجموعه_dim({B, NH, T, HS}).مجموعه_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).مجموعه_uid(dQ_UID);
    dK->مجموعه_output(درست).مجموعه_dim({B, NH, T, HS}).مجموعه_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).مجموعه_uid(dK_UID);
    dV->مجموعه_output(درست).مجموعه_dim({B, NH, T, HS}).مجموعه_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).مجموعه_uid(dV_UID);

    checkCudnnFE(گراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(گراف->build_operation_گراف(cudnn_handle));
    خودکار plans = گراف->create_execution_plans({fe::Heur_نما_t::A});
    checkCudnnFE(گراف->check_support(cudnn_handle));
    checkCudnnFE(گراف->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    إذا (گراف->get_workspace_اندازه() > cudnn_workspace_اندازه) {
        إذا (cudnn_workspace_اندازه > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_اندازه = گراف->get_workspace_اندازه();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_اندازه));
    }

    user_maintained_cache_bwd.insert({کلید, گراف});
    أرجع گراف;
}

فراغ attention_forward_cudnn(floatX* بیرون,  // output: (B, T, NH, HS)
                             عائم* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحيح B, صحيح T, صحيح NH, صحيح C, cuda_جریان_t جریان) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head
    منطقي is_inference_only = (stats == اشاره_تهی);

    cuDNNCheck(cudnn_مجموعه_جریان(cudnn_handle, جریان));

    // Get graph and tensors from cache (or generate it on first use)
    خودکار گراف = lookup_cache_or_build_گراف_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    فراغ* devPtrQ = inp;
    فراغ* devPtrK = (inp + C);
    فراغ* devPtrV = (inp + 2 * C);
    عائم attn_scale_cpu = 1.0 / sqrtf(HS);
    فراغ* devPtrO = بیرون;

    // Build variant pack
    std::unordered_map<int64_t , فراغ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    إذا (is_inference_only == خطأ) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخرین_Error());
}

فراغ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, عائم* stats, // inputs
                              صحيح B, صحيح T, صحيح NH, صحيح C, cuda_جریان_t جریان) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    خودکار گراف = lookup_cache_or_build_گراف_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    فراغ* devPtrQ = qkvr;
    فراغ* devPtrK = (qkvr + NH * HS);
    فراغ* devPtrV = (qkvr + 2 * NH * HS);
    فراغ* devPtrO = o;
    فراغ* devPtrdO = dout;
    فراغ* devPtrStats = stats;
    عائم attn_scale_cpu = 1.0 / sqrtf(HS);

    فراغ* devPtrdQ = dqkvr;
    فراغ* devPtrdK = (dqkvr + NH * HS);
    فراغ* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, فراغ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnn_مجموعه_جریان(cudnn_handle, جریان));
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخرین_Error());
}

فراغ create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

فراغ destroy_cudnn() {
    إذا (cudnn_workspace != عدم) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}