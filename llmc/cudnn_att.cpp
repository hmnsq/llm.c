// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#وصف_ڪريو NOMINMAX
#ضمن <unistd.h>
#ضمن "cudnn_att.h"
#ضمن <cudnn_frontend.h>

نطاق fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
static_assert(خطأ, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
#وصف_ڪريو CUDNN_16_ٻٽو_انگ fe::معلومات_Type_t::HALF
#وإلا // Default to bfloat16
#وصف_ڪريو CUDNN_16_ٻٽو_انگ fe::معلومات_Type_t::BFLOAT16
#جيڪڏهن_پورو

ساكن cudnnHandle_t cudnn_handle;
ساكن size_t cudnn_workspace_سائيز = 0; // dynamically allocated as needed (up to 256MiB!)
ساكن فراغ* cudnn_workspace = عدم;

ساكن فراغ cuDNNCheck(cudnn_حالت_t error, ثابت حرف *file, صحيح سٽ) {
    إذا (error != CUDNN_حالت_SUCCESS) {
        لکو("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سٽ, cudnnGetErrorString(error));
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
};
#وصف_ڪريو cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ساكن فراغ checkCudnnFE(ثابت fe::error_object& e, ثابت حرف *file, صحيح سٽ) {
    إذا(!e.is_good()) {
        لکو("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سٽ, e.err_نياپو.c_str());
        ٻاهر_نڪرو(EXIT_FAILURE);
    }
}
#وصف_ڪريو checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

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
استعمال cache_type_fwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح, صحيح>, std::shared_ptr<fe::گراف::گراف_2>>;
استعمال cache_type_bwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح>, std::shared_ptr<fe::گراف::گراف_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
خودڪار lookup_cache_or_build_گراف_fwd(صحيح B,صحيح H,صحيح T,صحيح HS, صحيح is_inference_only) {

    ساكن cache_type_fwd user_maintained_cache_fwd;

    خودڪار ڪنجي = std::make_tuple(B, H, T, HS, is_inference_only);

    خودڪار it = user_maintained_cache_fwd.find(ڪنجي);
    إذا (it != user_maintained_cache_fwd.end()) {
        أرجع it->second;
    }

    خودڪار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->سيٽ_io_معلومات_type(CUDNN_16_ٻٽو_انگ)
          .سيٽ_intermediate_معلومات_type(fe::معلومات_Type_t::FLOAT)
          .سيٽ_compute_معلومات_type(fe::معلومات_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    خودڪار Q = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("Q")
                               .سيٽ_dim({B, H, T, HS})
                               .سيٽ_uid(Q_UID)
                               .سيٽ_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    خودڪار K = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("K")
                               .سيٽ_dim({B, H, T, HS})
                               .سيٽ_uid(K_UID)
                               .سيٽ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودڪار V = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("V")
                               .سيٽ_dim({B, H, T, HS})
                               .سيٽ_uid(V_UID)
                               .سيٽ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودڪار attn_scale = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("attn_scale")
                               .سيٽ_dim({1, 1, 1, 1})
                               .سيٽ_stride({1, 1, 1, 1})
                               .سيٽ_uid(Attn_scale_UID)
                               .سيٽ_is_pass_by_قدر(سچ)
                               .سيٽ_معلومات_type(fe::معلومات_Type_t::FLOAT));

    خودڪار sdpa_options = fe::گراف::SDPA_خاصيتون().سيٽ_name("flash_attention");
    sdpa_options.سيٽ_is_inference(is_inference_only);
    sdpa_options.سيٽ_attn_scale(attn_scale);
    sdpa_options.سيٽ_causal_نقاب(سچ);

    // Create the graph operation and get the output tensors back
    خودڪار [O, stats] = گراف->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->سيٽ_output(سچ).سيٽ_dim({B, H, T, HS}).سيٽ_stride({H * HS * T, HS, H * HS, 1}).سيٽ_uid(O_UID);

    أكد(stats == خالي_اشارو || is_inference_only == خطأ);
    إذا (is_inference_only == خطأ) {
        stats->سيٽ_output(سچ).سيٽ_معلومات_type(fe::معلومات_Type_t::FLOAT)
                               .سيٽ_dim({B, H, T, 1})
                               .سيٽ_stride({H * T, T, 1, 1})
                               .سيٽ_uid(Stats_UID);
    }

    checkCudnnFE(گراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(گراف->build_operation_گراف(cudnn_handle));
    خودڪار plans = گراف->create_execution_plans({fe::Heur_گهڻائي_t::A});
    checkCudnnFE(گراف->check_support(cudnn_handle));
    checkCudnnFE(گراف->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    إذا (گراف->get_workspace_سائيز() > cudnn_workspace_سائيز) {
        إذا (cudnn_workspace_سائيز > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_سائيز = گراف->get_workspace_سائيز();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_سائيز));
    }

    user_maintained_cache_fwd.insert({ڪنجي, گراف});

    أرجع گراف;
}

خودڪار lookup_cache_or_build_گراف_bwd(صحيح B, صحيح NH, صحيح T, صحيح HS) {
    ساكن cache_type_bwd user_maintained_cache_bwd;

    خودڪار ڪنجي = std::make_tuple(B, NH, T, HS);

    خودڪار it = user_maintained_cache_bwd.find(ڪنجي);
    إذا (it != user_maintained_cache_bwd.end()) {
        أرجع it->second;
    }

    خودڪار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->سيٽ_io_معلومات_type(CUDNN_16_ٻٽو_انگ)
          .سيٽ_intermediate_معلومات_type(fe::معلومات_Type_t::FLOAT)
          .سيٽ_compute_معلومات_type(fe::معلومات_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    خودڪار Q = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("Q")
                            .سيٽ_dim({B, NH, T, HS})
                            .سيٽ_uid(Q_UID)
                            .سيٽ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودڪار K = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("K")
                            .سيٽ_dim({B, NH, T, HS})
                            .سيٽ_uid(K_UID)
                            .سيٽ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودڪار V = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("V")
                            .سيٽ_dim({B, NH, T, HS})
                            .سيٽ_uid(V_UID)
                            .سيٽ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودڪار O = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("O")
                            .سيٽ_dim({B, NH, T, HS})
                            .سيٽ_uid(O_UID)
                            .سيٽ_stride({NH * HS * T, HS, NH * HS, 1}));
    خودڪار dO = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("dO")
                            .سيٽ_dim({B, NH, T, HS})
                            .سيٽ_uid(dO_UID)
                            .سيٽ_stride({NH * HS * T, HS, NH * HS, 1}));

    خودڪار stats = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("stats")
                            .سيٽ_dim({B, NH, T, 1})
                            .سيٽ_uid(Stats_UID)
                            .سيٽ_stride({NH * T, T, 1, 1})
                            .سيٽ_معلومات_type(fe::معلومات_Type_t::FLOAT));
    خودڪار attn_scale = گراف->tensor(fe::گراف::Tensor_خاصيتون().سيٽ_name("attn_scale")
                            .سيٽ_dim({1, 1, 1, 1})
                            .سيٽ_stride({1, 1, 1, 1})
                            .سيٽ_is_pass_by_قدر(سچ)
                            .سيٽ_uid(Attn_scale_UID)
                            .سيٽ_معلومات_type(fe::معلومات_Type_t::FLOAT));
    خودڪار sdpa_backward_options = fe::گراف::SDPA_backward_خاصيتون().سيٽ_name("flash_attention_backward")
#إذا CUDNN_FRONTEND_MAJOR_نسخو > 1 || CUDNN_FRONTEND_MINOR_نسخو >= 5
                            .سيٽ_deterministic_algorithm(سچ) // 1.5+ needs this for determinism
#جيڪڏهن_پورو
                            .سيٽ_causal_نقاب(سچ)
                            .سيٽ_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    خودڪار [dQ, dK, dV] = گراف->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->سيٽ_output(سچ).سيٽ_dim({B, NH, T, HS}).سيٽ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سيٽ_uid(dQ_UID);
    dK->سيٽ_output(سچ).سيٽ_dim({B, NH, T, HS}).سيٽ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سيٽ_uid(dK_UID);
    dV->سيٽ_output(سچ).سيٽ_dim({B, NH, T, HS}).سيٽ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سيٽ_uid(dV_UID);

    checkCudnnFE(گراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(گراف->build_operation_گراف(cudnn_handle));
    خودڪار plans = گراف->create_execution_plans({fe::Heur_گهڻائي_t::A});
    checkCudnnFE(گراف->check_support(cudnn_handle));
    checkCudnnFE(گراف->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    إذا (گراف->get_workspace_سائيز() > cudnn_workspace_سائيز) {
        إذا (cudnn_workspace_سائيز > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_سائيز = گراف->get_workspace_سائيز();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_سائيز));
    }

    user_maintained_cache_bwd.insert({ڪنجي, گراف});
    أرجع گراف;
}

فراغ attention_forward_cudnn(floatX* ٻاهريون,  // output: (B, T, NH, HS)
                             عائم* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحيح B, صحيح T, صحيح NH, صحيح C, cuda_وهڪرو_t وهڪرو) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head
    منطقي is_inference_only = (stats == خالي_اشارو);

    cuDNNCheck(cudnn_سيٽ_وهڪرو(cudnn_handle, وهڪرو));

    // Get graph and tensors from cache (or generate it on first use)
    خودڪار گراف = lookup_cache_or_build_گراف_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    فراغ* devPtrQ = inp;
    فراغ* devPtrK = (inp + C);
    فراغ* devPtrV = (inp + 2 * C);
    عائم attn_scale_cpu = 1.0 / sqrtf(HS);
    فراغ* devPtrO = ٻاهريون;

    // Build variant pack
    std::unordered_map<int64_t , فراغ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    إذا (is_inference_only == خطأ) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخري_Error());
}

فراغ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, عائم* stats, // inputs
                              صحيح B, صحيح T, صحيح NH, صحيح C, cuda_وهڪرو_t وهڪرو) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    خودڪار گراف = lookup_cache_or_build_گراف_bwd(B, NH, T, HS);

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
    cuDNNCheck(cudnn_سيٽ_وهڪرو(cudnn_handle, وهڪرو));
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخري_Error());
}

فراغ create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

فراغ destroy_cudnn() {
    إذا (cudnn_workspace != عدم) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}