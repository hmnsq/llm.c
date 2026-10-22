// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#تعریف_کړه NOMINMAX
#ضمن <unistd.h>
#ضمن "cudnn_att.h"
#ضمن <cudnn_frontend.h>

نطاق fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
static_assert(خطأ, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
#تعریف_کړه CUDNN_16_بټ fe::معلومات_Type_t::HALF
#وإلا // Default to bfloat16
#تعریف_کړه CUDNN_16_بټ fe::معلومات_Type_t::BFLOAT16
#د_که_پای

ساكن cudnnHandle_t cudnn_handle;
ساكن size_t cudnn_workspace_اندازه = 0; // dynamically allocated as needed (up to 256MiB!)
ساكن فراغ* cudnn_workspace = عدم;

ساكن فراغ cuDNNCheck(cudnn_حالت_t error, ثابت حرف *file, صحيح کرښه) {
    إذا (error != CUDNN_حالت_SUCCESS) {
        چاپ_کړه("[CUDNN ERROR] at file %s:%d:\n%s\n", file, کرښه, cudnnGetErrorString(error));
        وتل(EXIT_FAILURE);
    }
};
#تعریف_کړه cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ساكن فراغ checkCudnnFE(ثابت fe::error_object& e, ثابت حرف *file, صحيح کرښه) {
    إذا(!e.is_good()) {
        چاپ_کړه("[CUDNN ERROR] at file %s:%d:\n%s\n", file, کرښه, e.err_پیغام.c_str());
        وتل(EXIT_FAILURE);
    }
}
#تعریف_کړه checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

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
کارول cache_type_fwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح, صحيح>, std::shared_ptr<fe::ګراف::ګراف_2>>;
کارول cache_type_bwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح>, std::shared_ptr<fe::ګراف::ګراف_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
خپلکاری lookup_cache_or_build_ګراف_fwd(صحيح B,صحيح H,صحيح T,صحيح HS, صحيح is_inference_only) {

    ساكن cache_type_fwd user_maintained_cache_fwd;

    خپلکاری کلی = std::make_tuple(B, H, T, HS, is_inference_only);

    خپلکاری it = user_maintained_cache_fwd.find(کلی);
    إذا (it != user_maintained_cache_fwd.end()) {
        أرجع it->second;
    }

    خپلکاری ګراف = std::make_shared<fe::ګراف::ګراف_2>();
    ګراف->سېټ_io_معلومات_type(CUDNN_16_بټ)
          .سېټ_intermediate_معلومات_type(fe::معلومات_Type_t::FLOAT)
          .سېټ_compute_معلومات_type(fe::معلومات_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    خپلکاری Q = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("Q")
                               .سېټ_dim({B, H, T, HS})
                               .سېټ_uid(Q_UID)
                               .سېټ_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    خپلکاری K = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("K")
                               .سېټ_dim({B, H, T, HS})
                               .سېټ_uid(K_UID)
                               .سېټ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خپلکاری V = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("V")
                               .سېټ_dim({B, H, T, HS})
                               .سېټ_uid(V_UID)
                               .سېټ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خپلکاری attn_scale = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("attn_scale")
                               .سېټ_dim({1, 1, 1, 1})
                               .سېټ_stride({1, 1, 1, 1})
                               .سېټ_uid(Attn_scale_UID)
                               .سېټ_is_pass_by_ارزښت(ریښتیا)
                               .سېټ_معلومات_type(fe::معلومات_Type_t::FLOAT));

    خپلکاری sdpa_options = fe::ګراف::SDPA_ځانګړتیاوې().سېټ_name("flash_attention");
    sdpa_options.سېټ_is_inference(is_inference_only);
    sdpa_options.سېټ_attn_scale(attn_scale);
    sdpa_options.سېټ_causal_ماسک(ریښتیا);

    // Create the graph operation and get the output tensors back
    خپلکاری [O, stats] = ګراف->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->سېټ_output(ریښتیا).سېټ_dim({B, H, T, HS}).سېټ_stride({H * HS * T, HS, H * HS, 1}).سېټ_uid(O_UID);

    أكد(stats == تش_اشاره || is_inference_only == خطأ);
    إذا (is_inference_only == خطأ) {
        stats->سېټ_output(ریښتیا).سېټ_معلومات_type(fe::معلومات_Type_t::FLOAT)
                               .سېټ_dim({B, H, T, 1})
                               .سېټ_stride({H * T, T, 1, 1})
                               .سېټ_uid(Stats_UID);
    }

    checkCudnnFE(ګراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(ګراف->build_operation_ګراف(cudnn_handle));
    خپلکاری plans = ګراف->create_execution_plans({fe::Heur_ډېرپېښ_t::A});
    checkCudnnFE(ګراف->check_support(cudnn_handle));
    checkCudnnFE(ګراف->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    إذا (ګراف->get_workspace_اندازه() > cudnn_workspace_اندازه) {
        إذا (cudnn_workspace_اندازه > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_اندازه = ګراف->get_workspace_اندازه();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_اندازه));
    }

    user_maintained_cache_fwd.insert({کلی, ګراف});

    أرجع ګراف;
}

خپلکاری lookup_cache_or_build_ګراف_bwd(صحيح B, صحيح NH, صحيح T, صحيح HS) {
    ساكن cache_type_bwd user_maintained_cache_bwd;

    خپلکاری کلی = std::make_tuple(B, NH, T, HS);

    خپلکاری it = user_maintained_cache_bwd.find(کلی);
    إذا (it != user_maintained_cache_bwd.end()) {
        أرجع it->second;
    }

    خپلکاری ګراف = std::make_shared<fe::ګراف::ګراف_2>();
    ګراف->سېټ_io_معلومات_type(CUDNN_16_بټ)
          .سېټ_intermediate_معلومات_type(fe::معلومات_Type_t::FLOAT)
          .سېټ_compute_معلومات_type(fe::معلومات_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    خپلکاری Q = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("Q")
                            .سېټ_dim({B, NH, T, HS})
                            .سېټ_uid(Q_UID)
                            .سېټ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خپلکاری K = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("K")
                            .سېټ_dim({B, NH, T, HS})
                            .سېټ_uid(K_UID)
                            .سېټ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خپلکاری V = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("V")
                            .سېټ_dim({B, NH, T, HS})
                            .سېټ_uid(V_UID)
                            .سېټ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خپلکاری O = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("O")
                            .سېټ_dim({B, NH, T, HS})
                            .سېټ_uid(O_UID)
                            .سېټ_stride({NH * HS * T, HS, NH * HS, 1}));
    خپلکاری dO = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("dO")
                            .سېټ_dim({B, NH, T, HS})
                            .سېټ_uid(dO_UID)
                            .سېټ_stride({NH * HS * T, HS, NH * HS, 1}));

    خپلکاری stats = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("stats")
                            .سېټ_dim({B, NH, T, 1})
                            .سېټ_uid(Stats_UID)
                            .سېټ_stride({NH * T, T, 1, 1})
                            .سېټ_معلومات_type(fe::معلومات_Type_t::FLOAT));
    خپلکاری attn_scale = ګراف->tensor(fe::ګراف::Tensor_ځانګړتیاوې().سېټ_name("attn_scale")
                            .سېټ_dim({1, 1, 1, 1})
                            .سېټ_stride({1, 1, 1, 1})
                            .سېټ_is_pass_by_ارزښت(ریښتیا)
                            .سېټ_uid(Attn_scale_UID)
                            .سېټ_معلومات_type(fe::معلومات_Type_t::FLOAT));
    خپلکاری sdpa_backward_options = fe::ګراف::SDPA_backward_ځانګړتیاوې().سېټ_name("flash_attention_backward")
#إذا CUDNN_FRONTEND_MAJOR_نسخه > 1 || CUDNN_FRONTEND_MINOR_نسخه >= 5
                            .سېټ_deterministic_algorithm(ریښتیا) // 1.5+ needs this for determinism
#د_که_پای
                            .سېټ_causal_ماسک(ریښتیا)
                            .سېټ_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    خپلکاری [dQ, dK, dV] = ګراف->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->سېټ_output(ریښتیا).سېټ_dim({B, NH, T, HS}).سېټ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سېټ_uid(dQ_UID);
    dK->سېټ_output(ریښتیا).سېټ_dim({B, NH, T, HS}).سېټ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سېټ_uid(dK_UID);
    dV->سېټ_output(ریښتیا).سېټ_dim({B, NH, T, HS}).سېټ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سېټ_uid(dV_UID);

    checkCudnnFE(ګراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(ګراف->build_operation_ګراف(cudnn_handle));
    خپلکاری plans = ګراف->create_execution_plans({fe::Heur_ډېرپېښ_t::A});
    checkCudnnFE(ګراف->check_support(cudnn_handle));
    checkCudnnFE(ګراف->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    إذا (ګراف->get_workspace_اندازه() > cudnn_workspace_اندازه) {
        إذا (cudnn_workspace_اندازه > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_اندازه = ګراف->get_workspace_اندازه();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_اندازه));
    }

    user_maintained_cache_bwd.insert({کلی, ګراف});
    أرجع ګراف;
}

فراغ attention_forward_cudnn(floatX* بهر,  // output: (B, T, NH, HS)
                             عائم* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحيح B, صحيح T, صحيح NH, صحيح C, cuda_بهیر_t بهیر) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head
    منطقي is_inference_only = (stats == تش_اشاره);

    cuDNNCheck(cudnn_سېټ_بهیر(cudnn_handle, بهیر));

    // Get graph and tensors from cache (or generate it on first use)
    خپلکاری ګراف = lookup_cache_or_build_ګراف_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    فراغ* devPtrQ = inp;
    فراغ* devPtrK = (inp + C);
    فراغ* devPtrV = (inp + 2 * C);
    عائم attn_scale_cpu = 1.0 / sqrtf(HS);
    فراغ* devPtrO = بهر;

    // Build variant pack
    std::unordered_map<int64_t , فراغ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    إذا (is_inference_only == خطأ) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(ګراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_وروستی_Error());
}

فراغ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, عائم* stats, // inputs
                              صحيح B, صحيح T, صحيح NH, صحيح C, cuda_بهیر_t بهیر) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    خپلکاری ګراف = lookup_cache_or_build_ګراف_bwd(B, NH, T, HS);

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
    cuDNNCheck(cudnn_سېټ_بهیر(cudnn_handle, بهیر));
    checkCudnnFE(ګراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_وروستی_Error());
}

فراغ create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

فراغ destroy_cudnn() {
    إذا (cudnn_workspace != عدم) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}