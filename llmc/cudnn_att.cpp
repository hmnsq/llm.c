// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#تعریف_کرو NOMINMAX
#ضمن <unistd.h>
#ضمن "cudnn_att.h"
#ضمن <cudnn_frontend.h>

نطاق fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#إذا defined(ENABLE_FP32)
static_assert(خطأ, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وإلاإذا defined(ENABLE_FP16)
#تعریف_کرو CUDNN_16_ثنائی_ہندسہ fe::انکڑے_Type_t::HALF
#وإلا // Default to bfloat16
#تعریف_کرو CUDNN_16_ثنائی_ہندسہ fe::انکڑے_Type_t::BFLOAT16
#جے_انت

ساكن cudnnHandle_t cudnn_handle;
ساكن size_t cudnn_workspace_ناپ = 0; // dynamically allocated as needed (up to 256MiB!)
ساكن فراغ* cudnn_workspace = عدم;

ساكن فراغ cuDNNCheck(cudnn_حالت_t error, ثابت حرف *file, صحيح سطر) {
    إذا (error != CUDNN_حالت_SUCCESS) {
        لکھو("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سطر, cudnnGetErrorString(error));
        باہر_نکلو(EXIT_FAILURE);
    }
};
#تعریف_کرو cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ساكن فراغ checkCudnnFE(ثابت fe::error_object& e, ثابت حرف *file, صحيح سطر) {
    إذا(!e.is_good()) {
        لکھو("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سطر, e.err_پیغام.c_str());
        باہر_نکلو(EXIT_FAILURE);
    }
}
#تعریف_کرو checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

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
ورتوں cache_type_fwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح, صحيح>, std::shared_ptr<fe::گراف::گراف_2>>;
ورتوں cache_type_bwd = std::map<std::tuple<صحيح,صحيح,صحيح,صحيح>, std::shared_ptr<fe::گراف::گراف_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
خودکار lookup_cache_or_build_گراف_fwd(صحيح B,صحيح H,صحيح T,صحيح HS, صحيح is_inference_only) {

    ساكن cache_type_fwd user_maintained_cache_fwd;

    خودکار کلید = std::make_tuple(B, H, T, HS, is_inference_only);

    خودکار it = user_maintained_cache_fwd.find(کلید);
    إذا (it != user_maintained_cache_fwd.end()) {
        أرجع it->second;
    }

    خودکار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->سیٹ_io_انکڑے_type(CUDNN_16_ثنائی_ہندسہ)
          .سیٹ_intermediate_انکڑے_type(fe::انکڑے_Type_t::FLOAT)
          .سیٹ_compute_انکڑے_type(fe::انکڑے_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    خودکار Q = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("Q")
                               .سیٹ_dim({B, H, T, HS})
                               .سیٹ_uid(Q_UID)
                               .سیٹ_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    خودکار K = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("K")
                               .سیٹ_dim({B, H, T, HS})
                               .سیٹ_uid(K_UID)
                               .سیٹ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودکار V = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("V")
                               .سیٹ_dim({B, H, T, HS})
                               .سیٹ_uid(V_UID)
                               .سیٹ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودکار attn_scale = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("attn_scale")
                               .سیٹ_dim({1, 1, 1, 1})
                               .سیٹ_stride({1, 1, 1, 1})
                               .سیٹ_uid(Attn_scale_UID)
                               .سیٹ_is_pass_by_قدر(سچ)
                               .سیٹ_انکڑے_type(fe::انکڑے_Type_t::FLOAT));

    خودکار sdpa_options = fe::گراف::SDPA_گن().سیٹ_name("flash_attention");
    sdpa_options.سیٹ_is_inference(is_inference_only);
    sdpa_options.سیٹ_attn_scale(attn_scale);
    sdpa_options.سیٹ_causal_نقاب(سچ);

    // Create the graph operation and get the output tensors back
    خودکار [O, stats] = گراف->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->سیٹ_output(سچ).سیٹ_dim({B, H, T, HS}).سیٹ_stride({H * HS * T, HS, H * HS, 1}).سیٹ_uid(O_UID);

    أكد(stats == خالی_سوچک || is_inference_only == خطأ);
    إذا (is_inference_only == خطأ) {
        stats->سیٹ_output(سچ).سیٹ_انکڑے_type(fe::انکڑے_Type_t::FLOAT)
                               .سیٹ_dim({B, H, T, 1})
                               .سیٹ_stride({H * T, T, 1, 1})
                               .سیٹ_uid(Stats_UID);
    }

    checkCudnnFE(گراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(گراف->build_operation_گراف(cudnn_handle));
    خودکار plans = گراف->create_execution_plans({fe::Heur_منوال_t::A});
    checkCudnnFE(گراف->check_support(cudnn_handle));
    checkCudnnFE(گراف->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    إذا (گراف->get_workspace_ناپ() > cudnn_workspace_ناپ) {
        إذا (cudnn_workspace_ناپ > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_ناپ = گراف->get_workspace_ناپ();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_ناپ));
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
    گراف->سیٹ_io_انکڑے_type(CUDNN_16_ثنائی_ہندسہ)
          .سیٹ_intermediate_انکڑے_type(fe::انکڑے_Type_t::FLOAT)
          .سیٹ_compute_انکڑے_type(fe::انکڑے_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    خودکار Q = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("Q")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(Q_UID)
                            .سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار K = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("K")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(K_UID)
                            .سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار V = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("V")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(V_UID)
                            .سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار O = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("O")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(O_UID)
                            .سیٹ_stride({NH * HS * T, HS, NH * HS, 1}));
    خودکار dO = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("dO")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(dO_UID)
                            .سیٹ_stride({NH * HS * T, HS, NH * HS, 1}));

    خودکار stats = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("stats")
                            .سیٹ_dim({B, NH, T, 1})
                            .سیٹ_uid(Stats_UID)
                            .سیٹ_stride({NH * T, T, 1, 1})
                            .سیٹ_انکڑے_type(fe::انکڑے_Type_t::FLOAT));
    خودکار attn_scale = گراف->tensor(fe::گراف::Tensor_گن().سیٹ_name("attn_scale")
                            .سیٹ_dim({1, 1, 1, 1})
                            .سیٹ_stride({1, 1, 1, 1})
                            .سیٹ_is_pass_by_قدر(سچ)
                            .سیٹ_uid(Attn_scale_UID)
                            .سیٹ_انکڑے_type(fe::انکڑے_Type_t::FLOAT));
    خودکار sdpa_backward_options = fe::گراف::SDPA_backward_گن().سیٹ_name("flash_attention_backward")
#إذا CUDNN_FRONTEND_MAJOR_نسخہ > 1 || CUDNN_FRONTEND_MINOR_نسخہ >= 5
                            .سیٹ_deterministic_algorithm(سچ) // 1.5+ needs this for determinism
#جے_انت
                            .سیٹ_causal_نقاب(سچ)
                            .سیٹ_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    خودکار [dQ, dK, dV] = گراف->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->سیٹ_output(سچ).سیٹ_dim({B, NH, T, HS}).سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سیٹ_uid(dQ_UID);
    dK->سیٹ_output(سچ).سیٹ_dim({B, NH, T, HS}).سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سیٹ_uid(dK_UID);
    dV->سیٹ_output(سچ).سیٹ_dim({B, NH, T, HS}).سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).سیٹ_uid(dV_UID);

    checkCudnnFE(گراف->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(گراف->build_operation_گراف(cudnn_handle));
    خودکار plans = گراف->create_execution_plans({fe::Heur_منوال_t::A});
    checkCudnnFE(گراف->check_support(cudnn_handle));
    checkCudnnFE(گراف->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    إذا (گراف->get_workspace_ناپ() > cudnn_workspace_ناپ) {
        إذا (cudnn_workspace_ناپ > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_ناپ = گراف->get_workspace_ناپ();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_ناپ));
    }

    user_maintained_cache_bwd.insert({کلید, گراف});
    أرجع گراف;
}

فراغ attention_forward_cudnn(floatX* باہرلا,  // output: (B, T, NH, HS)
                             عائم* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحيح B, صحيح T, صحيح NH, صحيح C, cuda_دھارا_t دھارا) {
    NVTX_RANGE_FN();
    صحيح HS = C / NH; // number of features per head
    منطقي is_inference_only = (stats == خالی_سوچک);

    cuDNNCheck(cudnn_سیٹ_دھارا(cudnn_handle, دھارا));

    // Get graph and tensors from cache (or generate it on first use)
    خودکار گراف = lookup_cache_or_build_گراف_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    فراغ* devPtrQ = inp;
    فراغ* devPtrK = (inp + C);
    فراغ* devPtrV = (inp + 2 * C);
    عائم attn_scale_cpu = 1.0 / sqrtf(HS);
    فراغ* devPtrO = باہرلا;

    // Build variant pack
    std::unordered_map<int64_t , فراغ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    إذا (is_inference_only == خطأ) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخری_Error());
}

فراغ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, عائم* stats, // inputs
                              صحيح B, صحيح T, صحيح NH, صحيح C, cuda_دھارا_t دھارا) {
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
    cuDNNCheck(cudnn_سیٹ_دھارا(cudnn_handle, دھارا));
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخری_Error());
}

فراغ create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

فراغ destroy_cudnn() {
    إذا (cudnn_workspace != عدم) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}