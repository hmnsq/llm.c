// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#تعریف_کرو NOMINMAX
#شامل <unistd.h>
#شامل "cudnn_att.h"
#شامل <cudnn_frontend.h>

نامخلا fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#اگر defined(ENABLE_FP32)
static_assert(جھوٹ, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#وگرنہ_اگر defined(ENABLE_FP16)
#تعریف_کرو CUDNN_16_ثنائی_ہندسہ fe::ڈیٹا_Type_t::HALF
#ورنہ // Default to bfloat16
#تعریف_کرو CUDNN_16_ثنائی_ہندسہ fe::ڈیٹا_Type_t::BFLOAT16
#اگر_ختم

ساکن cudnnHandle_t cudnn_handle;
ساکن size_t cudnn_workspace_سائز = 0; // dynamically allocated as needed (up to 256MiB!)
ساکن خالی* cudnn_workspace = تہی;

ساکن خالی cuDNNCheck(cudnn_حالت_t error, ثابت حرف *file, صحیح سطر) {
    اگر (error != CUDNN_حالت_SUCCESS) {
        لکھو("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سطر, cudnnGetErrorString(error));
        باہر_نکلو(EXIT_FAILURE);
    }
};
#تعریف_کرو cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ساکن خالی checkCudnnFE(ثابت fe::error_object& e, ثابت حرف *file, صحیح سطر) {
    اگر(!e.is_good()) {
        لکھو("[CUDNN ERROR] at file %s:%d:\n%s\n", file, سطر, e.err_پیغام.c_str());
        باہر_نکلو(EXIT_FAILURE);
    }
}
#تعریف_کرو checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

شمار UIDs {
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
استعمال cache_type_fwd = std::map<std::tuple<صحیح,صحیح,صحیح,صحیح, صحیح>, std::shared_ptr<fe::گراف::گراف_2>>;
استعمال cache_type_bwd = std::map<std::tuple<صحیح,صحیح,صحیح,صحیح>, std::shared_ptr<fe::گراف::گراف_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
خودکار lookup_cache_or_build_گراف_fwd(صحیح B,صحیح H,صحیح T,صحیح HS, صحیح is_inference_only) {

    ساکن cache_type_fwd user_maintained_cache_fwd;

    خودکار کلید = std::make_tuple(B, H, T, HS, is_inference_only);

    خودکار it = user_maintained_cache_fwd.find(کلید);
    اگر (it != user_maintained_cache_fwd.end()) {
        واپس it->second;
    }

    خودکار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->سیٹ_io_ڈیٹا_type(CUDNN_16_ثنائی_ہندسہ)
          .سیٹ_intermediate_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT)
          .سیٹ_compute_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    خودکار Q = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("Q")
                               .سیٹ_dim({B, H, T, HS})
                               .سیٹ_uid(Q_UID)
                               .سیٹ_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    خودکار K = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("K")
                               .سیٹ_dim({B, H, T, HS})
                               .سیٹ_uid(K_UID)
                               .سیٹ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودکار V = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("V")
                               .سیٹ_dim({B, H, T, HS})
                               .سیٹ_uid(V_UID)
                               .سیٹ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    خودکار attn_scale = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("attn_scale")
                               .سیٹ_dim({1, 1, 1, 1})
                               .سیٹ_stride({1, 1, 1, 1})
                               .سیٹ_uid(Attn_scale_UID)
                               .سیٹ_is_pass_by_قدر(سچ)
                               .سیٹ_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT));

    خودکار sdpa_options = fe::گراف::SDPA_اوصاف().سیٹ_name("flash_attention");
    sdpa_options.سیٹ_is_inference(is_inference_only);
    sdpa_options.سیٹ_attn_scale(attn_scale);
    sdpa_options.سیٹ_causal_نقاب(سچ);

    // Create the graph operation and get the output tensors back
    خودکار [O, stats] = گراف->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->سیٹ_output(سچ).سیٹ_dim({B, H, T, HS}).سیٹ_stride({H * HS * T, HS, H * HS, 1}).سیٹ_uid(O_UID);

    أكد(stats == خالی_اشاریہ || is_inference_only == جھوٹ);
    اگر (is_inference_only == جھوٹ) {
        stats->سیٹ_output(سچ).سیٹ_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT)
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
    اگر (گراف->get_workspace_سائز() > cudnn_workspace_سائز) {
        اگر (cudnn_workspace_سائز > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_سائز = گراف->get_workspace_سائز();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_سائز));
    }

    user_maintained_cache_fwd.insert({کلید, گراف});

    واپس گراف;
}

خودکار lookup_cache_or_build_گراف_bwd(صحیح B, صحیح NH, صحیح T, صحیح HS) {
    ساکن cache_type_bwd user_maintained_cache_bwd;

    خودکار کلید = std::make_tuple(B, NH, T, HS);

    خودکار it = user_maintained_cache_bwd.find(کلید);
    اگر (it != user_maintained_cache_bwd.end()) {
        واپس it->second;
    }

    خودکار گراف = std::make_shared<fe::گراف::گراف_2>();
    گراف->سیٹ_io_ڈیٹا_type(CUDNN_16_ثنائی_ہندسہ)
          .سیٹ_intermediate_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT)
          .سیٹ_compute_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    خودکار Q = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("Q")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(Q_UID)
                            .سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار K = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("K")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(K_UID)
                            .سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار V = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("V")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(V_UID)
                            .سیٹ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    خودکار O = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("O")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(O_UID)
                            .سیٹ_stride({NH * HS * T, HS, NH * HS, 1}));
    خودکار dO = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("dO")
                            .سیٹ_dim({B, NH, T, HS})
                            .سیٹ_uid(dO_UID)
                            .سیٹ_stride({NH * HS * T, HS, NH * HS, 1}));

    خودکار stats = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("stats")
                            .سیٹ_dim({B, NH, T, 1})
                            .سیٹ_uid(Stats_UID)
                            .سیٹ_stride({NH * T, T, 1, 1})
                            .سیٹ_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT));
    خودکار attn_scale = گراف->tensor(fe::گراف::Tensor_اوصاف().سیٹ_name("attn_scale")
                            .سیٹ_dim({1, 1, 1, 1})
                            .سیٹ_stride({1, 1, 1, 1})
                            .سیٹ_is_pass_by_قدر(سچ)
                            .سیٹ_uid(Attn_scale_UID)
                            .سیٹ_ڈیٹا_type(fe::ڈیٹا_Type_t::FLOAT));
    خودکار sdpa_backward_options = fe::گراف::SDPA_backward_اوصاف().سیٹ_name("flash_attention_backward")
#اگر CUDNN_FRONTEND_MAJOR_نسخہ > 1 || CUDNN_FRONTEND_MINOR_نسخہ >= 5
                            .سیٹ_deterministic_algorithm(سچ) // 1.5+ needs this for determinism
#اگر_ختم
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
    اگر (گراف->get_workspace_سائز() > cudnn_workspace_سائز) {
        اگر (cudnn_workspace_سائز > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_سائز = گراف->get_workspace_سائز();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_سائز));
    }

    user_maintained_cache_bwd.insert({کلید, گراف});
    واپس گراف;
}

خالی attention_forward_cudnn(floatX* برآمد,  // output: (B, T, NH, HS)
                             اعشاری* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحیح B, صحیح T, صحیح NH, صحیح C, cuda_دھارا_t دھارا) {
    NVTX_RANGE_FN();
    صحیح HS = C / NH; // number of features per head
    منطقی is_inference_only = (stats == خالی_اشاریہ);

    cuDNNCheck(cudnn_سیٹ_دھارا(cudnn_handle, دھارا));

    // Get graph and tensors from cache (or generate it on first use)
    خودکار گراف = lookup_cache_or_build_گراف_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    خالی* devPtrQ = inp;
    خالی* devPtrK = (inp + C);
    خالی* devPtrV = (inp + 2 * C);
    اعشاری attn_scale_cpu = 1.0 / sqrtf(HS);
    خالی* devPtrO = برآمد;

    // Build variant pack
    std::unordered_map<int64_t , خالی*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    اگر (is_inference_only == جھوٹ) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخری_Error());
}

خالی attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, اعشاری* stats, // inputs
                              صحیح B, صحیح T, صحیح NH, صحیح C, cuda_دھارا_t دھارا) {
    NVTX_RANGE_FN();
    صحیح HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    خودکار گراف = lookup_cache_or_build_گراف_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    خالی* devPtrQ = qkvr;
    خالی* devPtrK = (qkvr + NH * HS);
    خالی* devPtrV = (qkvr + 2 * NH * HS);
    خالی* devPtrO = o;
    خالی* devPtrdO = dout;
    خالی* devPtrStats = stats;
    اعشاری attn_scale_cpu = 1.0 / sqrtf(HS);

    خالی* devPtrdQ = dqkvr;
    خالی* devPtrdK = (dqkvr + NH * HS);
    خالی* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, خالی*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnn_سیٹ_دھارا(cudnn_handle, دھارا));
    checkCudnnFE(گراف->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_آخری_Error());
}

خالی create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

خالی destroy_cudnn() {
    اگر (cudnn_workspace != تہی) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}