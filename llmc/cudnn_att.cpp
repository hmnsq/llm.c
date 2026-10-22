// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOMINMAX
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "cudnn_att.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <cudnn_frontend.h>

ਨਾਮਾਕਾਸ਼ fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#ਜੇ defined(ENABLE_FP32)
static_assert(ਝੂਠ, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#ਜਾਂਜੇ defined(ENABLE_FP16)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDNN_16_ਦੋਆਧਾਰੀ_ਅੰਕ fe::ਅੰਕੜੇ_Type_t::HALF
#ਨਹੀਂਤਾਂ // Default to bfloat16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDNN_16_ਦੋਆਧਾਰੀ_ਅੰਕ fe::ਅੰਕੜੇ_Type_t::BFLOAT16
#ਜੇ_ਅੰਤ

ਸਥਿਰ cudnnHandle_t cudnn_handle;
ਸਥਿਰ size_t cudnn_workspace_ਮਾਪ = 0; // dynamically allocated as needed (up to 256MiB!)
ਸਥਿਰ ਖਾਲੀ* cudnn_workspace = ਨਿਰਾ;

ਸਥਿਰ ਖਾਲੀ cuDNNCheck(cudnn_ਸਥਿਤੀ_t error, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (error != CUDNN_ਸਥਿਤੀ_SUCCESS) {
        ਲਿਖੋ("[CUDNN ERROR] at file %s:%d:\n%s\n", file, ਸਤਰ, cudnnGetErrorString(error));
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
};
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ਸਥਿਰ ਖਾਲੀ checkCudnnFE(ਅਚਲ fe::error_object& e, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ(!e.is_good()) {
        ਲਿਖੋ("[CUDNN ERROR] at file %s:%d:\n%s\n", file, ਸਤਰ, e.err_ਸੁਨੇਹਾ.c_str());
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

ਗਿਣਤੀ UIDs {
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
ਵਰਤੋਂ cache_type_fwd = std::map<std::tuple<ਪੂਰਨ,ਪੂਰਨ,ਪੂਰਨ,ਪੂਰਨ, ਪੂਰਨ>, std::shared_ptr<fe::ਗ੍ਰਾਫ::ਗ੍ਰਾਫ_2>>;
ਵਰਤੋਂ cache_type_bwd = std::map<std::tuple<ਪੂਰਨ,ਪੂਰਨ,ਪੂਰਨ,ਪੂਰਨ>, std::shared_ptr<fe::ਗ੍ਰਾਫ::ਗ੍ਰਾਫ_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
ਸਵੈਚਾਲਿਤ lookup_cache_or_build_ਗ੍ਰਾਫ_fwd(ਪੂਰਨ B,ਪੂਰਨ H,ਪੂਰਨ T,ਪੂਰਨ HS, ਪੂਰਨ is_inference_only) {

    ਸਥਿਰ cache_type_fwd user_maintained_cache_fwd;

    ਸਵੈਚਾਲਿਤ ਕੁੰਜੀ = std::make_tuple(B, H, T, HS, is_inference_only);

    ਸਵੈਚਾਲਿਤ it = user_maintained_cache_fwd.find(ਕੁੰਜੀ);
    ਜੇ (it != user_maintained_cache_fwd.end()) {
        ਵਾਪਸ it->second;
    }

    ਸਵੈਚਾਲਿਤ ਗ੍ਰਾਫ = std::make_shared<fe::ਗ੍ਰਾਫ::ਗ੍ਰਾਫ_2>();
    ਗ੍ਰਾਫ->ਸੈੱਟ_io_ਅੰਕੜੇ_type(CUDNN_16_ਦੋਆਧਾਰੀ_ਅੰਕ)
          .ਸੈੱਟ_intermediate_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT)
          .ਸੈੱਟ_compute_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    ਸਵੈਚਾਲਿਤ Q = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("Q")
                               .ਸੈੱਟ_dim({B, H, T, HS})
                               .ਸੈੱਟ_uid(Q_UID)
                               .ਸੈੱਟ_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    ਸਵੈਚਾਲਿਤ K = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("K")
                               .ਸੈੱਟ_dim({B, H, T, HS})
                               .ਸੈੱਟ_uid(K_UID)
                               .ਸੈੱਟ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    ਸਵੈਚਾਲਿਤ V = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("V")
                               .ਸੈੱਟ_dim({B, H, T, HS})
                               .ਸੈੱਟ_uid(V_UID)
                               .ਸੈੱਟ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    ਸਵੈਚਾਲਿਤ attn_scale = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("attn_scale")
                               .ਸੈੱਟ_dim({1, 1, 1, 1})
                               .ਸੈੱਟ_stride({1, 1, 1, 1})
                               .ਸੈੱਟ_uid(Attn_scale_UID)
                               .ਸੈੱਟ_is_pass_by_ਮੁੱਲ(ਸੱਚ)
                               .ਸੈੱਟ_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT));

    ਸਵੈਚਾਲਿਤ sdpa_options = fe::ਗ੍ਰਾਫ::SDPA_ਗੁਣ().ਸੈੱਟ_name("flash_attention");
    sdpa_options.ਸੈੱਟ_is_inference(is_inference_only);
    sdpa_options.ਸੈੱਟ_attn_scale(attn_scale);
    sdpa_options.ਸੈੱਟ_causal_ਨਕਾਬ(ਸੱਚ);

    // Create the graph operation and get the output tensors back
    ਸਵੈਚਾਲਿਤ [O, stats] = ਗ੍ਰਾਫ->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->ਸੈੱਟ_output(ਸੱਚ).ਸੈੱਟ_dim({B, H, T, HS}).ਸੈੱਟ_stride({H * HS * T, HS, H * HS, 1}).ਸੈੱਟ_uid(O_UID);

    ਪ੍ਰਤਿਜ੍ਞਾ(stats == ਖਾਲੀ_ਸੂਚਕ || is_inference_only == ਝੂਠ);
    ਜੇ (is_inference_only == ਝੂਠ) {
        stats->ਸੈੱਟ_output(ਸੱਚ).ਸੈੱਟ_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT)
                               .ਸੈੱਟ_dim({B, H, T, 1})
                               .ਸੈੱਟ_stride({H * T, T, 1, 1})
                               .ਸੈੱਟ_uid(Stats_UID);
    }

    checkCudnnFE(ਗ੍ਰਾਫ->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(ਗ੍ਰਾਫ->build_operation_ਗ੍ਰਾਫ(cudnn_handle));
    ਸਵੈਚਾਲਿਤ plans = ਗ੍ਰਾਫ->create_execution_plans({fe::Heur_ਬਹੁਲਕ_t::A});
    checkCudnnFE(ਗ੍ਰਾਫ->check_support(cudnn_handle));
    checkCudnnFE(ਗ੍ਰਾਫ->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    ਜੇ (ਗ੍ਰਾਫ->get_workspace_ਮਾਪ() > cudnn_workspace_ਮਾਪ) {
        ਜੇ (cudnn_workspace_ਮਾਪ > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_ਮਾਪ = ਗ੍ਰਾਫ->get_workspace_ਮਾਪ();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_ਮਾਪ));
    }

    user_maintained_cache_fwd.insert({ਕੁੰਜੀ, ਗ੍ਰਾਫ});

    ਵਾਪਸ ਗ੍ਰਾਫ;
}

ਸਵੈਚਾਲਿਤ lookup_cache_or_build_ਗ੍ਰਾਫ_bwd(ਪੂਰਨ B, ਪੂਰਨ NH, ਪੂਰਨ T, ਪੂਰਨ HS) {
    ਸਥਿਰ cache_type_bwd user_maintained_cache_bwd;

    ਸਵੈਚਾਲਿਤ ਕੁੰਜੀ = std::make_tuple(B, NH, T, HS);

    ਸਵੈਚਾਲਿਤ it = user_maintained_cache_bwd.find(ਕੁੰਜੀ);
    ਜੇ (it != user_maintained_cache_bwd.end()) {
        ਵਾਪਸ it->second;
    }

    ਸਵੈਚਾਲਿਤ ਗ੍ਰਾਫ = std::make_shared<fe::ਗ੍ਰਾਫ::ਗ੍ਰਾਫ_2>();
    ਗ੍ਰਾਫ->ਸੈੱਟ_io_ਅੰਕੜੇ_type(CUDNN_16_ਦੋਆਧਾਰੀ_ਅੰਕ)
          .ਸੈੱਟ_intermediate_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT)
          .ਸੈੱਟ_compute_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    ਸਵੈਚਾਲਿਤ Q = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("Q")
                            .ਸੈੱਟ_dim({B, NH, T, HS})
                            .ਸੈੱਟ_uid(Q_UID)
                            .ਸੈੱਟ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    ਸਵੈਚਾਲਿਤ K = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("K")
                            .ਸੈੱਟ_dim({B, NH, T, HS})
                            .ਸੈੱਟ_uid(K_UID)
                            .ਸੈੱਟ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    ਸਵੈਚਾਲਿਤ V = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("V")
                            .ਸੈੱਟ_dim({B, NH, T, HS})
                            .ਸੈੱਟ_uid(V_UID)
                            .ਸੈੱਟ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    ਸਵੈਚਾਲਿਤ O = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("O")
                            .ਸੈੱਟ_dim({B, NH, T, HS})
                            .ਸੈੱਟ_uid(O_UID)
                            .ਸੈੱਟ_stride({NH * HS * T, HS, NH * HS, 1}));
    ਸਵੈਚਾਲਿਤ dO = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("dO")
                            .ਸੈੱਟ_dim({B, NH, T, HS})
                            .ਸੈੱਟ_uid(dO_UID)
                            .ਸੈੱਟ_stride({NH * HS * T, HS, NH * HS, 1}));

    ਸਵੈਚਾਲਿਤ stats = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("stats")
                            .ਸੈੱਟ_dim({B, NH, T, 1})
                            .ਸੈੱਟ_uid(Stats_UID)
                            .ਸੈੱਟ_stride({NH * T, T, 1, 1})
                            .ਸੈੱਟ_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT));
    ਸਵੈਚਾਲਿਤ attn_scale = ਗ੍ਰਾਫ->tensor(fe::ਗ੍ਰਾਫ::Tensor_ਗੁਣ().ਸੈੱਟ_name("attn_scale")
                            .ਸੈੱਟ_dim({1, 1, 1, 1})
                            .ਸੈੱਟ_stride({1, 1, 1, 1})
                            .ਸੈੱਟ_is_pass_by_ਮੁੱਲ(ਸੱਚ)
                            .ਸੈੱਟ_uid(Attn_scale_UID)
                            .ਸੈੱਟ_ਅੰਕੜੇ_type(fe::ਅੰਕੜੇ_Type_t::FLOAT));
    ਸਵੈਚਾਲਿਤ sdpa_backward_options = fe::ਗ੍ਰਾਫ::SDPA_backward_ਗੁਣ().ਸੈੱਟ_name("flash_attention_backward")
#ਜੇ CUDNN_FRONTEND_MAJOR_ਸੰਸਕਰਣ > 1 || CUDNN_FRONTEND_MINOR_ਸੰਸਕਰਣ >= 5
                            .ਸੈੱਟ_deterministic_algorithm(ਸੱਚ) // 1.5+ needs this for determinism
#ਜੇ_ਅੰਤ
                            .ਸੈੱਟ_causal_ਨਕਾਬ(ਸੱਚ)
                            .ਸੈੱਟ_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    ਸਵੈਚਾਲਿਤ [dQ, dK, dV] = ਗ੍ਰਾਫ->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->ਸੈੱਟ_output(ਸੱਚ).ਸੈੱਟ_dim({B, NH, T, HS}).ਸੈੱਟ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).ਸੈੱਟ_uid(dQ_UID);
    dK->ਸੈੱਟ_output(ਸੱਚ).ਸੈੱਟ_dim({B, NH, T, HS}).ਸੈੱਟ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).ਸੈੱਟ_uid(dK_UID);
    dV->ਸੈੱਟ_output(ਸੱਚ).ਸੈੱਟ_dim({B, NH, T, HS}).ਸੈੱਟ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).ਸੈੱਟ_uid(dV_UID);

    checkCudnnFE(ਗ੍ਰਾਫ->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(ਗ੍ਰਾਫ->build_operation_ਗ੍ਰਾਫ(cudnn_handle));
    ਸਵੈਚਾਲਿਤ plans = ਗ੍ਰਾਫ->create_execution_plans({fe::Heur_ਬਹੁਲਕ_t::A});
    checkCudnnFE(ਗ੍ਰਾਫ->check_support(cudnn_handle));
    checkCudnnFE(ਗ੍ਰਾਫ->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    ਜੇ (ਗ੍ਰਾਫ->get_workspace_ਮਾਪ() > cudnn_workspace_ਮਾਪ) {
        ਜੇ (cudnn_workspace_ਮਾਪ > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_ਮਾਪ = ਗ੍ਰਾਫ->get_workspace_ਮਾਪ();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_ਮਾਪ));
    }

    user_maintained_cache_bwd.insert({ਕੁੰਜੀ, ਗ੍ਰਾਫ});
    ਵਾਪਸ ਗ੍ਰਾਫ;
}

ਖਾਲੀ attention_forward_cudnn(floatX* ਬਾਹਰਲਾ,  // output: (B, T, NH, HS)
                             ਦਸ਼ਮਲਵ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ NH, ਪੂਰਨ C, cuda_ਧਾਰਾ_t ਧਾਰਾ) {
    NVTX_RANGE_FN();
    ਪੂਰਨ HS = C / NH; // number of features per head
    ਤਾਰਕਿਕ is_inference_only = (stats == ਖਾਲੀ_ਸੂਚਕ);

    cuDNNCheck(cudnn_ਸੈੱਟ_ਧਾਰਾ(cudnn_handle, ਧਾਰਾ));

    // Get graph and tensors from cache (or generate it on first use)
    ਸਵੈਚਾਲਿਤ ਗ੍ਰਾਫ = lookup_cache_or_build_ਗ੍ਰਾਫ_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    ਖਾਲੀ* devPtrQ = inp;
    ਖਾਲੀ* devPtrK = (inp + C);
    ਖਾਲੀ* devPtrV = (inp + 2 * C);
    ਦਸ਼ਮਲਵ attn_scale_cpu = 1.0 / sqrtf(HS);
    ਖਾਲੀ* devPtrO = ਬਾਹਰਲਾ;

    // Build variant pack
    std::unordered_map<int64_t , ਖਾਲੀ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    ਜੇ (is_inference_only == ਝੂਠ) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(ਗ੍ਰਾਫ->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_ਆਖਰੀ_Error());
}

ਖਾਲੀ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ਦਸ਼ਮਲਵ* stats, // inputs
                              ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ NH, ਪੂਰਨ C, cuda_ਧਾਰਾ_t ਧਾਰਾ) {
    NVTX_RANGE_FN();
    ਪੂਰਨ HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    ਸਵੈਚਾਲਿਤ ਗ੍ਰਾਫ = lookup_cache_or_build_ਗ੍ਰਾਫ_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    ਖਾਲੀ* devPtrQ = qkvr;
    ਖਾਲੀ* devPtrK = (qkvr + NH * HS);
    ਖਾਲੀ* devPtrV = (qkvr + 2 * NH * HS);
    ਖਾਲੀ* devPtrO = o;
    ਖਾਲੀ* devPtrdO = dout;
    ਖਾਲੀ* devPtrStats = stats;
    ਦਸ਼ਮਲਵ attn_scale_cpu = 1.0 / sqrtf(HS);

    ਖਾਲੀ* devPtrdQ = dqkvr;
    ਖਾਲੀ* devPtrdK = (dqkvr + NH * HS);
    ਖਾਲੀ* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, ਖਾਲੀ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnn_ਸੈੱਟ_ਧਾਰਾ(cudnn_handle, ਧਾਰਾ));
    checkCudnnFE(ਗ੍ਰਾਫ->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_ਆਖਰੀ_Error());
}

ਖਾਲੀ create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

ਖਾਲੀ destroy_cudnn() {
    ਜੇ (cudnn_workspace != ਨਿਰਾ) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}