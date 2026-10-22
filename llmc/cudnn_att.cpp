// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#ವ್ಯಾಖ್ಯಾನಿಸು NOMINMAX
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ "cudnn_att.h"
#ಅನ್ತರ್ಭಾವ <cudnn_frontend.h>

ನಾಮಾಕಾಶ fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#ಆದರೆ defined(ENABLE_FP32)
static_assert(ಸುಳ್ಳು, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#ಅಥವಾಆದರೆ defined(ENABLE_FP16)
#ವ್ಯಾಖ್ಯಾನಿಸು CUDNN_16_ಬಿಟ್ fe::ದತ್ತಾಂಶ_Type_t::HALF
#ಇಲ್ಲದಿದ್ದರೆ // Default to bfloat16
#ವ್ಯಾಖ್ಯಾನಿಸು CUDNN_16_ಬಿಟ್ fe::ದತ್ತಾಂಶ_Type_t::BFLOAT16
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ cudnnHandle_t cudnn_handle;
ಸ್ಥಿರ size_t cudnn_workspace_ಗಾತ್ರ = 0; // dynamically allocated as needed (up to 256MiB!)
ಸ್ಥಿರ ಶೂನ್ಯ* cudnn_workspace = ಖಾಲಿ;

ಸ್ಥಿರ ಶೂನ್ಯ cuDNNCheck(cudnn_ಸ್ಥಿತಿ_t error, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (error != CUDNN_ಸ್ಥಿತಿ_SUCCESS) {
        ಬರೆ("[CUDNN ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು, cudnnGetErrorString(error));
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
};
#ವ್ಯಾಖ್ಯಾನಿಸು cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

ಸ್ಥಿರ ಶೂನ್ಯ checkCudnnFE(ಸ್ಥಿರಾಂಕ fe::error_object& e, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ(!e.is_good()) {
        ಬರೆ("[CUDNN ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು, e.err_ಸಂದೇಶ.c_str());
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
}
#ವ್ಯಾಖ್ಯಾನಿಸು checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

ಎಣಿಕೆ UIDs {
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
ಬಳಸಿ cache_type_fwd = std::map<std::tuple<ಪೂರ್ಣಾಂಕ,ಪೂರ್ಣಾಂಕ,ಪೂರ್ಣಾಂಕ,ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ>, std::shared_ptr<fe::ನಕ್ಷೆ::ನಕ್ಷೆ_2>>;
ಬಳಸಿ cache_type_bwd = std::map<std::tuple<ಪೂರ್ಣಾಂಕ,ಪೂರ್ಣಾಂಕ,ಪೂರ್ಣಾಂಕ,ಪೂರ್ಣಾಂಕ>, std::shared_ptr<fe::ನಕ್ಷೆ::ನಕ್ಷೆ_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
ಸ್ವಯಂಚಾಲಿತ lookup_cache_or_build_ನಕ್ಷೆ_fwd(ಪೂರ್ಣಾಂಕ B,ಪೂರ್ಣಾಂಕ H,ಪೂರ್ಣಾಂಕ T,ಪೂರ್ಣಾಂಕ HS, ಪೂರ್ಣಾಂಕ is_inference_only) {

    ಸ್ಥಿರ cache_type_fwd user_maintained_cache_fwd;

    ಸ್ವಯಂಚಾಲಿತ ಕೀಲಿ = std::make_tuple(B, H, T, HS, is_inference_only);

    ಸ್ವಯಂಚಾಲಿತ it = user_maintained_cache_fwd.find(ಕೀಲಿ);
    ಆದರೆ (it != user_maintained_cache_fwd.end()) {
        ಹಿಂತಿರುಗಿ it->second;
    }

    ಸ್ವಯಂಚಾಲಿತ ನಕ್ಷೆ = std::make_shared<fe::ನಕ್ಷೆ::ನಕ್ಷೆ_2>();
    ನಕ್ಷೆ->ಗಣ_io_ದತ್ತಾಂಶ_type(CUDNN_16_ಬಿಟ್)
          .ಗಣ_intermediate_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT)
          .ಗಣ_compute_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    ಸ್ವಯಂಚಾಲಿತ Q = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("Q")
                               .ಗಣ_dim({B, H, T, HS})
                               .ಗಣ_uid(Q_UID)
                               .ಗಣ_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ K = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("K")
                               .ಗಣ_dim({B, H, T, HS})
                               .ಗಣ_uid(K_UID)
                               .ಗಣ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ V = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("V")
                               .ಗಣ_dim({B, H, T, HS})
                               .ಗಣ_uid(V_UID)
                               .ಗಣ_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ attn_scale = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("attn_scale")
                               .ಗಣ_dim({1, 1, 1, 1})
                               .ಗಣ_stride({1, 1, 1, 1})
                               .ಗಣ_uid(Attn_scale_UID)
                               .ಗಣ_is_pass_by_ಮೌಲ್ಯ(ನಿಜ)
                               .ಗಣ_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT));

    ಸ್ವಯಂಚಾಲಿತ sdpa_options = fe::ನಕ್ಷೆ::SDPA_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("flash_attention");
    sdpa_options.ಗಣ_is_inference(is_inference_only);
    sdpa_options.ಗಣ_attn_scale(attn_scale);
    sdpa_options.ಗಣ_causal_ಮುಖವಾಡ(ನಿಜ);

    // Create the graph operation and get the output tensors back
    ಸ್ವಯಂಚಾಲಿತ [O, stats] = ನಕ್ಷೆ->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->ಗಣ_output(ನಿಜ).ಗಣ_dim({B, H, T, HS}).ಗಣ_stride({H * HS * T, HS, H * HS, 1}).ಗಣ_uid(O_UID);

    ಪ್ರತಿಜ್ಞಾ(stats == ಶೂನ್ಯಸೂಚಕ || is_inference_only == ಸುಳ್ಳು);
    ಆದರೆ (is_inference_only == ಸುಳ್ಳು) {
        stats->ಗಣ_output(ನಿಜ).ಗಣ_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT)
                               .ಗಣ_dim({B, H, T, 1})
                               .ಗಣ_stride({H * T, T, 1, 1})
                               .ಗಣ_uid(Stats_UID);
    }

    checkCudnnFE(ನಕ್ಷೆ->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(ನಕ್ಷೆ->build_operation_ನಕ್ಷೆ(cudnn_handle));
    ಸ್ವಯಂಚಾಲಿತ plans = ನಕ್ಷೆ->create_execution_plans({fe::Heur_ಬಹುಲಕ_t::A});
    checkCudnnFE(ನಕ್ಷೆ->check_support(cudnn_handle));
    checkCudnnFE(ನಕ್ಷೆ->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    ಆದರೆ (ನಕ್ಷೆ->get_workspace_ಗಾತ್ರ() > cudnn_workspace_ಗಾತ್ರ) {
        ಆದರೆ (cudnn_workspace_ಗಾತ್ರ > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_ಗಾತ್ರ = ನಕ್ಷೆ->get_workspace_ಗಾತ್ರ();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_ಗಾತ್ರ));
    }

    user_maintained_cache_fwd.insert({ಕೀಲಿ, ನಕ್ಷೆ});

    ಹಿಂತಿರುಗಿ ನಕ್ಷೆ;
}

ಸ್ವಯಂಚಾಲಿತ lookup_cache_or_build_ನಕ್ಷೆ_bwd(ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ NH, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ HS) {
    ಸ್ಥಿರ cache_type_bwd user_maintained_cache_bwd;

    ಸ್ವಯಂಚಾಲಿತ ಕೀಲಿ = std::make_tuple(B, NH, T, HS);

    ಸ್ವಯಂಚಾಲಿತ it = user_maintained_cache_bwd.find(ಕೀಲಿ);
    ಆದರೆ (it != user_maintained_cache_bwd.end()) {
        ಹಿಂತಿರುಗಿ it->second;
    }

    ಸ್ವಯಂಚಾಲಿತ ನಕ್ಷೆ = std::make_shared<fe::ನಕ್ಷೆ::ನಕ್ಷೆ_2>();
    ನಕ್ಷೆ->ಗಣ_io_ದತ್ತಾಂಶ_type(CUDNN_16_ಬಿಟ್)
          .ಗಣ_intermediate_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT)
          .ಗಣ_compute_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    ಸ್ವಯಂಚಾಲಿತ Q = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("Q")
                            .ಗಣ_dim({B, NH, T, HS})
                            .ಗಣ_uid(Q_UID)
                            .ಗಣ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ K = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("K")
                            .ಗಣ_dim({B, NH, T, HS})
                            .ಗಣ_uid(K_UID)
                            .ಗಣ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ V = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("V")
                            .ಗಣ_dim({B, NH, T, HS})
                            .ಗಣ_uid(V_UID)
                            .ಗಣ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ O = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("O")
                            .ಗಣ_dim({B, NH, T, HS})
                            .ಗಣ_uid(O_UID)
                            .ಗಣ_stride({NH * HS * T, HS, NH * HS, 1}));
    ಸ್ವಯಂಚಾಲಿತ dO = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("dO")
                            .ಗಣ_dim({B, NH, T, HS})
                            .ಗಣ_uid(dO_UID)
                            .ಗಣ_stride({NH * HS * T, HS, NH * HS, 1}));

    ಸ್ವಯಂಚಾಲಿತ stats = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("stats")
                            .ಗಣ_dim({B, NH, T, 1})
                            .ಗಣ_uid(Stats_UID)
                            .ಗಣ_stride({NH * T, T, 1, 1})
                            .ಗಣ_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT));
    ಸ್ವಯಂಚಾಲಿತ attn_scale = ನಕ್ಷೆ->tensor(fe::ನಕ್ಷೆ::Tensor_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("attn_scale")
                            .ಗಣ_dim({1, 1, 1, 1})
                            .ಗಣ_stride({1, 1, 1, 1})
                            .ಗಣ_is_pass_by_ಮೌಲ್ಯ(ನಿಜ)
                            .ಗಣ_uid(Attn_scale_UID)
                            .ಗಣ_ದತ್ತಾಂಶ_type(fe::ದತ್ತಾಂಶ_Type_t::FLOAT));
    ಸ್ವಯಂಚಾಲಿತ sdpa_backward_options = fe::ನಕ್ಷೆ::SDPA_backward_ಗುಣಲಕ್ಷಣಗಳು().ಗಣ_name("flash_attention_backward")
#ಆದರೆ CUDNN_FRONTEND_MAJOR_ಆವೃತ್ತಿ > 1 || CUDNN_FRONTEND_MINOR_ಆವೃತ್ತಿ >= 5
                            .ಗಣ_deterministic_algorithm(ನಿಜ) // 1.5+ needs this for determinism
#ಆದರೆ_ಅಂತ್ಯ
                            .ಗಣ_causal_ಮುಖವಾಡ(ನಿಜ)
                            .ಗಣ_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    ಸ್ವಯಂಚಾಲಿತ [dQ, dK, dV] = ನಕ್ಷೆ->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->ಗಣ_output(ನಿಜ).ಗಣ_dim({B, NH, T, HS}).ಗಣ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).ಗಣ_uid(dQ_UID);
    dK->ಗಣ_output(ನಿಜ).ಗಣ_dim({B, NH, T, HS}).ಗಣ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).ಗಣ_uid(dK_UID);
    dV->ಗಣ_output(ನಿಜ).ಗಣ_dim({B, NH, T, HS}).ಗಣ_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).ಗಣ_uid(dV_UID);

    checkCudnnFE(ನಕ್ಷೆ->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(ನಕ್ಷೆ->build_operation_ನಕ್ಷೆ(cudnn_handle));
    ಸ್ವಯಂಚಾಲಿತ plans = ನಕ್ಷೆ->create_execution_plans({fe::Heur_ಬಹುಲಕ_t::A});
    checkCudnnFE(ನಕ್ಷೆ->check_support(cudnn_handle));
    checkCudnnFE(ನಕ್ಷೆ->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    ಆದರೆ (ನಕ್ಷೆ->get_workspace_ಗಾತ್ರ() > cudnn_workspace_ಗಾತ್ರ) {
        ಆದರೆ (cudnn_workspace_ಗಾತ್ರ > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_ಗಾತ್ರ = ನಕ್ಷೆ->get_workspace_ಗಾತ್ರ();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_ಗಾತ್ರ));
    }

    user_maintained_cache_bwd.insert({ಕೀಲಿ, ನಕ್ಷೆ});
    ಹಿಂತಿರುಗಿ ನಕ್ಷೆ;
}

ಶೂನ್ಯ attention_forward_cudnn(floatX* ಹೊರಹೋಗು,  // output: (B, T, NH, HS)
                             ದಶಮಾಂಶ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ NH, ಪೂರ್ಣಾಂಕ C, cuda_ಹರಿವು_t ಹರಿವು) {
    NVTX_RANGE_FN();
    ಪೂರ್ಣಾಂಕ HS = C / NH; // number of features per head
    ತಾರ್ಕಿಕ is_inference_only = (stats == ಶೂನ್ಯಸೂಚಕ);

    cuDNNCheck(cudnn_ಗಣ_ಹರಿವು(cudnn_handle, ಹರಿವು));

    // Get graph and tensors from cache (or generate it on first use)
    ಸ್ವಯಂಚಾಲಿತ ನಕ್ಷೆ = lookup_cache_or_build_ನಕ್ಷೆ_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    ಶೂನ್ಯ* devPtrQ = inp;
    ಶೂನ್ಯ* devPtrK = (inp + C);
    ಶೂನ್ಯ* devPtrV = (inp + 2 * C);
    ದಶಮಾಂಶ attn_scale_cpu = 1.0 / sqrtf(HS);
    ಶೂನ್ಯ* devPtrO = ಹೊರಹೋಗು;

    // Build variant pack
    std::unordered_map<int64_t , ಶೂನ್ಯ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    ಆದರೆ (is_inference_only == ಸುಳ್ಳು) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(ನಕ್ಷೆ->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_ಕೊನೆಯ_Error());
}

ಶೂನ್ಯ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ದಶಮಾಂಶ* stats, // inputs
                              ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ NH, ಪೂರ್ಣಾಂಕ C, cuda_ಹರಿವು_t ಹರಿವು) {
    NVTX_RANGE_FN();
    ಪೂರ್ಣಾಂಕ HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    ಸ್ವಯಂಚಾಲಿತ ನಕ್ಷೆ = lookup_cache_or_build_ನಕ್ಷೆ_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    ಶೂನ್ಯ* devPtrQ = qkvr;
    ಶೂನ್ಯ* devPtrK = (qkvr + NH * HS);
    ಶೂನ್ಯ* devPtrV = (qkvr + 2 * NH * HS);
    ಶೂನ್ಯ* devPtrO = o;
    ಶೂನ್ಯ* devPtrdO = dout;
    ಶೂನ್ಯ* devPtrStats = stats;
    ದಶಮಾಂಶ attn_scale_cpu = 1.0 / sqrtf(HS);

    ಶೂನ್ಯ* devPtrdQ = dqkvr;
    ಶೂನ್ಯ* devPtrdK = (dqkvr + NH * HS);
    ಶೂನ್ಯ* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, ಶೂನ್ಯ*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnn_ಗಣ_ಹರಿವು(cudnn_handle, ಹರಿವು));
    checkCudnnFE(ನಕ್ಷೆ->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_ಕೊನೆಯ_Error());
}

ಶೂನ್ಯ create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

ಶೂನ್ಯ destroy_cudnn() {
    ಆದರೆ (cudnn_workspace != ಖಾಲಿ) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}