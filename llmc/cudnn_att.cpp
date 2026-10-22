// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#నిర్వచించు NOMINMAX
#అన్తర్భావ <unistd.h>
#అన్తర్భావ "cudnn_att.h"
#అన్తర్భావ <cudnn_frontend.h>

నామాకాశ fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#ఒకవేళ defined(ENABLE_FP32)
static_assert(అబద్ధం, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#లేదావేళ defined(ENABLE_FP16)
#నిర్వచించు CUDNN_16_బిట్టు fe::దత్తాంశం_Type_t::HALF
#లేకపోతే // Default to bfloat16
#నిర్వచించు CUDNN_16_బిట్టు fe::దత్తాంశం_Type_t::BFLOAT16
#అయితే_ముగింపు

స్థిర cudnnHandle_t cudnn_handle;
స్థిర size_t cudnn_workspace_కొలత = 0; // dynamically allocated as needed (up to 256MiB!)
స్థిర శూన్య* cudnn_workspace = ఖాళీ;

స్థిర శూన్య cuDNNCheck(cudnn_స్థితి_t error, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (error != CUDNN_స్థితి_SUCCESS) {
        రాయి("[CUDNN ERROR] at file %s:%d:\n%s\n", file, పంక్తి, cudnnGetErrorString(error));
        నిష్క్రమించు(EXIT_FAILURE);
    }
};
#నిర్వచించు cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

స్థిర శూన్య checkCudnnFE(స్థిరరాశి fe::error_object& e, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ(!e.is_good()) {
        రాయి("[CUDNN ERROR] at file %s:%d:\n%s\n", file, పంక్తి, e.err_సందేశం.c_str());
        నిష్క్రమించు(EXIT_FAILURE);
    }
}
#నిర్వచించు checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

గణన UIDs {
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
ఉపయోగించి cache_type_fwd = std::map<std::tuple<పూర్ణాంకం,పూర్ణాంకం,పూర్ణాంకం,పూర్ణాంకం, పూర్ణాంకం>, std::shared_ptr<fe::లేఖాచిత్రం::లేఖాచిత్రం_2>>;
ఉపయోగించి cache_type_bwd = std::map<std::tuple<పూర్ణాంకం,పూర్ణాంకం,పూర్ణాంకం,పూర్ణాంకం>, std::shared_ptr<fe::లేఖాచిత్రం::లేఖాచిత్రం_2>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
స్వయంచాలక lookup_cache_or_build_లేఖాచిత్రం_fwd(పూర్ణాంకం B,పూర్ణాంకం H,పూర్ణాంకం T,పూర్ణాంకం HS, పూర్ణాంకం is_inference_only) {

    స్థిర cache_type_fwd user_maintained_cache_fwd;

    స్వయంచాలక కీలకం = std::make_tuple(B, H, T, HS, is_inference_only);

    స్వయంచాలక it = user_maintained_cache_fwd.find(కీలకం);
    ఒకవేళ (it != user_maintained_cache_fwd.end()) {
        తిరిగి it->second;
    }

    స్వయంచాలక లేఖాచిత్రం = std::make_shared<fe::లేఖాచిత్రం::లేఖాచిత్రం_2>();
    లేఖాచిత్రం->సమితి_io_దత్తాంశం_type(CUDNN_16_బిట్టు)
          .సమితి_intermediate_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT)
          .సమితి_compute_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    స్వయంచాలక Q = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("Q")
                               .సమితి_dim({B, H, T, HS})
                               .సమితి_uid(Q_UID)
                               .సమితి_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    స్వయంచాలక K = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("K")
                               .సమితి_dim({B, H, T, HS})
                               .సమితి_uid(K_UID)
                               .సమితి_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    స్వయంచాలక V = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("V")
                               .సమితి_dim({B, H, T, HS})
                               .సమితి_uid(V_UID)
                               .సమితి_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    స్వయంచాలక attn_scale = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("attn_scale")
                               .సమితి_dim({1, 1, 1, 1})
                               .సమితి_stride({1, 1, 1, 1})
                               .సమితి_uid(Attn_scale_UID)
                               .సమితి_is_pass_by_విలువ(నిజం)
                               .సమితి_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT));

    స్వయంచాలక sdpa_options = fe::లేఖాచిత్రం::SDPA_లక్షణాలు().సమితి_name("flash_attention");
    sdpa_options.సమితి_is_inference(is_inference_only);
    sdpa_options.సమితి_attn_scale(attn_scale);
    sdpa_options.సమితి_causal_ముసుగు(నిజం);

    // Create the graph operation and get the output tensors back
    స్వయంచాలక [O, stats] = లేఖాచిత్రం->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->సమితి_output(నిజం).సమితి_dim({B, H, T, HS}).సమితి_stride({H * HS * T, HS, H * HS, 1}).సమితి_uid(O_UID);

    ప్రతిజ్ఞా(stats == శూన్యసూచిక || is_inference_only == అబద్ధం);
    ఒకవేళ (is_inference_only == అబద్ధం) {
        stats->సమితి_output(నిజం).సమితి_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT)
                               .సమితి_dim({B, H, T, 1})
                               .సమితి_stride({H * T, T, 1, 1})
                               .సమితి_uid(Stats_UID);
    }

    checkCudnnFE(లేఖాచిత్రం->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(లేఖాచిత్రం->build_operation_లేఖాచిత్రం(cudnn_handle));
    స్వయంచాలక plans = లేఖాచిత్రం->create_execution_plans({fe::Heur_బహుళకం_t::A});
    checkCudnnFE(లేఖాచిత్రం->check_support(cudnn_handle));
    checkCudnnFE(లేఖాచిత్రం->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    ఒకవేళ (లేఖాచిత్రం->get_workspace_కొలత() > cudnn_workspace_కొలత) {
        ఒకవేళ (cudnn_workspace_కొలత > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_కొలత = లేఖాచిత్రం->get_workspace_కొలత();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_కొలత));
    }

    user_maintained_cache_fwd.insert({కీలకం, లేఖాచిత్రం});

    తిరిగి లేఖాచిత్రం;
}

స్వయంచాలక lookup_cache_or_build_లేఖాచిత్రం_bwd(పూర్ణాంకం B, పూర్ణాంకం NH, పూర్ణాంకం T, పూర్ణాంకం HS) {
    స్థిర cache_type_bwd user_maintained_cache_bwd;

    స్వయంచాలక కీలకం = std::make_tuple(B, NH, T, HS);

    స్వయంచాలక it = user_maintained_cache_bwd.find(కీలకం);
    ఒకవేళ (it != user_maintained_cache_bwd.end()) {
        తిరిగి it->second;
    }

    స్వయంచాలక లేఖాచిత్రం = std::make_shared<fe::లేఖాచిత్రం::లేఖాచిత్రం_2>();
    లేఖాచిత్రం->సమితి_io_దత్తాంశం_type(CUDNN_16_బిట్టు)
          .సమితి_intermediate_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT)
          .సమితి_compute_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    స్వయంచాలక Q = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("Q")
                            .సమితి_dim({B, NH, T, HS})
                            .సమితి_uid(Q_UID)
                            .సమితి_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    స్వయంచాలక K = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("K")
                            .సమితి_dim({B, NH, T, HS})
                            .సమితి_uid(K_UID)
                            .సమితి_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    స్వయంచాలక V = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("V")
                            .సమితి_dim({B, NH, T, HS})
                            .సమితి_uid(V_UID)
                            .సమితి_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    స్వయంచాలక O = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("O")
                            .సమితి_dim({B, NH, T, HS})
                            .సమితి_uid(O_UID)
                            .సమితి_stride({NH * HS * T, HS, NH * HS, 1}));
    స్వయంచాలక dO = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("dO")
                            .సమితి_dim({B, NH, T, HS})
                            .సమితి_uid(dO_UID)
                            .సమితి_stride({NH * HS * T, HS, NH * HS, 1}));

    స్వయంచాలక stats = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("stats")
                            .సమితి_dim({B, NH, T, 1})
                            .సమితి_uid(Stats_UID)
                            .సమితి_stride({NH * T, T, 1, 1})
                            .సమితి_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT));
    స్వయంచాలక attn_scale = లేఖాచిత్రం->tensor(fe::లేఖాచిత్రం::Tensor_లక్షణాలు().సమితి_name("attn_scale")
                            .సమితి_dim({1, 1, 1, 1})
                            .సమితి_stride({1, 1, 1, 1})
                            .సమితి_is_pass_by_విలువ(నిజం)
                            .సమితి_uid(Attn_scale_UID)
                            .సమితి_దత్తాంశం_type(fe::దత్తాంశం_Type_t::FLOAT));
    స్వయంచాలక sdpa_backward_options = fe::లేఖాచిత్రం::SDPA_backward_లక్షణాలు().సమితి_name("flash_attention_backward")
#ఒకవేళ CUDNN_FRONTEND_MAJOR_సంస్కరణ > 1 || CUDNN_FRONTEND_MINOR_సంస్కరణ >= 5
                            .సమితి_deterministic_algorithm(నిజం) // 1.5+ needs this for determinism
#అయితే_ముగింపు
                            .సమితి_causal_ముసుగు(నిజం)
                            .సమితి_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    స్వయంచాలక [dQ, dK, dV] = లేఖాచిత్రం->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->సమితి_output(నిజం).సమితి_dim({B, NH, T, HS}).సమితి_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).సమితి_uid(dQ_UID);
    dK->సమితి_output(నిజం).సమితి_dim({B, NH, T, HS}).సమితి_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).సమితి_uid(dK_UID);
    dV->సమితి_output(నిజం).సమితి_dim({B, NH, T, HS}).సమితి_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).సమితి_uid(dV_UID);

    checkCudnnFE(లేఖాచిత్రం->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(లేఖాచిత్రం->build_operation_లేఖాచిత్రం(cudnn_handle));
    స్వయంచాలక plans = లేఖాచిత్రం->create_execution_plans({fe::Heur_బహుళకం_t::A});
    checkCudnnFE(లేఖాచిత్రం->check_support(cudnn_handle));
    checkCudnnFE(లేఖాచిత్రం->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    ఒకవేళ (లేఖాచిత్రం->get_workspace_కొలత() > cudnn_workspace_కొలత) {
        ఒకవేళ (cudnn_workspace_కొలత > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_కొలత = లేఖాచిత్రం->get_workspace_కొలత();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_కొలత));
    }

    user_maintained_cache_bwd.insert({కీలకం, లేఖాచిత్రం});
    తిరిగి లేఖాచిత్రం;
}

శూన్య attention_forward_cudnn(floatX* బహిర్గతం,  // output: (B, T, NH, HS)
                             దశాంశం* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం NH, పూర్ణాంకం C, cuda_ప్రవాహం_t ప్రవాహం) {
    NVTX_RANGE_FN();
    పూర్ణాంకం HS = C / NH; // number of features per head
    తార్కిక is_inference_only = (stats == శూన్యసూచిక);

    cuDNNCheck(cudnn_సమితి_ప్రవాహం(cudnn_handle, ప్రవాహం));

    // Get graph and tensors from cache (or generate it on first use)
    స్వయంచాలక లేఖాచిత్రం = lookup_cache_or_build_లేఖాచిత్రం_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    శూన్య* devPtrQ = inp;
    శూన్య* devPtrK = (inp + C);
    శూన్య* devPtrV = (inp + 2 * C);
    దశాంశం attn_scale_cpu = 1.0 / sqrtf(HS);
    శూన్య* devPtrO = బహిర్గతం;

    // Build variant pack
    std::unordered_map<int64_t , శూన్య*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    ఒకవేళ (is_inference_only == అబద్ధం) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(లేఖాచిత్రం->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_చివరి_Error());
}

శూన్య attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, దశాంశం* stats, // inputs
                              పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం NH, పూర్ణాంకం C, cuda_ప్రవాహం_t ప్రవాహం) {
    NVTX_RANGE_FN();
    పూర్ణాంకం HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    స్వయంచాలక లేఖాచిత్రం = lookup_cache_or_build_లేఖాచిత్రం_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    శూన్య* devPtrQ = qkvr;
    శూన్య* devPtrK = (qkvr + NH * HS);
    శూన్య* devPtrV = (qkvr + 2 * NH * HS);
    శూన్య* devPtrO = o;
    శూన్య* devPtrdO = dout;
    శూన్య* devPtrStats = stats;
    దశాంశం attn_scale_cpu = 1.0 / sqrtf(HS);

    శూన్య* devPtrdQ = dqkvr;
    శూన్య* devPtrdK = (dqkvr + NH * HS);
    శూన్య* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, శూన్య*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnn_సమితి_ప్రవాహం(cudnn_handle, ప్రవాహం));
    checkCudnnFE(లేఖాచిత్రం->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cuda_Get_చివరి_Error());
}

శూన్య create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

శూన్య destroy_cudnn() {
    ఒకవేళ (cudnn_workspace != ఖాళీ) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}