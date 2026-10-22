// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#define NOMINMAX
#अन्तर्भाव <unistd.h>
#अन्तर्भाव "cudnn_att.h"
#अन्तर्भाव <cudnn_frontend.h>

नामाकाश fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#यदि defined(ENABLE_FP32)
static_assert(असत्य, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#नोचेद्यदि defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#अन्यथा // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

स्थिर cudnnHandle_t cudnn_handle;
स्थिर size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
स्थिर शून्य* cudnn_workspace = रिक्त;

स्थिर शून्य cuDNNCheck(cudnnStatus_t error, ध्रुव वर्ण *file, पूर्णाङ्क ओळ) {
    यदि (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, ओळ, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

स्थिर शून्य checkCudnnFE(ध्रुव fe::error_object& e, ध्रुव वर्ण *file, पूर्णाङ्क ओळ) {
    यदि(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, ओळ, e.err_संदेश.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

गणना UIDs {
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
using cache_type_fwd = std::map<std::tuple<पूर्णाङ्क,पूर्णाङ्क,पूर्णाङ्क,पूर्णाङ्क, पूर्णाङ्क>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<पूर्णाङ्क,पूर्णाङ्क,पूर्णाङ्क,पूर्णाङ्क>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(पूर्णाङ्क B,पूर्णाङ्क H,पूर्णाङ्क T,पूर्णाङ्क HS, पूर्णाङ्क is_inference_only) {

    स्थिर cache_type_fwd user_maintained_cache_fwd;

    auto किल्ली = std::make_tuple(B, H, T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(किल्ली);
    यदि (it != user_maintained_cache_fwd.end()) {
        प्रत्यागम it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_uid(K_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_uid(V_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_मूल्य(सत्य)
                               .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(सत्य);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(सत्य).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);

    प्रतिज्ञा(stats == nullptr || is_inference_only == असत्य);
    यदि (is_inference_only == असत्य) {
        stats->set_output(सत्य).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    यदि (graph->get_workspace_size() > cudnn_workspace_size) {
        यदि (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_fwd.insert({किल्ली, graph});

    प्रत्यागम graph;
}

auto lookup_cache_or_build_graph_bwd(पूर्णाङ्क B, पूर्णाङ्क NH, पूर्णाङ्क T, पूर्णाङ्क HS) {
    स्थिर cache_type_bwd user_maintained_cache_bwd;

    auto किल्ली = std::make_tuple(B, NH, T, HS);

    auto it = user_maintained_cache_bwd.find(किल्ली);
    यदि (it != user_maintained_cache_bwd.end()) {
        प्रत्यागम it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, NH, T, HS})
                            .set_uid(O_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, NH, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, NH, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_मूल्य(सत्य)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#यदि CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(सत्य) // 1.5+ needs this for determinism
#endif
                            .set_causal_mask(सत्य)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(सत्य).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(सत्य).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(सत्य).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    यदि (graph->get_workspace_size() > cudnn_workspace_size) {
        यदि (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_bwd.insert({किल्ली, graph});
    प्रत्यागम graph;
}

शून्य attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             भिन्न* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क NH, पूर्णाङ्क C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    पूर्णाङ्क HS = C / NH; // number of features per head
    तार्किक is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    शून्य* devPtrQ = inp;
    शून्य* devPtrK = (inp + C);
    शून्य* devPtrV = (inp + 2 * C);
    भिन्न attn_scale_cpu = 1.0 / sqrtf(HS);
    शून्य* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , शून्य*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    यदि (is_inference_only == असत्य) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

शून्य attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, भिन्न* stats, // inputs
                              पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क NH, पूर्णाङ्क C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    पूर्णाङ्क HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    शून्य* devPtrQ = qkvr;
    शून्य* devPtrK = (qkvr + NH * HS);
    शून्य* devPtrV = (qkvr + 2 * NH * HS);
    शून्य* devPtrO = o;
    शून्य* devPtrdO = dout;
    शून्य* devPtrStats = stats;
    भिन्न attn_scale_cpu = 1.0 / sqrtf(HS);

    शून्य* devPtrdQ = dqkvr;
    शून्य* devPtrdK = (dqkvr + NH * HS);
    शून्य* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, शून्य*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

शून्य create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

शून्य destroy_cudnn() {
    यदि (cudnn_workspace != रिक्त) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}