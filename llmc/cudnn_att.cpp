// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#घोषणा NOMINMAX
#समावेश <unistd.h>
#समावेश "cudnn_att.h"
#समावेश <cudnn_frontend.h>

namespace fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#अगर defined(ENABLE_FP32)
static_assert(false, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#या_अगर defined(ENABLE_FP16)
#घोषणा CUDNN_16BIT fe::तथ्य_Type_t::HALF
#अन्यथा // Default to bfloat16
#घोषणा CUDNN_16BIT fe::तथ्य_Type_t::BFLOAT16
#पूर्ण_अगर

अटल cudnnHandle_t cudnn_handle;
अटल माप_प्रकार cudnn_workspace_आकार = 0; // dynamically allocated as needed (up to 256MiB!)
अटल व्योम* cudnn_workspace = शून्य;

अटल व्योम cuDNNCheck(cudnnStatus_t error, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (error != CUDNN_STATUS_SUCCESS) {
        म_लिखो("[CUDNN ERROR] at file %s:%d:\n%s\n", file, पंक्ति, cudnnGetErrorString(error));
        निकास(निकास_त्रुटि);
    }
};
#घोषणा cuDNNCheck(err) (cuDNNCheck(err, __खाता__, __LINE__))

अटल व्योम checkCudnnFE(स्थिर fe::error_object& e, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर(!e.is_good()) {
        म_लिखो("[CUDNN ERROR] at file %s:%d:\n%s\n", file, पंक्ति, e.err_संदेश.c_str());
        निकास(निकास_त्रुटि);
    }
}
#घोषणा checkCudnnFE(err) checkCudnnFE(err, __खाता__, __LINE__)

क्रमागत UIDs {
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
using cache_type_fwd = std::map<std::tuple<पूर्णांक,पूर्णांक,पूर्णांक,पूर्णांक, पूर्णांक>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<पूर्णांक,पूर्णांक,पूर्णांक,पूर्णांक>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
स्वतः lookup_cache_or_build_graph_fwd(पूर्णांक B,पूर्णांक H,पूर्णांक T,पूर्णांक HS, पूर्णांक is_inference_only) {

    अटल cache_type_fwd user_maintained_cache_fwd;

    स्वतः कुंजी = std::make_tuple(B, H, T, HS, is_inference_only);

    स्वतः it = user_maintained_cache_fwd.find(कुंजी);
    अगर (it != user_maintained_cache_fwd.end()) {
        वापस it->second;
    }

    स्वतः graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_तथ्य_type(CUDNN_16BIT)
          .set_intermediate_तथ्य_type(fe::तथ्य_Type_t::FLOAT)
          .set_compute_तथ्य_type(fe::तथ्य_Type_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    स्वतः Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    स्वतः K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_uid(K_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    स्वतः V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_uid(V_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    स्वतः attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_मूल्य(true)
                               .set_तथ्य_type(fe::तथ्य_Type_t::FLOAT));

    स्वतः sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    स्वतः [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);

    निश्चित(stats == nullptr || is_inference_only == false);
    अगर (is_inference_only == false) {
        stats->set_output(true).set_तथ्य_type(fe::तथ्य_Type_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    स्वतः plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    अगर (graph->get_workspace_आकार() > cudnn_workspace_आकार) {
        अगर (cudnn_workspace_आकार > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_आकार = graph->get_workspace_आकार();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_आकार));
    }

    user_maintained_cache_fwd.insert({कुंजी, graph});

    वापस graph;
}

स्वतः lookup_cache_or_build_graph_bwd(पूर्णांक B, पूर्णांक NH, पूर्णांक T, पूर्णांक HS) {
    अटल cache_type_bwd user_maintained_cache_bwd;

    स्वतः कुंजी = std::make_tuple(B, NH, T, HS);

    स्वतः it = user_maintained_cache_bwd.find(कुंजी);
    अगर (it != user_maintained_cache_bwd.end()) {
        वापस it->second;
    }

    स्वतः graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_तथ्य_type(CUDNN_16BIT)
          .set_intermediate_तथ्य_type(fe::तथ्य_Type_t::FLOAT)
          .set_compute_तथ्य_type(fe::तथ्य_Type_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    स्वतः Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    स्वतः K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    स्वतः V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    स्वतः O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, NH, T, HS})
                            .set_uid(O_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));
    स्वतः dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, NH, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));

    स्वतः stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, NH, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_तथ्य_type(fe::तथ्य_Type_t::FLOAT));
    स्वतः attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_मूल्य(true)
                            .set_uid(Attn_scale_UID)
                            .set_तथ्य_type(fe::तथ्य_Type_t::FLOAT));
    स्वतः sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#अगर CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(true) // 1.5+ needs this for determinism
#पूर्ण_अगर
                            .set_causal_mask(true)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    स्वतः [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    स्वतः plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    अगर (graph->get_workspace_आकार() > cudnn_workspace_आकार) {
        अगर (cudnn_workspace_आकार > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_आकार = graph->get_workspace_आकार();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_आकार));
    }

    user_maintained_cache_bwd.insert({कुंजी, graph});
    वापस graph;
}

व्योम attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             भग्न* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             पूर्णांक B, पूर्णांक T, पूर्णांक NH, पूर्णांक C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    पूर्णांक HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    स्वतः graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    व्योम* devPtrQ = inp;
    व्योम* devPtrK = (inp + C);
    व्योम* devPtrV = (inp + 2 * C);
    भग्न attn_scale_cpu = 1.0 / sqrtf(HS);
    व्योम* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , व्योम*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    अगर (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

व्योम attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, भग्न* stats, // inputs
                              पूर्णांक B, पूर्णांक T, पूर्णांक NH, पूर्णांक C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    पूर्णांक HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    स्वतः graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    व्योम* devPtrQ = qkvr;
    व्योम* devPtrK = (qkvr + NH * HS);
    व्योम* devPtrV = (qkvr + 2 * NH * HS);
    व्योम* devPtrO = o;
    व्योम* devPtrdO = dout;
    व्योम* devPtrStats = stats;
    भग्न attn_scale_cpu = 1.0 / sqrtf(HS);

    व्योम* devPtrdQ = dqkvr;
    व्योम* devPtrdK = (dqkvr + NH * HS);
    व्योम* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, व्योम*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

व्योम create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

व्योम destroy_cudnn() {
    अगर (cudnn_workspace != शून्य) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}