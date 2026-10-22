// all cudnn-related functions are in this file, so that they don't need to be recompiled everytime
// we change some unrelated piece of the code.
// TODO this currently duplicates some of the utilities from the main file

#define NOMINMAX
#અન્તર્ભાવ <unistd.h>
#અન્તર્ભાવ "cudnn_att.h"
#અન્તર્ભાવ <cudnn_frontend.h>

નામાકાશ fe = cudnn_frontend;

// Specific configurations based on the enabled precision
#જો defined(ENABLE_FP32)
static_assert(ખોટું, "cuDNN is not supported in FP32 mode.")
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#અથવાજો defined(ENABLE_FP16)
#define CUDNN_16BIT fe::DataType_t::HALF
#નહીંતર // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

સ્થિર cudnnHandle_t cudnn_handle;
સ્થિર size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
સ્થિર શૂન્ય* cudnn_workspace = ખાલી;

સ્થિર શૂન્ય cuDNNCheck(cudnnStatus_t error, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, લીટી, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

સ્થિર શૂન્ય checkCudnnFE(અચલ fe::error_object& e, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, લીટી, e.err_સંદેશ.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

ગણના UIDs {
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
using cache_type_fwd = std::map<std::tuple<પૂર્ણાંક,પૂર્ણાંક,પૂર્ણાંક,પૂર્ણાંક, પૂર્ણાંક>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<પૂર્ણાંક,પૂર્ણાંક,પૂર્ણાંક,પૂર્ણાંક>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(પૂર્ણાંક B,પૂર્ણાંક H,પૂર્ણાંક T,પૂર્ણાંક HS, પૂર્ણાંક is_inference_only) {

    સ્થિર cache_type_fwd user_maintained_cache_fwd;

    auto ચાવી = std::make_tuple(B, H, T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(ચાવી);
    જો (it != user_maintained_cache_fwd.end()) {
        પાછા it->second;
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
                               .set_is_pass_by_કિંમત(સાચું)
                               .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(સાચું);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(સાચું).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);

    પ્રતિજ્ઞા(stats == nullptr || is_inference_only == ખોટું);
    જો (is_inference_only == ખોટું) {
        stats->set_output(સાચું).set_data_type(fe::DataType_t::FLOAT)
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
    જો (graph->get_workspace_size() > cudnn_workspace_size) {
        જો (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_fwd.insert({ચાવી, graph});

    પાછા graph;
}

auto lookup_cache_or_build_graph_bwd(પૂર્ણાંક B, પૂર્ણાંક NH, પૂર્ણાંક T, પૂર્ણાંક HS) {
    સ્થિર cache_type_bwd user_maintained_cache_bwd;

    auto ચાવી = std::make_tuple(B, NH, T, HS);

    auto it = user_maintained_cache_bwd.find(ચાવી);
    જો (it != user_maintained_cache_bwd.end()) {
        પાછા it->second;
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
                            .set_is_pass_by_કિંમત(સાચું)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#જો CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(સાચું) // 1.5+ needs this for determinism
#endif
                            .set_causal_mask(સાચું)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(સાચું).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(સાચું).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(સાચું).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    જો (graph->get_workspace_size() > cudnn_workspace_size) {
        જો (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_bwd.insert({ચાવી, graph});
    પાછા graph;
}

શૂન્ય attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             દશાંશ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક NH, પૂર્ણાંક C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    પૂર્ણાંક HS = C / NH; // number of features per head
    તાર્કિક is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    શૂન્ય* devPtrQ = inp;
    શૂન્ય* devPtrK = (inp + C);
    શૂન્ય* devPtrV = (inp + 2 * C);
    દશાંશ attn_scale_cpu = 1.0 / sqrtf(HS);
    શૂન્ય* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , શૂન્ય*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    જો (is_inference_only == ખોટું) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

શૂન્ય attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, દશાંશ* stats, // inputs
                              પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક NH, પૂર્ણાંક C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    પૂર્ણાંક HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    શૂન્ય* devPtrQ = qkvr;
    શૂન્ય* devPtrK = (qkvr + NH * HS);
    શૂન્ય* devPtrV = (qkvr + 2 * NH * HS);
    શૂન્ય* devPtrO = o;
    શૂન્ય* devPtrdO = dout;
    શૂન્ય* devPtrStats = stats;
    દશાંશ attn_scale_cpu = 1.0 / sqrtf(HS);

    શૂન્ય* devPtrdQ = dqkvr;
    શૂન્ય* devPtrdK = (dqkvr + NH * HS);
    શૂન્ય* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, શૂન્ય*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

શૂન્ય create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

શૂન્ય destroy_cudnn() {
    જો (cudnn_workspace != ખાલી) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}