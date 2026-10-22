/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#அந்தர்பாவ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
வெறுமை create_cudnn();
வெறுமை destroy_cudnn();
வெறுமை attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             தசம* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             முழு B, முழு T, முழு NH, முழு C, cudaStream_t stream);

வெறுமை attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, தசம* stats, // inputs
                              முழு B, முழு T, முழு NH, முழு C, cudaStream_t stream);

#endif // CUDNN_ATT_H