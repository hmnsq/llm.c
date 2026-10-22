/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#شامل "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
تهی create_cudnn();
تهی destroy_cudnn();
تهی attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             اعشاری* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحیح B, صحیح T, صحیح NH, صحیح C, cudaStream_t stream);

تهی attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, اعشاری* stats, // inputs
                              صحیح B, صحیح T, صحیح NH, صحیح C, cudaStream_t stream);

#endif // CUDNN_ATT_H