/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
void create_cudnn();
void destroy_cudnn();
void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             puur_naa_nka B, puur_naa_nka T, puur_naa_nka NH, puur_naa_nka C, cudaStream_t stream);

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              puur_naa_nka B, puur_naa_nka T, puur_naa_nka NH, puur_naa_nka C, cudaStream_t stream);

#endif // CUDNN_ATT_H