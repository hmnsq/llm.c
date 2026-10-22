/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#অন্তর্ভাব "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
শূন্য create_cudnn();
শূন্য destroy_cudnn();
শূন্য attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             ভিন্ন* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক NH, পূর্ণাঙ্ক C, cudaStream_t stream);

শূন্য attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ভিন্ন* stats, // inputs
                              পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক NH, পূর্ণাঙ্ক C, cudaStream_t stream);

#endif // CUDNN_ATT_H