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
                             দশমিক* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা NH, পূর্ণসংখ্যা C, cudaStream_t stream);

শূন্য attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, দশমিক* stats, // inputs
                              পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা NH, পূর্ণসংখ্যা C, cudaStream_t stream);

#endif // CUDNN_ATT_H