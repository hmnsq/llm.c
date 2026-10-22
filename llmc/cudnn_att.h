/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#ଅନ୍ତର୍ଭାଵ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
ଶୂନ୍ୟ create_cudnn();
ଶୂନ୍ୟ destroy_cudnn();
ଶୂନ୍ୟ attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             ଦଶମିକ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NH, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, cudaStream_t stream);

ଶୂନ୍ୟ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ଦଶମିକ* stats, // inputs
                              ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା NH, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, cudaStream_t stream);

#endif // CUDNN_ATT_H