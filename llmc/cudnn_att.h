/*
cuDNN (flash) attention
*/
#که_نه_تعریف_شوی CUDNN_ATT_H
#تعریف_کړه CUDNN_ATT_H

#ضمن "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
فراغ create_cudnn();
فراغ destroy_cudnn();
فراغ attention_forward_cudnn(floatX* بهر,  // output: (B, T, NH, HS)
                             عائم* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحيح B, صحيح T, صحيح NH, صحيح C, cuda_بهیر_t بهیر);

فراغ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, عائم* stats, // inputs
                              صحيح B, صحيح T, صحيح NH, صحيح C, cuda_بهیر_t بهیر);

#د_که_پای // CUDNN_ATT_H