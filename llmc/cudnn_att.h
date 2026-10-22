/*
cuDNN (flash) attention
*/
#اگر_غیر_تعریف_شدہ CUDNN_ATT_H
#تعریف_کرو CUDNN_ATT_H

#شامل "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
خالی create_cudnn();
خالی destroy_cudnn();
خالی attention_forward_cudnn(floatX* برآمد,  // output: (B, T, NH, HS)
                             اعشاری* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             صحیح B, صحیح T, صحیح NH, صحیح C, cuda_دھارا_t دھارا);

خالی attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, اعشاری* stats, // inputs
                              صحیح B, صحیح T, صحیح NH, صحیح C, cuda_دھارا_t دھارا);

#اگر_ختم // CUDNN_ATT_H