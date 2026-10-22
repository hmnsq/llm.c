/*
cuDNN (flash) attention
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ CUDNN_ATT_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDNN_ATT_H

#ਅਨ੍ਤਰ੍ਭਾਵ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
ਖਾਲੀ create_cudnn();
ਖਾਲੀ destroy_cudnn();
ਖਾਲੀ attention_forward_cudnn(floatX* ਬਾਹਰਲਾ,  // output: (B, T, NH, HS)
                             ਦਸ਼ਮਲਵ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ NH, ਪੂਰਨ C, cuda_ਧਾਰਾ_t ਧਾਰਾ);

ਖਾਲੀ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ਦਸ਼ਮਲਵ* stats, // inputs
                              ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ NH, ਪੂਰਨ C, cuda_ਧਾਰਾ_t ਧਾਰਾ);

#ਜੇ_ਅੰਤ // CUDNN_ATT_H