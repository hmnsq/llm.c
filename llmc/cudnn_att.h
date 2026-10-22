/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#അന്തര്ഭാവ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
ശൂന്യം create_cudnn();
ശൂന്യം destroy_cudnn();
ശൂന്യം attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             ദശാംശം* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ NH, പൂർണ്ണസംഖ്യ C, cudaStream_t stream);

ശൂന്യം attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ദശാംശം* stats, // inputs
                              പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ NH, പൂർണ്ണസംഖ്യ C, cudaStream_t stream);

#endif // CUDNN_ATT_H