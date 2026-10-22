/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#ಅನ್ತರ್ಭಾವ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
ಶೂನ್ಯ create_cudnn();
ಶೂನ್ಯ destroy_cudnn();
ಶೂನ್ಯ attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             ಭಿನ್ನ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ NH, ಪೂರ್ಣಾಙ್ಕ C, cudaStream_t stream);

ಶೂನ್ಯ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ಭಿನ್ನ* stats, // inputs
                              ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ NH, ಪೂರ್ಣಾಙ್ಕ C, cudaStream_t stream);

#endif // CUDNN_ATT_H