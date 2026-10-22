/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#अन्तर्भाव "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
शून्य create_cudnn();
शून्य destroy_cudnn();
शून्य attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             दशांश* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             पूर्णांक B, पूर्णांक T, पूर्णांक NH, पूर्णांक C, cudaStream_t stream);

शून्य attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, दशांश* stats, // inputs
                              पूर्णांक B, पूर्णांक T, पूर्णांक NH, पूर्णांक C, cudaStream_t stream);

#endif // CUDNN_ATT_H