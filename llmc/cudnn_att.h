/*
cuDNN (flash) attention
*/
#अगर_अघोषित CUDNN_ATT_H
#घोषणा CUDNN_ATT_H

#समावेश "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
व्योम create_cudnn();
व्योम destroy_cudnn();
व्योम attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             भग्न* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             पूर्णांक B, पूर्णांक T, पूर्णांक NH, पूर्णांक C, cudaStream_t stream);

व्योम attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, भग्न* stats, // inputs
                              पूर्णांक B, पूर्णांक T, पूर्णांक NH, पूर्णांक C, cudaStream_t stream);

#पूर्ण_अगर // CUDNN_ATT_H