/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#અન્તર્ભાવ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
શૂન્ય create_cudnn();
શૂન્ય destroy_cudnn();
શૂન્ય attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             દશાંશ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક NH, પૂર્ણાંક C, cudaStream_t stream);

શૂન્ય attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, દશાંશ* stats, // inputs
                              પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક NH, પૂર્ણાંક C, cudaStream_t stream);

#endif // CUDNN_ATT_H