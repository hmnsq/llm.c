/*
cuDNN (flash) attention
*/
#నిర్వచితం_కాకపోతే CUDNN_ATT_H
#నిర్వచించు CUDNN_ATT_H

#అన్తర్భావ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
శూన్య create_cudnn();
శూన్య destroy_cudnn();
శూన్య attention_forward_cudnn(floatX* బహిర్గతం,  // output: (B, T, NH, HS)
                             దశాంశం* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం NH, పూర్ణాంకం C, cuda_ప్రవాహం_t ప్రవాహం);

శూన్య attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, దశాంశం* stats, // inputs
                              పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం NH, పూర్ణాంకం C, cuda_ప్రవాహం_t ప్రవాహం);

#అయితే_ముగింపు // CUDNN_ATT_H