/*
cuDNN (flash) attention
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ CUDNN_ATT_H
#ವ್ಯಾಖ್ಯಾನಿಸು CUDNN_ATT_H

#ಅನ್ತರ್ಭಾವ "cuda_common.h"

// forward declarations of functions defined in cudnn_att.cpp
ಶೂನ್ಯ create_cudnn();
ಶೂನ್ಯ destroy_cudnn();
ಶೂನ್ಯ attention_forward_cudnn(floatX* ಹೊರಹೋಗು,  // output: (B, T, NH, HS)
                             ದಶಮಾಂಶ* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ NH, ಪೂರ್ಣಾಂಕ C, cuda_ಹರಿವು_t ಹರಿವು);

ಶೂನ್ಯ attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, ದಶಮಾಂಶ* stats, // inputs
                              ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ NH, ಪೂರ್ಣಾಂಕ C, cuda_ಹರಿವು_t ಹರಿವು);

#ಆದರೆ_ಅಂತ್ಯ // CUDNN_ATT_H