/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <ctype.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <time.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ OMP
#ಅನ್ತರ್ಭಾವ <omp.h>
#ಆದರೆ_ಅಂತ್ಯ
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ಅನ್ತರ್ಭಾವ "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#ಅನ್ತರ್ಭಾವ "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#ಅನ್ತರ್ಭಾವ "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

ಶೂನ್ಯ encoder_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು,
                   ಪೂರ್ಣಾಂಕ* inp, ದಶಮಾಂಶ* wte, ದಶಮಾಂಶ* wpe,
                   ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            ದಶಮಾಂಶ* ಹೊರಹೋಗು_bt = ಹೊರಹೋಗು + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            ಪೂರ್ಣಾಂಕ ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            ದಶಮಾಂಶ* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            ದಶಮಾಂಶ* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ಹೊರಹೋಗು_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

ಶೂನ್ಯ encoder_backward(ದಶಮಾಂಶ* dwte, ದಶಮಾಂಶ* dwpe,
                      ದಶಮಾಂಶ* dout, ಪೂರ್ಣಾಂಕ* inp,
                      ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ದಶಮಾಂಶ* dout_bt = dout + b * T * C + t * C;
            ಪೂರ್ಣಾಂಕ ix = inp[b * T + t];
            ದಶಮಾಂಶ* dwte_ix = dwte + ix * C;
            ದಶಮಾಂಶ* dwpe_t = dwpe + t * C;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

ಶೂನ್ಯ layernorm_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು, ದಶಮಾಂಶ* ಸರಾಸರಿ, ದಶಮಾಂಶ* rstd,
                       ದಶಮಾಂಶ* inp, ದಶಮಾಂಶ* ತೂಕ, ದಶಮಾಂಶ* bias,
                       ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    ದಶಮಾಂಶ eps = 1e-5f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ದಶಮಾಂಶ* x = inp + b * T * C + t * C;
            // calculate the mean
            ದಶಮಾಂಶ m = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ದಶಮಾಂಶ v = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            ದಶಮಾಂಶ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ದಶಮಾಂಶ* ಹೊರಹೋಗು_bt = ಹೊರಹೋಗು + b * T * C + t * C;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ n = (s * (x[i] - m)); // normalize
                ದಶಮಾಂಶ o = n * ತೂಕ[i] + bias[i]; // scale and shift
                ಹೊರಹೋಗು_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            ಸರಾಸರಿ[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ಶೂನ್ಯ layernorm_backward(ದಶಮಾಂಶ* dinp, ದಶಮಾಂಶ* dweight, ದಶಮಾಂಶ* dbias,
                        ದಶಮಾಂಶ* dout, ದಶಮಾಂಶ* inp, ದಶಮಾಂಶ* ತೂಕ, ದಶಮಾಂಶ* ಸರಾಸರಿ, ದಶಮಾಂಶ* rstd,
                        ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ದಶಮಾಂಶ* dout_bt = dout + b * T * C + t * C;
            ದಶಮಾಂಶ* inp_bt = inp + b * T * C + t * C;
            ದಶಮಾಂಶ* dinp_bt = dinp + b * T * C + t * C;
            ದಶಮಾಂಶ ಸರಾಸರಿ_bt = ಸರಾಸರಿ[b * T + t];
            ದಶಮಾಂಶ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ದಶಮಾಂಶ dnorm_ಸರಾಸರಿ = 0.0f;
            ದಶಮಾಂಶ dnorm_norm_ಸರಾಸರಿ = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ norm_bti = (inp_bt[i] - ಸರಾಸರಿ_bt) * rstd_bt;
                ದಶಮಾಂಶ dnorm_i = ತೂಕ[i] * dout_bt[i];
                dnorm_ಸರಾಸರಿ += dnorm_i;
                dnorm_norm_ಸರಾಸರಿ += dnorm_i * norm_bti;
            }
            dnorm_ಸರಾಸರಿ = dnorm_ಸರಾಸರಿ / C;
            dnorm_norm_ಸರಾಸರಿ = dnorm_norm_ಸರಾಸರಿ / C;

            // now iterate again and accumulate all the gradients
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ norm_bti = (inp_bt[i] - ಸರಾಸರಿ_bt) * rstd_bt;
                ದಶಮಾಂಶ dnorm_i = ತೂಕ[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ದಶಮಾಂಶ dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_ಸರಾಸರಿ; // term 2
                dval -= norm_bti * dnorm_norm_ಸರಾಸರಿ; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

ಶೂನ್ಯ matmul_forward_naive(ದಶಮಾಂಶ* ಹೊರಹೋಗು,
                         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ತೂಕ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* bias,
                         ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #ಸೂಚನೆ omp parallel ಗಾಗಿ collapse(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ಪೂರ್ಣಾಂಕ bt = b * T + t;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < OC; o++) {
                ದಶಮಾಂಶ val = (bias != ಖಾಲಿ) ? bias[o] : 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                    val += inp[bt * C + i] * ತೂಕ[o*C + i];
                }
                ಹೊರಹೋಗು[bt * OC + o] = val;
            }
        }
    }
}

ಶೂನ್ಯ matmul_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು,
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ತೂಕ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* bias,
                    ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ LOOP_UNROLL = 8;
    ಆದರೆ (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(ಹೊರಹೋಗು, inp, ತೂಕ, bias, B, T, C, OC);
        ಹಿಂತಿರುಗಿ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #ಸೂಚನೆ omp parallel ಗಾಗಿ
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            ದಶಮಾಂಶ ಫಲಿತಾಂಶ[LOOP_UNROLL];
            // initialize the bias, if it exists
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                ಫಲಿತಾಂಶ[ibt] = (bias != ಖಾಲಿ) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ w = ತೂಕ[i + o * C];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    ಪೂರ್ಣಾಂಕ bt = obt + ibt;
                    ಫಲಿತಾಂಶ[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                ಪೂರ್ಣಾಂಕ bt = obt + ibt;
                ಹೊರಹೋಗು[bt * OC + o] = ಫಲಿತಾಂಶ[ibt];
            }
        }
    }
}

ಶೂನ್ಯ matmul_backward(ದಶಮಾಂಶ* dinp, ದಶಮಾಂಶ* dweight, ದಶಮಾಂಶ* dbias,
                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* dout, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ತೂಕ,
                     ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #ಸೂಚನೆ omp parallel ಗಾಗಿ collapse(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* dout_bt = dout + b * T * OC + t * OC;
            ದಶಮಾಂಶ* dinp_bt = dinp + b * T * C + t * C;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < OC; o++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* wrow = ತೂಕ + o*C;
                ದಶಮಾಂಶ d = dout_bt[o];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #ಸೂಚನೆ omp parallel ಗಾಗಿ
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < OC; o++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* dout_bt = dout + b * T * OC + t * OC;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp_bt = inp + b * T * C + t * C;
                ದಶಮಾಂಶ* dwrow = dweight + o*C;
                ದಶಮಾಂಶ d = dout_bt[o];
                ಆದರೆ (dbias != ಖಾಲಿ) { dbias[o] += d; }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

ಶೂನ್ಯ attention_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು, ದಶಮಾಂಶ* preatt, ದಶಮಾಂಶ* att,
                       ದಶಮಾಂಶ* inp,
                       ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    ಪೂರ್ಣಾಂಕ C3 = C*3;
    ಪೂರ್ಣಾಂಕ hs = C / NH; // head size
    ದಶಮಾಂಶ scale = 1.0 / sqrtf(hs);

    #ಸೂಚನೆ omp parallel ಗಾಗಿ collapse(3)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = 0; h < NH; h++) {
                ದಶಮಾಂಶ* query_t = inp + b * T * C3 + t * C3 + h * hs;
                ದಶಮಾಂಶ* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                ದಶಮಾಂಶ* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                ದಶಮಾಂಶ maxval = -10000.0f; // TODO something better
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 <= t; t2++) {
                    ದಶಮಾಂಶ* ಕೀಲಿ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    ದಶಮಾಂಶ val = 0.0f;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < hs; i++) {
                        val += query_t[i] * ಕೀಲಿ_t2[i];
                    }
                    val *= scale;
                    ಆದರೆ (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                ದಶಮಾಂಶ expsum = 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 <= t; t2++) {
                    ದಶಮಾಂಶ expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                ದಶಮಾಂಶ expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 < T; t2++) {
                    ಆದರೆ (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                ದಶಮಾಂಶ* ಹೊರಹೋಗು_bth = ಹೊರಹೋಗು + b * T * C + t * C + h * hs;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < hs; i++) { ಹೊರಹೋಗು_bth[i] = 0.0f; }
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 <= t; t2++) {
                    ದಶಮಾಂಶ* ಮೌಲ್ಯ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    ದಶಮಾಂಶ att_btht2 = att_bth[t2];
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < hs; i++) {
                        ಹೊರಹೋಗು_bth[i] += att_btht2 * ಮೌಲ್ಯ_t2[i];
                    }
                }
            }
        }
    }
}

ಶೂನ್ಯ attention_backward(ದಶಮಾಂಶ* dinp, ದಶಮಾಂಶ* dpreatt, ದಶಮಾಂಶ* datt,
                        ದಶಮಾಂಶ* dout, ದಶಮಾಂಶ* inp, ದಶಮಾಂಶ* att,
                        ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    ಪೂರ್ಣಾಂಕ C3 = C*3;
    ಪೂರ್ಣಾಂಕ hs = C / NH; // head size
    ದಶಮಾಂಶ scale = 1.f / sqrtf(hs);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ h = 0; h < NH; h++) {
                ದಶಮಾಂಶ* att_bth = att + b*NH*T*T + h*T*T + t*T;
                ದಶಮಾಂಶ* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                ದಶಮಾಂಶ* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                ದಶಮಾಂಶ* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                ದಶಮಾಂಶ* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                ದಶಮಾಂಶ* dout_bth = dout + b * T * C + t * C + h * hs;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 <= t; t2++) {
                    ದಶಮಾಂಶ* ಮೌಲ್ಯ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    ದಶಮಾಂಶ* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += ಮೌಲ್ಯ_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 <= t; t2++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t3 = 0; t3 <= t; t3++) {
                        ದಶಮಾಂಶ indicator = t2 == t3 ? 1.0f : 0.0f;
                        ದಶಮಾಂಶ local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t2 = 0; t2 <= t; t2++) {
                    ದಶಮಾಂಶ* ಕೀಲಿ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    ದಶಮಾಂಶ* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += ಕೀಲಿ_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು GELU_SCALING_ಗುಣಕ sqrtf(2.0f / M_PI)
ಶೂನ್ಯ gelu_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು, ದಶಮಾಂಶ* inp, ಪೂರ್ಣಾಂಕ N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < N; i++) {
        ದಶಮಾಂಶ x = inp[i];
        ದಶಮಾಂಶ cube = 0.044715f * x * x * x;
        ಹೊರಹೋಗು[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_ಗುಣಕ * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#ಸೂಚನೆ float_control(precise, on, ತಳ್ಳು)
#ಆದರೆ defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#ಆದರೆ_ಅಂತ್ಯ
ಶೂನ್ಯ gelu_backward(ದಶಮಾಂಶ* dinp, ದಶಮಾಂಶ* inp, ದಶಮಾಂಶ* dout, ಪೂರ್ಣಾಂಕ N) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < N; i++) {
        ದಶಮಾಂಶ x = inp[i];
        ದಶಮಾಂಶ cube = 0.044715f * x * x * x;
        ದಶಮಾಂಶ tanh_arg = GELU_SCALING_ಗುಣಕ * (x + cube);
        ದಶಮಾಂಶ tanh_ಹೊರಹೋಗು = tanhf(tanh_arg);
        ದಶಮಾಂಶ coshf_ಹೊರಹೋಗು = coshf(tanh_arg);
        ದಶಮಾಂಶ sech_ಹೊರಹೋಗು = 1.0f / (coshf_ಹೊರಹೋಗು * coshf_ಹೊರಹೋಗು);
        ದಶಮಾಂಶ local_grad = 0.5f * (1.0f + tanh_ಹೊರಹೋಗು) + x * 0.5f * sech_ಹೊರಹೋಗು * GELU_SCALING_ಗುಣಕ * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#ಸೂಚನೆ float_control(ತೆಗೆ_2)

ಶೂನ್ಯ residual_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು, ದಶಮಾಂಶ* inp1, ದಶಮಾಂಶ* inp2, ಪೂರ್ಣಾಂಕ N) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < N; i++) {
        ಹೊರಹೋಗು[i] = inp1[i] + inp2[i];
    }
}

ಶೂನ್ಯ residual_backward(ದಶಮಾಂಶ* dinp1, ದಶಮಾಂಶ* dinp2, ದಶಮಾಂಶ* dout, ಪೂರ್ಣಾಂಕ N) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

ಶೂನ್ಯ softmax_forward(ದಶಮಾಂಶ* probs, ದಶಮಾಂಶ* logits, ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ V, ಪೂರ್ಣಾಂಕ Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #ಸೂಚನೆ omp parallel ಗಾಗಿ collapse(2)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            // probs <- softmax(logits)
            ದಶಮಾಂಶ* logits_bt = logits + b * T * Vp + t * Vp;
            ದಶಮಾಂಶ* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            ದಶಮಾಂಶ maxval = -10000.0f; // TODO something better
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < V; i++) {
                ಆದರೆ (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                ಮೊತ್ತ += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < V; i++) {
                probs_bt[i] /= ಮೊತ್ತ;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

ಶೂನ್ಯ crossentropy_forward(ದಶಮಾಂಶ* losses,
                          ದಶಮಾಂಶ* probs, ಪೂರ್ಣಾಂಕ* targets,
                          ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            // loss = -log(probs[target])
            ದಶಮಾಂಶ* probs_bt = probs + b * T * Vp + t * Vp;
            ಪೂರ್ಣಾಂಕ ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

ಶೂನ್ಯ crossentropy_softmax_backward(ದಶಮಾಂಶ* dlogits,
                           ದಶಮಾಂಶ* dlosses, ದಶಮಾಂಶ* probs, ಪೂರ್ಣಾಂಕ* targets,
                           ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ V, ಪೂರ್ಣಾಂಕ Vp) {
    // backwards through both softmax and crossentropy
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ದಶಮಾಂಶ* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            ದಶಮಾಂಶ* probs_bt = probs + b * T * Vp + t * Vp;
            ದಶಮಾಂಶ dloss = dlosses[b * T + t];
            ಪೂರ್ಣಾಂಕ ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < V; i++) {
                ದಶಮಾಂಶ p = probs_bt[i];
                ದಶಮಾಂಶ indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಪೂರ್ಣಾಂಕ max_seq_len; // max sequence length, e.g. 1024
    ಪೂರ್ಣಾಂಕ vocab_ಗಾತ್ರ; // vocab size, e.g. 50257
    ಪೂರ್ಣಾಂಕ padded_vocab_ಗಾತ್ರ; // padded to e.g. %128==0, 50304
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_layers; // number of layers, e.g. 12
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_heads; // number of heads in attention, e.g. 12
    ಪೂರ್ಣಾಂಕ channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_PARAMETER_TENSORS 16
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ದಶಮಾಂಶ* wte; // (V, C)
    ದಶಮಾಂಶ* wpe; // (maxT, C)
    ದಶಮಾಂಶ* ln1w; // (L, C)
    ದಶಮಾಂಶ* ln1b; // (L, C)
    ದಶಮಾಂಶ* qkvw; // (L, 3*C, C)
    ದಶಮಾಂಶ* qkvb; // (L, 3*C)
    ದಶಮಾಂಶ* attprojw; // (L, C, C)
    ದಶಮಾಂಶ* attprojb; // (L, C)
    ದಶಮಾಂಶ* ln2w; // (L, C)
    ದಶಮಾಂಶ* ln2b; // (L, C)
    ದಶಮಾಂಶ* fcw; // (L, 4*C, C)
    ದಶಮಾಂಶ* fcb; // (L, 4*C)
    ದಶಮಾಂಶ* fcprojw; // (L, C, 4*C)
    ದಶಮಾಂಶ* fcprojb; // (L, C)
    ದಶಮಾಂಶ* lnfw; // (C)
    ದಶಮಾಂಶ* lnfb; // (C)
} ParameterTensors;

ಶೂನ್ಯ fill_ಒಳ_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_ಗಾತ್ರ;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.ಅಂಕೆ_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
ದಶಮಾಂಶ* malloc_and_ಬಿಂದು_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t ಅಂಕೆ_parameters = 0;
    ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_PARAMETER_TENSORS; i++) {
        ಅಂಕೆ_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    ದಶಮಾಂಶ* params_ಸ್ಮರಣೆ = (ದಶಮಾಂಶ*)mallocCheck(ಅಂಕೆ_parameters * ಗಾತ್ರ(ದಶಮಾಂಶ));
    // assign all the tensors
    ದಶಮಾಂಶ** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    ದಶಮಾಂಶ* params_ಸ್ಮರಣೆ_iterator = params_ಸ್ಮರಣೆ;
    ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_ಸ್ಮರಣೆ_iterator;
        params_ಸ್ಮರಣೆ_iterator += param_sizes[i];
    }
    ಹಿಂತಿರುಗಿ params_ಸ್ಮರಣೆ;
}

#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_ACTIVATION_TENSORS 23
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ದಶಮಾಂಶ* encoded; // (B, T, C)
    ದಶಮಾಂಶ* ln1; // (L, B, T, C)
    ದಶಮಾಂಶ* ln1_ಸರಾಸರಿ; // (L, B, T)
    ದಶಮಾಂಶ* ln1_rstd; // (L, B, T)
    ದಶಮಾಂಶ* qkv; // (L, B, T, 3*C)
    ದಶಮಾಂಶ* atty; // (L, B, T, C)
    ದಶಮಾಂಶ* preatt; // (L, B, NH, T, T)
    ದಶಮಾಂಶ* att; // (L, B, NH, T, T)
    ದಶಮಾಂಶ* attproj; // (L, B, T, C)
    ದಶಮಾಂಶ* residual2; // (L, B, T, C)
    ದಶಮಾಂಶ* ln2; // (L, B, T, C)
    ದಶಮಾಂಶ* ln2_ಸರಾಸರಿ; // (L, B, T)
    ದಶಮಾಂಶ* ln2_rstd; // (L, B, T)
    ದಶಮಾಂಶ* fch; // (L, B, T, 4*C)
    ದಶಮಾಂಶ* fch_gelu; // (L, B, T, 4*C)
    ದಶಮಾಂಶ* fcproj; // (L, B, T, C)
    ದಶಮಾಂಶ* residual3; // (L, B, T, C)
    ದಶಮಾಂಶ* lnf; // (B, T, C)
    ದಶಮಾಂಶ* lnf_ಸರಾಸರಿ; // (B, T)
    ದಶಮಾಂಶ* lnf_rstd; // (B, T)
    ದಶಮಾಂಶ* logits; // (B, T, V)
    ದಶಮಾಂಶ* probs; // (B, T, V)
    ದಶಮಾಂಶ* losses; // (B, T)
} ActivationTensors;

ಶೂನ್ಯ fill_ಒಳ_activation_sizes(size_t* act_sizes, GPT2Config config, ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T) {
    size_t C = config.channels;
    size_t NH = config.ಅಂಕೆ_heads;
    size_t L = config.ಅಂಕೆ_layers;
    size_t Vp = config.padded_vocab_ಗಾತ್ರ;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

ದಶಮಾಂಶ* malloc_and_ಬಿಂದು_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t ಅಂಕೆ_activations = 0;
    ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_ACTIVATION_TENSORS; i++) {
        ಅಂಕೆ_activations += act_sizes[i];
    }
    ದಶಮಾಂಶ* acts_ಸ್ಮರಣೆ = (ದಶಮಾಂಶ*)mallocCheck(ಅಂಕೆ_activations * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_ಸರಾಸರಿ, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_ಸರಾಸರಿ,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_ಸರಾಸರಿ, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    ದಶಮಾಂಶ* acts_ಸ್ಮರಣೆ_iterator = acts_ಸ್ಮರಣೆ;
    ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_ಸ್ಮರಣೆ_iterator;
        acts_ಸ್ಮರಣೆ_iterator += act_sizes[i];
    }
    ಹಿಂತಿರುಗಿ acts_ಸ್ಮರಣೆ;
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[ಅಂಕೆ_PARAMETER_TENSORS];
    ದಶಮಾಂಶ* params_ಸ್ಮರಣೆ;
    size_t ಅಂಕೆ_parameters;
    // gradients of the weights
    ParameterTensors grads;
    ದಶಮಾಂಶ* grads_ಸ್ಮರಣೆ;
    // buffers for the AdamW optimizer
    ದಶಮಾಂಶ* m_ಸ್ಮರಣೆ;
    ದಶಮಾಂಶ* v_ಸ್ಮರಣೆ;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[ಅಂಕೆ_ACTIVATION_TENSORS];
    ದಶಮಾಂಶ* acts_ಸ್ಮರಣೆ;
    size_t ಅಂಕೆ_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    ದಶಮಾಂಶ* grads_acts_ಸ್ಮರಣೆ;
    // other run state configuration
    ಪೂರ್ಣಾಂಕ batch_ಗಾತ್ರ; // the batch size (B) of current forward pass
    ಪೂರ್ಣಾಂಕ seq_len; // the sequence length (T) of current forward pass
    ಪೂರ್ಣಾಂಕ* inputs; // the input tokens for the current forward pass
    ಪೂರ್ಣಾಂಕ* targets; // the target tokens for the current forward pass
    ದಶಮಾಂಶ ಸರಾಸರಿ_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

ಶೂನ್ಯ gpt2_build_from_checkpoint(GPT2 *model, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* checkpoint_ಮಾರ್ಗ) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_ಮಾರ್ಗ, "rb");
    ಪೂರ್ಣಾಂಕ model_ಶೀರ್ಷಿಕೆ[256];
    freadCheck(model_ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), 256, model_file);
    ಆದರೆ (model_ಶೀರ್ಷಿಕೆ[0] != 20240326) { ಬರೆ("Bad magic model file\n"); ನಿರ್ಗಮಿಸು(1); }
    ಆದರೆ (model_ಶೀರ್ಷಿಕೆ[1] != 3) {
        ಬರೆ("Bad version in model file\n");
        ಬರೆ("---> HINT: try to re-run `python train_gpt2.py`\n");
        ನಿರ್ಗಮಿಸು(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_ಶೀರ್ಷಿಕೆ[2];
    model->config.vocab_ಗಾತ್ರ = V = model_ಶೀರ್ಷಿಕೆ[3];
    model->config.ಅಂಕೆ_layers = L = model_ಶೀರ್ಷಿಕೆ[4];
    model->config.ಅಂಕೆ_heads = NH = model_ಶೀರ್ಷಿಕೆ[5];
    model->config.channels = C = model_ಶೀರ್ಷಿಕೆ[6];
    model->config.padded_vocab_ಗಾತ್ರ = Vp = model_ಶೀರ್ಷಿಕೆ[7];
    ಬರೆ("[GPT-2]\n");
    ಬರೆ("max_seq_len: %zu\n", maxT);
    ಬರೆ("vocab_size: %zu\n", V);
    ಬರೆ("padded_vocab_size: %zu\n", Vp);
    ಬರೆ("num_layers: %zu\n", L);
    ಬರೆ("num_heads: %zu\n", NH);
    ಬರೆ("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_ಒಳ_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t ಅಂಕೆ_parameters = 0;
    ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_PARAMETER_TENSORS; i++) {
        ಅಂಕೆ_parameters += model->param_sizes[i];
    }
    ಬರೆ("num_parameters: %zu\n", ಅಂಕೆ_parameters);
    model->ಅಂಕೆ_parameters = ಅಂಕೆ_parameters;

    // read in all the parameters from file
    model->params_ಸ್ಮರಣೆ = malloc_and_ಬಿಂದು_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_ಸ್ಮರಣೆ, ಗಾತ್ರ(ದಶಮಾಂಶ), ಅಂಕೆ_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_ಸ್ಮರಣೆ = ಖಾಲಿ;
    model->grads_ಸ್ಮರಣೆ = ಖಾಲಿ;
    model->m_ಸ್ಮರಣೆ = ಖಾಲಿ;
    model->v_ಸ್ಮರಣೆ = ಖಾಲಿ;
    model->grads_acts_ಸ್ಮರಣೆ = ಖಾಲಿ;
    model->inputs = ಖಾಲಿ;
    model->targets = ಖಾಲಿ;
    model->batch_ಗಾತ್ರ = 0;
    model->seq_len = 0;
    model->ಸರಾಸರಿ_loss = -1.0f; // -1.0f will designate no loss
}

ಶೂನ್ಯ gpt2_forward(GPT2 *model, ಪೂರ್ಣಾಂಕ* inputs, ಪೂರ್ಣಾಂಕ* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    ಆದರೆ (model->params_ಸ್ಮರಣೆ == ಖಾಲಿ) {
        ಬರೆ("Error: model was not initialized properly.\n");
        ನಿರ್ಗಮಿಸು(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_ಗಾತ್ರ;
    size_t Vp = model->config.padded_vocab_ಗಾತ್ರ;
    size_t L = model->config.ಅಂಕೆ_layers;
    size_t NH = model->config.ಅಂಕೆ_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    ಗಾಗಿ(ಪೂರ್ಣಾಂಕ i = 0; i < B * T; i++) {
        ಪ್ರತಿಜ್ಞಾ(0 <= inputs[i] && inputs[i] < V);
        ಆದರೆ (targets != ಖಾಲಿ) {
            ಪ್ರತಿಜ್ಞಾ(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    ಆದರೆ(model->acts_ಸ್ಮರಣೆ == ಖಾಲಿ) {
        // record the current B,T as well
        model->batch_ಗಾತ್ರ = B;
        model->seq_len = T;
        // and now allocate the space
        fill_ಒಳ_activation_sizes(model->act_sizes, model->config, B, T);
        size_t ಅಂಕೆ_activations = 0;
        ಗಾಗಿ (size_t i = 0; i < ಅಂಕೆ_ACTIVATION_TENSORS; i++) {
            ಅಂಕೆ_activations += model->act_sizes[i];
        }
        ಬರೆ("num_activations: %zu\n", ಅಂಕೆ_activations);
        model->ಅಂಕೆ_activations = ಅಂಕೆ_activations;
        model->acts_ಸ್ಮರಣೆ = malloc_and_ಬಿಂದು_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (ಪೂರ್ಣಾಂಕ*)mallocCheck(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
        model->targets = (ಪೂರ್ಣಾಂಕ*)mallocCheck(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ)); // might be unused if we never have targets but it's small
    } ಇಲ್ಲದಿದ್ದರೆ {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        ಆದರೆ (B != model->batch_ಗಾತ್ರ || T != model->seq_len) {
            ಬರೆ("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_ಗಾತ್ರ, model->seq_len, (ಪೂರ್ಣಾಂಕ)B, (ಪೂರ್ಣಾಂಕ)T);
            ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    ಸ್ಮರಣೆ_ನಕಲು(model->inputs, inputs, B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಆದರೆ (targets != ಖಾಲಿ) {
        ಸ್ಮರಣೆ_ನಕಲು(model->targets, targets, B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    ದಶಮಾಂಶ* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        ದಶಮಾಂಶ* l_ln1w = params.ln1w + l * C;
        ದಶಮಾಂಶ* l_ln1b = params.ln1b + l * C;
        ದಶಮಾಂಶ* l_qkvw = params.qkvw + l * 3*C * C;
        ದಶಮಾಂಶ* l_qkvb = params.qkvb + l * 3*C;
        ದಶಮಾಂಶ* l_attprojw = params.attprojw + l * C * C;
        ದಶಮಾಂಶ* l_attprojb = params.attprojb + l * C;
        ದಶಮಾಂಶ* l_ln2w = params.ln2w + l * C;
        ದಶಮಾಂಶ* l_ln2b = params.ln2b + l * C;
        ದಶಮಾಂಶ* l_fcw = params.fcw + l * 4*C * C;
        ದಶಮಾಂಶ* l_fcb = params.fcb + l * 4*C;
        ದಶಮಾಂಶ* l_fcprojw = params.fcprojw + l * C * 4*C;
        ದಶಮಾಂಶ* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        ದಶಮಾಂಶ* l_ln1 = acts.ln1 + l * B * T * C;
        ದಶಮಾಂಶ* l_ln1_ಸರಾಸರಿ = acts.ln1_ಸರಾಸರಿ + l * B * T;
        ದಶಮಾಂಶ* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        ದಶಮಾಂಶ* l_qkv = acts.qkv + l * B * T * 3*C;
        ದಶಮಾಂಶ* l_atty = acts.atty + l * B * T * C;
        ದಶಮಾಂಶ* l_preatt = acts.preatt + l * B * NH * T * T;
        ದಶಮಾಂಶ* l_att = acts.att + l * B * NH * T * T;
        ದಶಮಾಂಶ* l_attproj = acts.attproj + l * B * T * C;
        ದಶಮಾಂಶ* l_residual2 = acts.residual2 + l * B * T * C;
        ದಶಮಾಂಶ* l_ln2 = acts.ln2 + l * B * T * C;
        ದಶಮಾಂಶ* l_ln2_ಸರಾಸರಿ = acts.ln2_ಸರಾಸರಿ + l * B * T;
        ದಶಮಾಂಶ* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        ದಶಮಾಂಶ* l_fch = acts.fch + l * B * T * 4*C;
        ದಶಮಾಂಶ* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        ದಶಮಾಂಶ* l_fcproj = acts.fcproj + l * B * T * C;
        ದಶಮಾಂಶ* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_ಸರಾಸರಿ, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_ಸರಾಸರಿ, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_ಸರಾಸರಿ, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, ಖಾಲಿ, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    ಆದರೆ (targets != ಖಾಲಿ) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        ದಶಮಾಂಶ ಸರಾಸರಿ_loss = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i=0; i<B*T; i++) { ಸರಾಸರಿ_loss += model->acts.losses[i]; }
        ಸರಾಸರಿ_loss /= B*T;
        model->ಸರಾಸರಿ_loss = ಸರಾಸರಿ_loss;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // if we don't have targets, we don't have a loss
        model->ಸರಾಸರಿ_loss = -1.0f;
    }
}

ಶೂನ್ಯ gpt2_zero_grad(GPT2 *model) {
    ಆದರೆ(model->grads_ಸ್ಮರಣೆ != ಖಾಲಿ) { memset(model->grads_ಸ್ಮರಣೆ, 0, model->ಅಂಕೆ_parameters * ಗಾತ್ರ(ದಶಮಾಂಶ)); }
    ಆದರೆ(model->grads_acts_ಸ್ಮರಣೆ != ಖಾಲಿ) { memset(model->grads_acts_ಸ್ಮರಣೆ, 0, model->ಅಂಕೆ_activations * ಗಾತ್ರ(ದಶಮಾಂಶ)); }
}

ಶೂನ್ಯ gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    ಆದರೆ (model->ಸರಾಸರಿ_loss == -1.0f) {
        ಬರೆ("Error: must forward with targets before backward\n");
        ನಿರ್ಗಮಿಸು(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    ಆದರೆ (model->grads_ಸ್ಮರಣೆ == ಖಾಲಿ) {
        model->grads_ಸ್ಮರಣೆ = malloc_and_ಬಿಂದು_parameters(&model->grads, model->param_sizes);
        model->grads_acts_ಸ್ಮರಣೆ = malloc_and_ಬಿಂದು_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_ಗಾತ್ರ;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_ಗಾತ್ರ;
    size_t Vp = model->config.padded_vocab_ಗಾತ್ರ;
    size_t L = model->config.ಅಂಕೆ_layers;
    size_t NH = model->config.ಅಂಕೆ_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    ದಶಮಾಂಶ dloss_ಸರಾಸರಿ = 1.0f / (B*T);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_ಸರಾಸರಿ; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, ಖಾಲಿ, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    ದಶಮಾಂಶ* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    ದಶಮಾಂಶ* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_ಸರಾಸರಿ, acts.lnf_rstd, B, T, C);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        ದಶಮಾಂಶ* l_ln1w = params.ln1w + l * C;
        ದಶಮಾಂಶ* l_qkvw = params.qkvw + l * 3*C * C;
        ದಶಮಾಂಶ* l_attprojw = params.attprojw + l * C * C;
        ದಶಮಾಂಶ* l_ln2w = params.ln2w + l * C;
        ದಶಮಾಂಶ* l_fcw = params.fcw + l * 4*C * C;
        ದಶಮಾಂಶ* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        ದಶಮಾಂಶ* dl_ln1w = grads.ln1w + l * C;
        ದಶಮಾಂಶ* dl_ln1b = grads.ln1b + l * C;
        ದಶಮಾಂಶ* dl_qkvw = grads.qkvw + l * 3*C * C;
        ದಶಮಾಂಶ* dl_qkvb = grads.qkvb + l * 3*C;
        ದಶಮಾಂಶ* dl_attprojw = grads.attprojw + l * C * C;
        ದಶಮಾಂಶ* dl_attprojb = grads.attprojb + l * C;
        ದಶಮಾಂಶ* dl_ln2w = grads.ln2w + l * C;
        ದಶಮಾಂಶ* dl_ln2b = grads.ln2b + l * C;
        ದಶಮಾಂಶ* dl_fcw = grads.fcw + l * 4*C * C;
        ದಶಮಾಂಶ* dl_fcb = grads.fcb + l * 4*C;
        ದಶಮಾಂಶ* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        ದಶಮಾಂಶ* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        ದಶಮಾಂಶ* l_ln1 = acts.ln1 + l * B * T * C;
        ದಶಮಾಂಶ* l_ln1_ಸರಾಸರಿ = acts.ln1_ಸರಾಸರಿ + l * B * T;
        ದಶಮಾಂಶ* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        ದಶಮಾಂಶ* l_qkv = acts.qkv + l * B * T * 3*C;
        ದಶಮಾಂಶ* l_atty = acts.atty + l * B * T * C;
        ದಶಮಾಂಶ* l_att = acts.att + l * B * NH * T * T;
        ದಶಮಾಂಶ* l_residual2 = acts.residual2 + l * B * T * C;
        ದಶಮಾಂಶ* l_ln2 = acts.ln2 + l * B * T * C;
        ದಶಮಾಂಶ* l_ln2_ಸರಾಸರಿ = acts.ln2_ಸರಾಸರಿ + l * B * T;
        ದಶಮಾಂಶ* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        ದಶಮಾಂಶ* l_fch = acts.fch + l * B * T * 4*C;
        ದಶಮಾಂಶ* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        ದಶಮಾಂಶ* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        ದಶಮಾಂಶ* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        ದಶಮಾಂಶ* dl_atty = grads_acts.atty + l * B * T * C;
        ದಶಮಾಂಶ* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        ದಶಮಾಂಶ* dl_att = grads_acts.att + l * B * NH * T * T;
        ದಶಮಾಂಶ* dl_attproj = grads_acts.attproj + l * B * T * C;
        ದಶಮಾಂಶ* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        ದಶಮಾಂಶ* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        ದಶಮಾಂಶ* dl_fch = grads_acts.fch + l * B * T * 4*C;
        ದಶಮಾಂಶ* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        ದಶಮಾಂಶ* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        ದಶಮಾಂಶ* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_ಸರಾಸರಿ, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_ಸರಾಸರಿ, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

ಶೂನ್ಯ gpt2_update(GPT2 *model, ದಶಮಾಂಶ learning_ದರ, ದಶಮಾಂಶ beta1, ದಶಮಾಂಶ beta2, ದಶಮಾಂಶ eps, ದಶಮಾಂಶ ತೂಕ_decay, ಪೂರ್ಣಾಂಕ t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    ಆದರೆ (model->m_ಸ್ಮರಣೆ == ಖಾಲಿ) {
        model->m_ಸ್ಮರಣೆ = (ದಶಮಾಂಶ*)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(model->ಅಂಕೆ_parameters, ಗಾತ್ರ(ದಶಮಾಂಶ));
        model->v_ಸ್ಮರಣೆ = (ದಶಮಾಂಶ*)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(model->ಅಂಕೆ_parameters, ಗಾತ್ರ(ದಶಮಾಂಶ));
    }

    ಗಾಗಿ (size_t i = 0; i < model->ಅಂಕೆ_parameters; i++) {
        ದಶಮಾಂಶ param = model->params_ಸ್ಮರಣೆ[i];
        ದಶಮಾಂಶ grad = model->grads_ಸ್ಮರಣೆ[i];

        // update the first moment (momentum)
        ದಶಮಾಂಶ m = beta1 * model->m_ಸ್ಮರಣೆ[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        ದಶಮಾಂಶ v = beta2 * model->v_ಸ್ಮರಣೆ[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        ದಶಮಾಂಶ m_hat = m / (1.0f - powf(beta1, t));
        ದಶಮಾಂಶ v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_ಸ್ಮರಣೆ[i] = m;
        model->v_ಸ್ಮರಣೆ[i] = v;
        model->params_ಸ್ಮರಣೆ[i] -= learning_ದರ * (m_hat / (sqrtf(v_hat) + eps) + ತೂಕ_decay * param);
    }
}

ಶೂನ್ಯ gpt2_free(GPT2 *model) {
    ಬಿಡುಗಡೆ(model->params_ಸ್ಮರಣೆ);
    ಬಿಡುಗಡೆ(model->grads_ಸ್ಮರಣೆ);
    ಬಿಡುಗಡೆ(model->m_ಸ್ಮರಣೆ);
    ಬಿಡುಗಡೆ(model->v_ಸ್ಮರಣೆ);
    ಬಿಡುಗಡೆ(model->acts_ಸ್ಮರಣೆ);
    ಬಿಡುಗಡೆ(model->grads_acts_ಸ್ಮರಣೆ);
    ಬಿಡುಗಡೆ(model->inputs);
    ಬಿಡುಗಡೆ(model->targets);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ಹಿಂತಿರುಗಿ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
ದಶಮಾಂಶ random_f32(uint64_t *state) { // random float32 in [0,1)
    ಹಿಂತಿರುಗಿ (random_u32(state) >> 8) / 16777216.0f;
}

ಪೂರ್ಣಾಂಕ sample_mult(ದಶಮಾಂಶ* probabilities, ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    ದಶಮಾಂಶ cdf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        cdf += probabilities[i];
        ಆದರೆ (coin < cdf) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    ಹಿಂತಿರುಗಿ n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* train_ಟೋಕನುಗಳು = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* val_ಟೋಕನುಗಳು = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    ಪೂರ್ಣಾಂಕ B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    ಪೂರ್ಣಾಂಕ T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    ದತ್ತಾಂಶ_Loader train_loader, val_loader;
    dataloader_init(&train_loader, train_ಟೋಕನುಗಳು, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_ಟೋಕನುಗಳು, B, T, 0, 1, 0);
    ಬರೆ("train dataset num_batches: %zu\n", train_loader.ಅಂಕೆ_ಟೋಕನುಗಳು / (B*T));
    ಬರೆ("val dataset num_batches: %zu\n", val_loader.ಅಂಕೆ_ಟೋಕನುಗಳು / (B*T));
    ಪೂರ್ಣಾಂಕ val_ಅಂಕೆ_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    ಪೂರ್ಣಾಂಕ* gen_ಟೋಕನುಗಳು = (ಪೂರ್ಣಾಂಕ*)mallocCheck(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ genT = 64; // number of steps of inference we will do

    // train
    ರಚನೆ timespec ಆರಂಭ, ಕೊನೆ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ <= 40; ಹೆಜ್ಜೆ++) {

        // once in a while estimate the validation loss
        ಆದರೆ (ಹೆಜ್ಜೆ % 10 == 0) {
            ದಶಮಾಂಶ val_loss = 0.0f;
            dataloader_reset(&val_loader);
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < val_ಅಂಕೆ_batches; i++) {
                dataloader_ಮುಂದಿನ_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.ಸರಾಸರಿ_loss;
            }
            val_loss /= val_ಅಂಕೆ_batches;
            ಬರೆ("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        ಆದರೆ (ಹೆಜ್ಜೆ > 0 && ಹೆಜ್ಜೆ % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            ಗಾಗಿ(ಪೂರ್ಣಾಂಕ i = 0; i < B * T; ++i) {
                gen_ಟೋಕನುಗಳು[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            ಬರೆ("generating:\n---\n");
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_ಟೋಕನುಗಳು, ಖಾಲಿ, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                ದಶಮಾಂಶ* probs = model.acts.probs + (t-1) * model.config.padded_vocab_ಗಾತ್ರ;
                ದಶಮಾಂಶ coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                ಪೂರ್ಣಾಂಕ ಮುಂದಿನ_token = sample_mult(probs, model.config.vocab_ಗಾತ್ರ, coin);
                gen_ಟೋಕನುಗಳು[t] = ಮುಂದಿನ_token;
                // print the generated token, either using the Tokenizer or a fallback
                ಆದರೆ (tokenizer.init_ok) {
                    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* token_str = tokenizer_decode(&tokenizer, ಮುಂದಿನ_token);
                    ಸುರಕ್ಷಿತ_printf(token_str);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // fall back to printing the token id
                    ಬರೆ("%d ", ಮುಂದಿನ_token);
                }
                fflush(stdout);
            }
            ಬರೆ("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &ಆರಂಭ);
        dataloader_ಮುಂದಿನ_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, ಹೆಜ್ಜೆ+1);
        clock_gettime(CLOCK_MONOTONIC, &ಕೊನೆ);
        ದ್ವಿಗುಣ ಸಮಯ_elapsed_s = (ಕೊನೆ.tv_sec - ಆರಂಭ.tv_sec) + (ಕೊನೆ.tv_nsec - ಆರಂಭ.tv_nsec) / 1e9;
        ಬರೆ("step %d: train loss %f (took %f ms)\n", ಹೆಜ್ಜೆ, model.ಸರಾಸರಿ_loss, ಸಮಯ_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    ಬಿಡುಗಡೆ(gen_ಟೋಕನುಗಳು);
    ಹಿಂತಿರುಗಿ 0;
}
#ಆದರೆ_ಅಂತ್ಯ
