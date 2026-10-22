/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <ctype.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ OMP
#ਅਨ੍ਤਰ੍ਭਾਵ <omp.h>
#ਜੇ_ਅੰਤ
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ਅਨ੍ਤਰ੍ਭਾਵ "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#ਅਨ੍ਤਰ੍ਭਾਵ "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#ਅਨ੍ਤਰ੍ਭਾਵ "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

ਖਾਲੀ encoder_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ,
                   ਪੂਰਨ* inp, ਦਸ਼ਮਲਵ* wte, ਦਸ਼ਮਲਵ* wpe,
                   ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ_bt = ਬਾਹਰਲਾ + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            ਪੂਰਨ ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            ਦਸ਼ਮਲਵ* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            ਦਸ਼ਮਲਵ* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਬਾਹਰਲਾ_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

ਖਾਲੀ encoder_backward(ਦਸ਼ਮਲਵ* dwte, ਦਸ਼ਮਲਵ* dwpe,
                      ਦਸ਼ਮਲਵ* dout, ਪੂਰਨ* inp,
                      ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C) {
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਦਸ਼ਮਲਵ* dout_bt = dout + b * T * C + t * C;
            ਪੂਰਨ ix = inp[b * T + t];
            ਦਸ਼ਮਲਵ* dwte_ix = dwte + ix * C;
            ਦਸ਼ਮਲਵ* dwpe_t = dwpe + t * C;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

ਖਾਲੀ layernorm_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ, ਦਸ਼ਮਲਵ* ਔਸਤ, ਦਸ਼ਮਲਵ* rstd,
                       ਦਸ਼ਮਲਵ* inp, ਦਸ਼ਮਲਵ* ਭਾਰ, ਦਸ਼ਮਲਵ* bias,
                       ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    ਦਸ਼ਮਲਵ eps = 1e-5f;
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ਦਸ਼ਮਲਵ* x = inp + b * T * C + t * C;
            // calculate the mean
            ਦਸ਼ਮਲਵ m = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ਦਸ਼ਮਲਵ v = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            ਦਸ਼ਮਲਵ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ_bt = ਬਾਹਰਲਾ + b * T * C + t * C;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ n = (s * (x[i] - m)); // normalize
                ਦਸ਼ਮਲਵ o = n * ਭਾਰ[i] + bias[i]; // scale and shift
                ਬਾਹਰਲਾ_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            ਔਸਤ[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ਖਾਲੀ layernorm_backward(ਦਸ਼ਮਲਵ* dinp, ਦਸ਼ਮਲਵ* dweight, ਦਸ਼ਮਲਵ* dbias,
                        ਦਸ਼ਮਲਵ* dout, ਦਸ਼ਮਲਵ* inp, ਦਸ਼ਮਲਵ* ਭਾਰ, ਦਸ਼ਮਲਵ* ਔਸਤ, ਦਸ਼ਮਲਵ* rstd,
                        ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C) {
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਦਸ਼ਮਲਵ* dout_bt = dout + b * T * C + t * C;
            ਦਸ਼ਮਲਵ* inp_bt = inp + b * T * C + t * C;
            ਦਸ਼ਮਲਵ* dinp_bt = dinp + b * T * C + t * C;
            ਦਸ਼ਮਲਵ ਔਸਤ_bt = ਔਸਤ[b * T + t];
            ਦਸ਼ਮਲਵ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ਦਸ਼ਮਲਵ dnorm_ਔਸਤ = 0.0f;
            ਦਸ਼ਮਲਵ dnorm_norm_ਔਸਤ = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ norm_bti = (inp_bt[i] - ਔਸਤ_bt) * rstd_bt;
                ਦਸ਼ਮਲਵ dnorm_i = ਭਾਰ[i] * dout_bt[i];
                dnorm_ਔਸਤ += dnorm_i;
                dnorm_norm_ਔਸਤ += dnorm_i * norm_bti;
            }
            dnorm_ਔਸਤ = dnorm_ਔਸਤ / C;
            dnorm_norm_ਔਸਤ = dnorm_norm_ਔਸਤ / C;

            // now iterate again and accumulate all the gradients
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ norm_bti = (inp_bt[i] - ਔਸਤ_bt) * rstd_bt;
                ਦਸ਼ਮਲਵ dnorm_i = ਭਾਰ[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ਦਸ਼ਮਲਵ dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_ਔਸਤ; // term 2
                dval -= norm_bti * dnorm_norm_ਔਸਤ; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

ਖਾਲੀ matmul_forward_naive(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ,
                         ਅਚਲ ਦਸ਼ਮਲਵ* inp, ਅਚਲ ਦਸ਼ਮਲਵ* ਭਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ* bias,
                         ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #ਹਦਾਇਤ omp parallel ਲਈ collapse(2)
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਪੂਰਨ bt = b * T + t;
            ਲਈ (ਪੂਰਨ o = 0; o < OC; o++) {
                ਦਸ਼ਮਲਵ val = (bias != ਨਿਰਾ) ? bias[o] : 0.0f;
                ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                    val += inp[bt * C + i] * ਭਾਰ[o*C + i];
                }
                ਬਾਹਰਲਾ[bt * OC + o] = val;
            }
        }
    }
}

ਖਾਲੀ matmul_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ,
                    ਅਚਲ ਦਸ਼ਮਲਵ* inp, ਅਚਲ ਦਸ਼ਮਲਵ* ਭਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ* bias,
                    ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    ਅਚਲ ਪੂਰਨ LOOP_UNROLL = 8;
    ਜੇ (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(ਬਾਹਰਲਾ, inp, ਭਾਰ, bias, B, T, C, OC);
        ਵਾਪਸ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #ਹਦਾਇਤ omp parallel ਲਈ
    ਲਈ (ਪੂਰਨ obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        ਲਈ (ਪੂਰਨ o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            ਦਸ਼ਮਲਵ ਨਤੀਜਾ[LOOP_UNROLL];
            // initialize the bias, if it exists
            ਲਈ (ਪੂਰਨ ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                ਨਤੀਜਾ[ibt] = (bias != ਨਿਰਾ) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ w = ਭਾਰ[i + o * C];
                ਲਈ (ਪੂਰਨ ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    ਪੂਰਨ bt = obt + ibt;
                    ਨਤੀਜਾ[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            ਲਈ (ਪੂਰਨ ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                ਪੂਰਨ bt = obt + ibt;
                ਬਾਹਰਲਾ[bt * OC + o] = ਨਤੀਜਾ[ibt];
            }
        }
    }
}

ਖਾਲੀ matmul_backward(ਦਸ਼ਮਲਵ* dinp, ਦਸ਼ਮਲਵ* dweight, ਦਸ਼ਮਲਵ* dbias,
                     ਅਚਲ ਦਸ਼ਮਲਵ* dout, ਅਚਲ ਦਸ਼ਮਲਵ* inp, ਅਚਲ ਦਸ਼ਮਲਵ* ਭਾਰ,
                     ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #ਹਦਾਇਤ omp parallel ਲਈ collapse(2)
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਅਚਲ ਦਸ਼ਮਲਵ* dout_bt = dout + b * T * OC + t * OC;
            ਦਸ਼ਮਲਵ* dinp_bt = dinp + b * T * C + t * C;
            ਲਈ (ਪੂਰਨ o = 0; o < OC; o++) {
                ਅਚਲ ਦਸ਼ਮਲਵ* wrow = ਭਾਰ + o*C;
                ਦਸ਼ਮਲਵ d = dout_bt[o];
                ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #ਹਦਾਇਤ omp parallel ਲਈ
    ਲਈ (ਪੂਰਨ o = 0; o < OC; o++) {
        ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
            ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
                ਅਚਲ ਦਸ਼ਮਲਵ* dout_bt = dout + b * T * OC + t * OC;
                ਅਚਲ ਦਸ਼ਮਲਵ* inp_bt = inp + b * T * C + t * C;
                ਦਸ਼ਮਲਵ* dwrow = dweight + o*C;
                ਦਸ਼ਮਲਵ d = dout_bt[o];
                ਜੇ (dbias != ਨਿਰਾ) { dbias[o] += d; }
                ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

ਖਾਲੀ attention_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ, ਦਸ਼ਮਲਵ* preatt, ਦਸ਼ਮਲਵ* att,
                       ਦਸ਼ਮਲਵ* inp,
                       ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    ਪੂਰਨ C3 = C*3;
    ਪੂਰਨ hs = C / NH; // head size
    ਦਸ਼ਮਲਵ scale = 1.0 / sqrtf(hs);

    #ਹਦਾਇਤ omp parallel ਲਈ collapse(3)
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਲਈ (ਪੂਰਨ h = 0; h < NH; h++) {
                ਦਸ਼ਮਲਵ* query_t = inp + b * T * C3 + t * C3 + h * hs;
                ਦਸ਼ਮਲਵ* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                ਦਸ਼ਮਲਵ* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                ਦਸ਼ਮਲਵ maxval = -10000.0f; // TODO something better
                ਲਈ (ਪੂਰਨ t2 = 0; t2 <= t; t2++) {
                    ਦਸ਼ਮਲਵ* ਕੁੰਜੀ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    ਦਸ਼ਮਲਵ val = 0.0f;
                    ਲਈ (ਪੂਰਨ i = 0; i < hs; i++) {
                        val += query_t[i] * ਕੁੰਜੀ_t2[i];
                    }
                    val *= scale;
                    ਜੇ (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                ਦਸ਼ਮਲਵ expsum = 0.0f;
                ਲਈ (ਪੂਰਨ t2 = 0; t2 <= t; t2++) {
                    ਦਸ਼ਮਲਵ expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                ਦਸ਼ਮਲਵ expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                ਲਈ (ਪੂਰਨ t2 = 0; t2 < T; t2++) {
                    ਜੇ (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } ਨਹੀਂਤਾਂ {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ_bth = ਬਾਹਰਲਾ + b * T * C + t * C + h * hs;
                ਲਈ (ਪੂਰਨ i = 0; i < hs; i++) { ਬਾਹਰਲਾ_bth[i] = 0.0f; }
                ਲਈ (ਪੂਰਨ t2 = 0; t2 <= t; t2++) {
                    ਦਸ਼ਮਲਵ* ਮੁੱਲ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    ਦਸ਼ਮਲਵ att_btht2 = att_bth[t2];
                    ਲਈ (ਪੂਰਨ i = 0; i < hs; i++) {
                        ਬਾਹਰਲਾ_bth[i] += att_btht2 * ਮੁੱਲ_t2[i];
                    }
                }
            }
        }
    }
}

ਖਾਲੀ attention_backward(ਦਸ਼ਮਲਵ* dinp, ਦਸ਼ਮਲਵ* dpreatt, ਦਸ਼ਮਲਵ* datt,
                        ਦਸ਼ਮਲਵ* dout, ਦਸ਼ਮਲਵ* inp, ਦਸ਼ਮਲਵ* att,
                        ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    ਪੂਰਨ C3 = C*3;
    ਪੂਰਨ hs = C / NH; // head size
    ਦਸ਼ਮਲਵ scale = 1.f / sqrtf(hs);

    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਲਈ (ਪੂਰਨ h = 0; h < NH; h++) {
                ਦਸ਼ਮਲਵ* att_bth = att + b*NH*T*T + h*T*T + t*T;
                ਦਸ਼ਮਲਵ* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                ਦਸ਼ਮਲਵ* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                ਦਸ਼ਮਲਵ* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                ਦਸ਼ਮਲਵ* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                ਦਸ਼ਮਲਵ* dout_bth = dout + b * T * C + t * C + h * hs;
                ਲਈ (ਪੂਰਨ t2 = 0; t2 <= t; t2++) {
                    ਦਸ਼ਮਲਵ* ਮੁੱਲ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    ਦਸ਼ਮਲਵ* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    ਲਈ (ਪੂਰਨ i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += ਮੁੱਲ_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                ਲਈ (ਪੂਰਨ t2 = 0; t2 <= t; t2++) {
                    ਲਈ (ਪੂਰਨ t3 = 0; t3 <= t; t3++) {
                        ਦਸ਼ਮਲਵ indicator = t2 == t3 ? 1.0f : 0.0f;
                        ਦਸ਼ਮਲਵ local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                ਲਈ (ਪੂਰਨ t2 = 0; t2 <= t; t2++) {
                    ਦਸ਼ਮਲਵ* ਕੁੰਜੀ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    ਦਸ਼ਮਲਵ* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    ਲਈ (ਪੂਰਨ i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += ਕੁੰਜੀ_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GELU_SCALING_ਗੁਣਕ sqrtf(2.0f / M_PI)
ਖਾਲੀ gelu_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ, ਦਸ਼ਮਲਵ* inp, ਪੂਰਨ N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    ਲਈ (ਪੂਰਨ i = 0; i < N; i++) {
        ਦਸ਼ਮਲਵ x = inp[i];
        ਦਸ਼ਮਲਵ cube = 0.044715f * x * x * x;
        ਬਾਹਰਲਾ[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_ਗੁਣਕ * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#ਹਦਾਇਤ float_control(precise, on, ਧੱਕੋ)
#ਜੇ defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#ਜੇ_ਅੰਤ
ਖਾਲੀ gelu_backward(ਦਸ਼ਮਲਵ* dinp, ਦਸ਼ਮਲਵ* inp, ਦਸ਼ਮਲਵ* dout, ਪੂਰਨ N) {
    ਲਈ (ਪੂਰਨ i = 0; i < N; i++) {
        ਦਸ਼ਮਲਵ x = inp[i];
        ਦਸ਼ਮਲਵ cube = 0.044715f * x * x * x;
        ਦਸ਼ਮਲਵ tanh_arg = GELU_SCALING_ਗੁਣਕ * (x + cube);
        ਦਸ਼ਮਲਵ tanh_ਬਾਹਰਲਾ = tanhf(tanh_arg);
        ਦਸ਼ਮਲਵ coshf_ਬਾਹਰਲਾ = coshf(tanh_arg);
        ਦਸ਼ਮਲਵ sech_ਬਾਹਰਲਾ = 1.0f / (coshf_ਬਾਹਰਲਾ * coshf_ਬਾਹਰਲਾ);
        ਦਸ਼ਮਲਵ local_grad = 0.5f * (1.0f + tanh_ਬਾਹਰਲਾ) + x * 0.5f * sech_ਬਾਹਰਲਾ * GELU_SCALING_ਗੁਣਕ * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#ਹਦਾਇਤ float_control(ਕੱਢੋ)

ਖਾਲੀ residual_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ, ਦਸ਼ਮਲਵ* inp1, ਦਸ਼ਮਲਵ* inp2, ਪੂਰਨ N) {
    ਲਈ (ਪੂਰਨ i = 0; i < N; i++) {
        ਬਾਹਰਲਾ[i] = inp1[i] + inp2[i];
    }
}

ਖਾਲੀ residual_backward(ਦਸ਼ਮਲਵ* dinp1, ਦਸ਼ਮਲਵ* dinp2, ਦਸ਼ਮਲਵ* dout, ਪੂਰਨ N) {
    ਲਈ (ਪੂਰਨ i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

ਖਾਲੀ softmax_forward(ਦਸ਼ਮਲਵ* probs, ਦਸ਼ਮਲਵ* logits, ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ V, ਪੂਰਨ Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #ਹਦਾਇਤ omp parallel ਲਈ collapse(2)
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            // probs <- softmax(logits)
            ਦਸ਼ਮਲਵ* logits_bt = logits + b * T * Vp + t * Vp;
            ਦਸ਼ਮਲਵ* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            ਦਸ਼ਮਲਵ maxval = -10000.0f; // TODO something better
            ਲਈ (ਪੂਰਨ i = 0; i < V; i++) {
                ਜੇ (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                ਜੋੜ += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            ਲਈ (ਪੂਰਨ i = 0; i < V; i++) {
                probs_bt[i] /= ਜੋੜ;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            ਲਈ (ਪੂਰਨ i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

ਖਾਲੀ crossentropy_forward(ਦਸ਼ਮਲਵ* losses,
                          ਦਸ਼ਮਲਵ* probs, ਪੂਰਨ* targets,
                          ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            // loss = -log(probs[target])
            ਦਸ਼ਮਲਵ* probs_bt = probs + b * T * Vp + t * Vp;
            ਪੂਰਨ ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

ਖਾਲੀ crossentropy_softmax_backward(ਦਸ਼ਮਲਵ* dlogits,
                           ਦਸ਼ਮਲਵ* dlosses, ਦਸ਼ਮਲਵ* probs, ਪੂਰਨ* targets,
                           ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ V, ਪੂਰਨ Vp) {
    // backwards through both softmax and crossentropy
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਦਸ਼ਮਲਵ* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            ਦਸ਼ਮਲਵ* probs_bt = probs + b * T * Vp + t * Vp;
            ਦਸ਼ਮਲਵ dloss = dlosses[b * T + t];
            ਪੂਰਨ ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            ਲਈ (ਪੂਰਨ i = 0; i < V; i++) {
                ਦਸ਼ਮਲਵ p = probs_bt[i];
                ਦਸ਼ਮਲਵ indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਪੂਰਨ max_seq_len; // max sequence length, e.g. 1024
    ਪੂਰਨ vocab_ਮਾਪ; // vocab size, e.g. 50257
    ਪੂਰਨ padded_vocab_ਮਾਪ; // padded to e.g. %128==0, 50304
    ਪੂਰਨ ਅੰਕ_layers; // number of layers, e.g. 12
    ਪੂਰਨ ਅੰਕ_heads; // number of heads in attention, e.g. 12
    ਪੂਰਨ channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_PARAMETER_TENSORS 16
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਦਸ਼ਮਲਵ* wte; // (V, C)
    ਦਸ਼ਮਲਵ* wpe; // (maxT, C)
    ਦਸ਼ਮਲਵ* ln1w; // (L, C)
    ਦਸ਼ਮਲਵ* ln1b; // (L, C)
    ਦਸ਼ਮਲਵ* qkvw; // (L, 3*C, C)
    ਦਸ਼ਮਲਵ* qkvb; // (L, 3*C)
    ਦਸ਼ਮਲਵ* attprojw; // (L, C, C)
    ਦਸ਼ਮਲਵ* attprojb; // (L, C)
    ਦਸ਼ਮਲਵ* ln2w; // (L, C)
    ਦਸ਼ਮਲਵ* ln2b; // (L, C)
    ਦਸ਼ਮਲਵ* fcw; // (L, 4*C, C)
    ਦਸ਼ਮਲਵ* fcb; // (L, 4*C)
    ਦਸ਼ਮਲਵ* fcprojw; // (L, C, 4*C)
    ਦਸ਼ਮਲਵ* fcprojb; // (L, C)
    ਦਸ਼ਮਲਵ* lnfw; // (C)
    ਦਸ਼ਮਲਵ* lnfb; // (C)
} ParameterTensors;

ਖਾਲੀ fill_ਵਿੱਚ_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_ਮਾਪ;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.ਅੰਕ_layers;
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
ਦਸ਼ਮਲਵ* malloc_and_ਬਿੰਦੂ_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t ਅੰਕ_parameters = 0;
    ਲਈ (size_t i = 0; i < ਅੰਕ_PARAMETER_TENSORS; i++) {
        ਅੰਕ_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    ਦਸ਼ਮਲਵ* params_ਸਮ੍ਰਿਤੀ = (ਦਸ਼ਮਲਵ*)mallocCheck(ਅੰਕ_parameters * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    // assign all the tensors
    ਦਸ਼ਮਲਵ** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    ਦਸ਼ਮਲਵ* params_ਸਮ੍ਰਿਤੀ_iterator = params_ਸਮ੍ਰਿਤੀ;
    ਲਈ (size_t i = 0; i < ਅੰਕ_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_ਸਮ੍ਰਿਤੀ_iterator;
        params_ਸਮ੍ਰਿਤੀ_iterator += param_sizes[i];
    }
    ਵਾਪਸ params_ਸਮ੍ਰਿਤੀ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_ACTIVATION_TENSORS 23
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਦਸ਼ਮਲਵ* encoded; // (B, T, C)
    ਦਸ਼ਮਲਵ* ln1; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* ln1_ਔਸਤ; // (L, B, T)
    ਦਸ਼ਮਲਵ* ln1_rstd; // (L, B, T)
    ਦਸ਼ਮਲਵ* qkv; // (L, B, T, 3*C)
    ਦਸ਼ਮਲਵ* atty; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* preatt; // (L, B, NH, T, T)
    ਦਸ਼ਮਲਵ* att; // (L, B, NH, T, T)
    ਦਸ਼ਮਲਵ* attproj; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* residual2; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* ln2; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* ln2_ਔਸਤ; // (L, B, T)
    ਦਸ਼ਮਲਵ* ln2_rstd; // (L, B, T)
    ਦਸ਼ਮਲਵ* fch; // (L, B, T, 4*C)
    ਦਸ਼ਮਲਵ* fch_gelu; // (L, B, T, 4*C)
    ਦਸ਼ਮਲਵ* fcproj; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* residual3; // (L, B, T, C)
    ਦਸ਼ਮਲਵ* lnf; // (B, T, C)
    ਦਸ਼ਮਲਵ* lnf_ਔਸਤ; // (B, T)
    ਦਸ਼ਮਲਵ* lnf_rstd; // (B, T)
    ਦਸ਼ਮਲਵ* logits; // (B, T, V)
    ਦਸ਼ਮਲਵ* probs; // (B, T, V)
    ਦਸ਼ਮਲਵ* losses; // (B, T)
} ActivationTensors;

ਖਾਲੀ fill_ਵਿੱਚ_activation_sizes(size_t* act_sizes, GPT2Config config, ਪੂਰਨ B, ਪੂਰਨ T) {
    size_t C = config.channels;
    size_t NH = config.ਅੰਕ_heads;
    size_t L = config.ਅੰਕ_layers;
    size_t Vp = config.padded_vocab_ਮਾਪ;
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

ਦਸ਼ਮਲਵ* malloc_and_ਬਿੰਦੂ_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t ਅੰਕ_activations = 0;
    ਲਈ (size_t i = 0; i < ਅੰਕ_ACTIVATION_TENSORS; i++) {
        ਅੰਕ_activations += act_sizes[i];
    }
    ਦਸ਼ਮਲਵ* acts_ਸਮ੍ਰਿਤੀ = (ਦਸ਼ਮਲਵ*)mallocCheck(ਅੰਕ_activations * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_ਔਸਤ, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_ਔਸਤ,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_ਔਸਤ, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    ਦਸ਼ਮਲਵ* acts_ਸਮ੍ਰਿਤੀ_iterator = acts_ਸਮ੍ਰਿਤੀ;
    ਲਈ (size_t i = 0; i < ਅੰਕ_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_ਸਮ੍ਰਿਤੀ_iterator;
        acts_ਸਮ੍ਰਿਤੀ_iterator += act_sizes[i];
    }
    ਵਾਪਸ acts_ਸਮ੍ਰਿਤੀ;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[ਅੰਕ_PARAMETER_TENSORS];
    ਦਸ਼ਮਲਵ* params_ਸਮ੍ਰਿਤੀ;
    size_t ਅੰਕ_parameters;
    // gradients of the weights
    ParameterTensors grads;
    ਦਸ਼ਮਲਵ* grads_ਸਮ੍ਰਿਤੀ;
    // buffers for the AdamW optimizer
    ਦਸ਼ਮਲਵ* m_ਸਮ੍ਰਿਤੀ;
    ਦਸ਼ਮਲਵ* v_ਸਮ੍ਰਿਤੀ;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[ਅੰਕ_ACTIVATION_TENSORS];
    ਦਸ਼ਮਲਵ* acts_ਸਮ੍ਰਿਤੀ;
    size_t ਅੰਕ_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    ਦਸ਼ਮਲਵ* grads_acts_ਸਮ੍ਰਿਤੀ;
    // other run state configuration
    ਪੂਰਨ batch_ਮਾਪ; // the batch size (B) of current forward pass
    ਪੂਰਨ seq_len; // the sequence length (T) of current forward pass
    ਪੂਰਨ* inputs; // the input tokens for the current forward pass
    ਪੂਰਨ* targets; // the target tokens for the current forward pass
    ਦਸ਼ਮਲਵ ਔਸਤ_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

ਖਾਲੀ gpt2_build_from_checkpoint(GPT2 *model, ਅਚਲ ਅੱਖਰ* checkpoint_ਰਸਤਾ) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_ਰਸਤਾ, "rb");
    ਪੂਰਨ model_ਸਿਰਲੇਖ[256];
    freadCheck(model_ਸਿਰਲੇਖ, ਆਕਾਰ(ਪੂਰਨ), 256, model_file);
    ਜੇ (model_ਸਿਰਲੇਖ[0] != 20240326) { ਲਿਖੋ("Bad magic model file\n"); ਬਾਹਰ_ਨਿਕਲੋ(1); }
    ਜੇ (model_ਸਿਰਲੇਖ[1] != 3) {
        ਲਿਖੋ("Bad version in model file\n");
        ਲਿਖੋ("---> HINT: try to re-run `python train_gpt2.py`\n");
        ਬਾਹਰ_ਨਿਕਲੋ(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_ਸਿਰਲੇਖ[2];
    model->config.vocab_ਮਾਪ = V = model_ਸਿਰਲੇਖ[3];
    model->config.ਅੰਕ_layers = L = model_ਸਿਰਲੇਖ[4];
    model->config.ਅੰਕ_heads = NH = model_ਸਿਰਲੇਖ[5];
    model->config.channels = C = model_ਸਿਰਲੇਖ[6];
    model->config.padded_vocab_ਮਾਪ = Vp = model_ਸਿਰਲੇਖ[7];
    ਲਿਖੋ("[GPT-2]\n");
    ਲਿਖੋ("max_seq_len: %zu\n", maxT);
    ਲਿਖੋ("vocab_size: %zu\n", V);
    ਲਿਖੋ("padded_vocab_size: %zu\n", Vp);
    ਲਿਖੋ("num_layers: %zu\n", L);
    ਲਿਖੋ("num_heads: %zu\n", NH);
    ਲਿਖੋ("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_ਵਿੱਚ_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t ਅੰਕ_parameters = 0;
    ਲਈ (size_t i = 0; i < ਅੰਕ_PARAMETER_TENSORS; i++) {
        ਅੰਕ_parameters += model->param_sizes[i];
    }
    ਲਿਖੋ("num_parameters: %zu\n", ਅੰਕ_parameters);
    model->ਅੰਕ_parameters = ਅੰਕ_parameters;

    // read in all the parameters from file
    model->params_ਸਮ੍ਰਿਤੀ = malloc_and_ਬਿੰਦੂ_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_ਸਮ੍ਰਿਤੀ, ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਅੰਕ_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_ਸਮ੍ਰਿਤੀ = ਨਿਰਾ;
    model->grads_ਸਮ੍ਰਿਤੀ = ਨਿਰਾ;
    model->m_ਸਮ੍ਰਿਤੀ = ਨਿਰਾ;
    model->v_ਸਮ੍ਰਿਤੀ = ਨਿਰਾ;
    model->grads_acts_ਸਮ੍ਰਿਤੀ = ਨਿਰਾ;
    model->inputs = ਨਿਰਾ;
    model->targets = ਨਿਰਾ;
    model->batch_ਮਾਪ = 0;
    model->seq_len = 0;
    model->ਔਸਤ_loss = -1.0f; // -1.0f will designate no loss
}

ਖਾਲੀ gpt2_forward(GPT2 *model, ਪੂਰਨ* inputs, ਪੂਰਨ* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    ਜੇ (model->params_ਸਮ੍ਰਿਤੀ == ਨਿਰਾ) {
        ਲਿਖੋ("Error: model was not initialized properly.\n");
        ਬਾਹਰ_ਨਿਕਲੋ(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_ਮਾਪ;
    size_t Vp = model->config.padded_vocab_ਮਾਪ;
    size_t L = model->config.ਅੰਕ_layers;
    size_t NH = model->config.ਅੰਕ_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    ਲਈ(ਪੂਰਨ i = 0; i < B * T; i++) {
        ਪ੍ਰਤਿਜ੍ਞਾ(0 <= inputs[i] && inputs[i] < V);
        ਜੇ (targets != ਨਿਰਾ) {
            ਪ੍ਰਤਿਜ੍ਞਾ(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    ਜੇ(model->acts_ਸਮ੍ਰਿਤੀ == ਨਿਰਾ) {
        // record the current B,T as well
        model->batch_ਮਾਪ = B;
        model->seq_len = T;
        // and now allocate the space
        fill_ਵਿੱਚ_activation_sizes(model->act_sizes, model->config, B, T);
        size_t ਅੰਕ_activations = 0;
        ਲਈ (size_t i = 0; i < ਅੰਕ_ACTIVATION_TENSORS; i++) {
            ਅੰਕ_activations += model->act_sizes[i];
        }
        ਲਿਖੋ("num_activations: %zu\n", ਅੰਕ_activations);
        model->ਅੰਕ_activations = ਅੰਕ_activations;
        model->acts_ਸਮ੍ਰਿਤੀ = malloc_and_ਬਿੰਦੂ_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (ਪੂਰਨ*)mallocCheck(B * T * ਆਕਾਰ(ਪੂਰਨ));
        model->targets = (ਪੂਰਨ*)mallocCheck(B * T * ਆਕਾਰ(ਪੂਰਨ)); // might be unused if we never have targets but it's small
    } ਨਹੀਂਤਾਂ {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        ਜੇ (B != model->batch_ਮਾਪ || T != model->seq_len) {
            ਲਿਖੋ("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_ਮਾਪ, model->seq_len, (ਪੂਰਨ)B, (ਪੂਰਨ)T);
            ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(model->inputs, inputs, B * T * ਆਕਾਰ(ਪੂਰਨ));
    ਜੇ (targets != ਨਿਰਾ) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(model->targets, targets, B * T * ਆਕਾਰ(ਪੂਰਨ));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    ਦਸ਼ਮਲਵ* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    ਲਈ (ਪੂਰਨ l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        ਦਸ਼ਮਲਵ* l_ln1w = params.ln1w + l * C;
        ਦਸ਼ਮਲਵ* l_ln1b = params.ln1b + l * C;
        ਦਸ਼ਮਲਵ* l_qkvw = params.qkvw + l * 3*C * C;
        ਦਸ਼ਮਲਵ* l_qkvb = params.qkvb + l * 3*C;
        ਦਸ਼ਮਲਵ* l_attprojw = params.attprojw + l * C * C;
        ਦਸ਼ਮਲਵ* l_attprojb = params.attprojb + l * C;
        ਦਸ਼ਮਲਵ* l_ln2w = params.ln2w + l * C;
        ਦਸ਼ਮਲਵ* l_ln2b = params.ln2b + l * C;
        ਦਸ਼ਮਲਵ* l_fcw = params.fcw + l * 4*C * C;
        ਦਸ਼ਮਲਵ* l_fcb = params.fcb + l * 4*C;
        ਦਸ਼ਮਲਵ* l_fcprojw = params.fcprojw + l * C * 4*C;
        ਦਸ਼ਮਲਵ* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        ਦਸ਼ਮਲਵ* l_ln1 = acts.ln1 + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_ln1_ਔਸਤ = acts.ln1_ਔਸਤ + l * B * T;
        ਦਸ਼ਮਲਵ* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        ਦਸ਼ਮਲਵ* l_qkv = acts.qkv + l * B * T * 3*C;
        ਦਸ਼ਮਲਵ* l_atty = acts.atty + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_preatt = acts.preatt + l * B * NH * T * T;
        ਦਸ਼ਮਲਵ* l_att = acts.att + l * B * NH * T * T;
        ਦਸ਼ਮਲਵ* l_attproj = acts.attproj + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_residual2 = acts.residual2 + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_ln2 = acts.ln2 + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_ln2_ਔਸਤ = acts.ln2_ਔਸਤ + l * B * T;
        ਦਸ਼ਮਲਵ* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        ਦਸ਼ਮਲਵ* l_fch = acts.fch + l * B * T * 4*C;
        ਦਸ਼ਮਲਵ* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        ਦਸ਼ਮਲਵ* l_fcproj = acts.fcproj + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_ਔਸਤ, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_ਔਸਤ, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_ਔਸਤ, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, ਨਿਰਾ, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    ਜੇ (targets != ਨਿਰਾ) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        ਦਸ਼ਮਲਵ ਔਸਤ_loss = 0.0f;
        ਲਈ (ਪੂਰਨ i=0; i<B*T; i++) { ਔਸਤ_loss += model->acts.losses[i]; }
        ਔਸਤ_loss /= B*T;
        model->ਔਸਤ_loss = ਔਸਤ_loss;
    } ਨਹੀਂਤਾਂ {
        // if we don't have targets, we don't have a loss
        model->ਔਸਤ_loss = -1.0f;
    }
}

ਖਾਲੀ gpt2_zero_grad(GPT2 *model) {
    ਜੇ(model->grads_ਸਮ੍ਰਿਤੀ != ਨਿਰਾ) { memset(model->grads_ਸਮ੍ਰਿਤੀ, 0, model->ਅੰਕ_parameters * ਆਕਾਰ(ਦਸ਼ਮਲਵ)); }
    ਜੇ(model->grads_acts_ਸਮ੍ਰਿਤੀ != ਨਿਰਾ) { memset(model->grads_acts_ਸਮ੍ਰਿਤੀ, 0, model->ਅੰਕ_activations * ਆਕਾਰ(ਦਸ਼ਮਲਵ)); }
}

ਖਾਲੀ gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    ਜੇ (model->ਔਸਤ_loss == -1.0f) {
        ਲਿਖੋ("Error: must forward with targets before backward\n");
        ਬਾਹਰ_ਨਿਕਲੋ(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    ਜੇ (model->grads_ਸਮ੍ਰਿਤੀ == ਨਿਰਾ) {
        model->grads_ਸਮ੍ਰਿਤੀ = malloc_and_ਬਿੰਦੂ_parameters(&model->grads, model->param_sizes);
        model->grads_acts_ਸਮ੍ਰਿਤੀ = malloc_and_ਬਿੰਦੂ_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_ਮਾਪ;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_ਮਾਪ;
    size_t Vp = model->config.padded_vocab_ਮਾਪ;
    size_t L = model->config.ਅੰਕ_layers;
    size_t NH = model->config.ਅੰਕ_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    ਦਸ਼ਮਲਵ dloss_ਔਸਤ = 1.0f / (B*T);
    ਲਈ (ਪੂਰਨ i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_ਔਸਤ; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, ਨਿਰਾ, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    ਦਸ਼ਮਲਵ* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    ਦਸ਼ਮਲਵ* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_ਔਸਤ, acts.lnf_rstd, B, T, C);

    ਲਈ (ਪੂਰਨ l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        ਦਸ਼ਮਲਵ* l_ln1w = params.ln1w + l * C;
        ਦਸ਼ਮਲਵ* l_qkvw = params.qkvw + l * 3*C * C;
        ਦਸ਼ਮਲਵ* l_attprojw = params.attprojw + l * C * C;
        ਦਸ਼ਮਲਵ* l_ln2w = params.ln2w + l * C;
        ਦਸ਼ਮਲਵ* l_fcw = params.fcw + l * 4*C * C;
        ਦਸ਼ਮਲਵ* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        ਦਸ਼ਮਲਵ* dl_ln1w = grads.ln1w + l * C;
        ਦਸ਼ਮਲਵ* dl_ln1b = grads.ln1b + l * C;
        ਦਸ਼ਮਲਵ* dl_qkvw = grads.qkvw + l * 3*C * C;
        ਦਸ਼ਮਲਵ* dl_qkvb = grads.qkvb + l * 3*C;
        ਦਸ਼ਮਲਵ* dl_attprojw = grads.attprojw + l * C * C;
        ਦਸ਼ਮਲਵ* dl_attprojb = grads.attprojb + l * C;
        ਦਸ਼ਮਲਵ* dl_ln2w = grads.ln2w + l * C;
        ਦਸ਼ਮਲਵ* dl_ln2b = grads.ln2b + l * C;
        ਦਸ਼ਮਲਵ* dl_fcw = grads.fcw + l * 4*C * C;
        ਦਸ਼ਮਲਵ* dl_fcb = grads.fcb + l * 4*C;
        ਦਸ਼ਮਲਵ* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        ਦਸ਼ਮਲਵ* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        ਦਸ਼ਮਲਵ* l_ln1 = acts.ln1 + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_ln1_ਔਸਤ = acts.ln1_ਔਸਤ + l * B * T;
        ਦਸ਼ਮਲਵ* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        ਦਸ਼ਮਲਵ* l_qkv = acts.qkv + l * B * T * 3*C;
        ਦਸ਼ਮਲਵ* l_atty = acts.atty + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_att = acts.att + l * B * NH * T * T;
        ਦਸ਼ਮਲਵ* l_residual2 = acts.residual2 + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_ln2 = acts.ln2 + l * B * T * C;
        ਦਸ਼ਮਲਵ* l_ln2_ਔਸਤ = acts.ln2_ਔਸਤ + l * B * T;
        ਦਸ਼ਮਲਵ* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        ਦਸ਼ਮਲਵ* l_fch = acts.fch + l * B * T * 4*C;
        ਦਸ਼ਮਲਵ* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        ਦਸ਼ਮਲਵ* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        ਦਸ਼ਮਲਵ* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        ਦਸ਼ਮਲਵ* dl_atty = grads_acts.atty + l * B * T * C;
        ਦਸ਼ਮਲਵ* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        ਦਸ਼ਮਲਵ* dl_att = grads_acts.att + l * B * NH * T * T;
        ਦਸ਼ਮਲਵ* dl_attproj = grads_acts.attproj + l * B * T * C;
        ਦਸ਼ਮਲਵ* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        ਦਸ਼ਮਲਵ* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        ਦਸ਼ਮਲਵ* dl_fch = grads_acts.fch + l * B * T * 4*C;
        ਦਸ਼ਮਲਵ* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        ਦਸ਼ਮਲਵ* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        ਦਸ਼ਮਲਵ* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_ਔਸਤ, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_ਔਸਤ, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

ਖਾਲੀ gpt2_update(GPT2 *model, ਦਸ਼ਮਲਵ learning_ਦਰ, ਦਸ਼ਮਲਵ beta1, ਦਸ਼ਮਲਵ beta2, ਦਸ਼ਮਲਵ eps, ਦਸ਼ਮਲਵ ਭਾਰ_decay, ਪੂਰਨ t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    ਜੇ (model->m_ਸਮ੍ਰਿਤੀ == ਨਿਰਾ) {
        model->m_ਸਮ੍ਰਿਤੀ = (ਦਸ਼ਮਲਵ*)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(model->ਅੰਕ_parameters, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        model->v_ਸਮ੍ਰਿਤੀ = (ਦਸ਼ਮਲਵ*)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(model->ਅੰਕ_parameters, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }

    ਲਈ (size_t i = 0; i < model->ਅੰਕ_parameters; i++) {
        ਦਸ਼ਮਲਵ param = model->params_ਸਮ੍ਰਿਤੀ[i];
        ਦਸ਼ਮਲਵ grad = model->grads_ਸਮ੍ਰਿਤੀ[i];

        // update the first moment (momentum)
        ਦਸ਼ਮਲਵ m = beta1 * model->m_ਸਮ੍ਰਿਤੀ[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        ਦਸ਼ਮਲਵ v = beta2 * model->v_ਸਮ੍ਰਿਤੀ[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        ਦਸ਼ਮਲਵ m_hat = m / (1.0f - powf(beta1, t));
        ਦਸ਼ਮਲਵ v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_ਸਮ੍ਰਿਤੀ[i] = m;
        model->v_ਸਮ੍ਰਿਤੀ[i] = v;
        model->params_ਸਮ੍ਰਿਤੀ[i] -= learning_ਦਰ * (m_hat / (sqrtf(v_hat) + eps) + ਭਾਰ_decay * param);
    }
}

ਖਾਲੀ gpt2_free(GPT2 *model) {
    ਮੁਕਤ(model->params_ਸਮ੍ਰਿਤੀ);
    ਮੁਕਤ(model->grads_ਸਮ੍ਰਿਤੀ);
    ਮੁਕਤ(model->m_ਸਮ੍ਰਿਤੀ);
    ਮੁਕਤ(model->v_ਸਮ੍ਰਿਤੀ);
    ਮੁਕਤ(model->acts_ਸਮ੍ਰਿਤੀ);
    ਮੁਕਤ(model->grads_acts_ਸਮ੍ਰਿਤੀ);
    ਮੁਕਤ(model->inputs);
    ਮੁਕਤ(model->targets);
}

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ਵਾਪਸ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
ਦਸ਼ਮਲਵ random_f32(uint64_t *state) { // random float32 in [0,1)
    ਵਾਪਸ (random_u32(state) >> 8) / 16777216.0f;
}

ਪੂਰਨ sample_mult(ਦਸ਼ਮਲਵ* probabilities, ਪੂਰਨ n, ਦਸ਼ਮਲਵ coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    ਦਸ਼ਮਲਵ cdf = 0.0f;
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        cdf += probabilities[i];
        ਜੇ (coin < cdf) {
            ਵਾਪਸ i;
        }
    }
    ਵਾਪਸ n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
ਪੂਰਨ ਮੁੱਖ() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    ਅਚਲ ਅੱਖਰ* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    ਅਚਲ ਅੱਖਰ* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    ਅਚਲ ਅੱਖਰ* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    ਅਚਲ ਅੱਖਰ* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    ਅਚਲ ਅੱਖਰ* train_ਟੋਕਨ = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    ਅਚਲ ਅੱਖਰ* val_ਟੋਕਨ = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    ਪੂਰਨ B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    ਪੂਰਨ T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    ਅੰਕੜੇ_Loader train_loader, val_loader;
    dataloader_init(&train_loader, train_ਟੋਕਨ, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_ਟੋਕਨ, B, T, 0, 1, 0);
    ਲਿਖੋ("train dataset num_batches: %zu\n", train_loader.ਅੰਕ_ਟੋਕਨ / (B*T));
    ਲਿਖੋ("val dataset num_batches: %zu\n", val_loader.ਅੰਕ_ਟੋਕਨ / (B*T));
    ਪੂਰਨ val_ਅੰਕ_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    ਪੂਰਨ* gen_ਟੋਕਨ = (ਪੂਰਨ*)mallocCheck(B * T * ਆਕਾਰ(ਪੂਰਨ));
    ਅਚਲ ਪੂਰਨ genT = 64; // number of steps of inference we will do

    // train
    ਬਣਤਰ timespec ਸ਼ੁਰੂ, ਸਿਰਾ;
    ਲਈ (ਪੂਰਨ ਕਦਮ = 0; ਕਦਮ <= 40; ਕਦਮ++) {

        // once in a while estimate the validation loss
        ਜੇ (ਕਦਮ % 10 == 0) {
            ਦਸ਼ਮਲਵ val_loss = 0.0f;
            dataloader_reset(&val_loader);
            ਲਈ (ਪੂਰਨ i = 0; i < val_ਅੰਕ_batches; i++) {
                dataloader_ਅਗਲਾ_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.ਔਸਤ_loss;
            }
            val_loss /= val_ਅੰਕ_batches;
            ਲਿਖੋ("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        ਜੇ (ਕਦਮ > 0 && ਕਦਮ % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            ਲਈ(ਪੂਰਨ i = 0; i < B * T; ++i) {
                gen_ਟੋਕਨ[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            ਲਿਖੋ("generating:\n---\n");
            ਲਈ (ਪੂਰਨ t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_ਟੋਕਨ, ਨਿਰਾ, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                ਦਸ਼ਮਲਵ* probs = model.acts.probs + (t-1) * model.config.padded_vocab_ਮਾਪ;
                ਦਸ਼ਮਲਵ coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                ਪੂਰਨ ਅਗਲਾ_token = sample_mult(probs, model.config.vocab_ਮਾਪ, coin);
                gen_ਟੋਕਨ[t] = ਅਗਲਾ_token;
                // print the generated token, either using the Tokenizer or a fallback
                ਜੇ (tokenizer.init_ok) {
                    ਅਚਲ ਅੱਖਰ* token_str = tokenizer_decode(&tokenizer, ਅਗਲਾ_token);
                    ਸੁਰੱਖਿਅਤ_printf(token_str);
                } ਨਹੀਂਤਾਂ {
                    // fall back to printing the token id
                    ਲਿਖੋ("%d ", ਅਗਲਾ_token);
                }
                fflush(stdout);
            }
            ਲਿਖੋ("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &ਸ਼ੁਰੂ);
        dataloader_ਅਗਲਾ_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, ਕਦਮ+1);
        clock_gettime(CLOCK_MONOTONIC, &ਸਿਰਾ);
        ਦੁੱਗਣਾ ਸਮਾਂ_elapsed_s = (ਸਿਰਾ.tv_sec - ਸ਼ੁਰੂ.tv_sec) + (ਸਿਰਾ.tv_nsec - ਸ਼ੁਰੂ.tv_nsec) / 1e9;
        ਲਿਖੋ("step %d: train loss %f (took %f ms)\n", ਕਦਮ, model.ਔਸਤ_loss, ਸਮਾਂ_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    ਮੁਕਤ(gen_ਟੋਕਨ);
    ਵਾਪਸ 0;
}
#ਜੇ_ਅੰਤ
