/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <ctype.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <unistd.h>
#ifdef OMP
#अन्तर्भाव <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#अन्तर्भाव "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#अन्तर्भाव "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#अन्तर्भाव "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

शून्य encoder_forward(दशांश* out,
                   पूर्णांक* inp, दशांश* wte, दशांश* wpe,
                   पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            दशांश* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            पूर्णांक ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            दशांश* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            दशांश* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            साठी (पूर्णांक i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

शून्य encoder_backward(दशांश* dwte, दशांश* dwpe,
                      दशांश* dout, पूर्णांक* inp,
                      पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            दशांश* dout_bt = dout + b * T * C + t * C;
            पूर्णांक ix = inp[b * T + t];
            दशांश* dwte_ix = dwte + ix * C;
            दशांश* dwpe_t = dwpe + t * C;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

शून्य layernorm_forward(दशांश* out, दशांश* mean, दशांश* rstd,
                       दशांश* inp, दशांश* weight, दशांश* bias,
                       पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    दशांश eps = 1e-5f;
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            दशांश* x = inp + b * T * C + t * C;
            // calculate the mean
            दशांश m = 0.0f;
            साठी (पूर्णांक i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            दशांश v = 0.0f;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            दशांश s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            दशांश* out_bt = out + b * T * C + t * C;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश n = (s * (x[i] - m)); // normalize
                दशांश o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

शून्य layernorm_backward(दशांश* dinp, दशांश* dweight, दशांश* dbias,
                        दशांश* dout, दशांश* inp, दशांश* weight, दशांश* mean, दशांश* rstd,
                        पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            दशांश* dout_bt = dout + b * T * C + t * C;
            दशांश* inp_bt = inp + b * T * C + t * C;
            दशांश* dinp_bt = dinp + b * T * C + t * C;
            दशांश mean_bt = mean[b * T + t];
            दशांश rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            दशांश dnorm_mean = 0.0f;
            दशांश dnorm_norm_mean = 0.0f;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                दशांश dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                दशांश dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                दशांश dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

शून्य matmul_forward_naive(दशांश* out,
                         स्थिरांक दशांश* inp, स्थिरांक दशांश* weight, स्थिरांक दशांश* bias,
                         पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel साठी collapse(2)
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            पूर्णांक bt = b * T + t;
            साठी (पूर्णांक o = 0; o < OC; o++) {
                दशांश val = (bias != रिक्त) ? bias[o] : 0.0f;
                साठी (पूर्णांक i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

शून्य matmul_forward(दशांश* out,
                    स्थिरांक दशांश* inp, स्थिरांक दशांश* weight, स्थिरांक दशांश* bias,
                    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    स्थिरांक पूर्णांक LOOP_UNROLL = 8;
    जर (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        परत;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel साठी
    साठी (पूर्णांक obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        साठी (पूर्णांक o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            दशांश result[LOOP_UNROLL];
            // initialize the bias, if it exists
            साठी (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != रिक्त) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश w = weight[i + o * C];
                साठी (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    पूर्णांक bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            साठी (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                पूर्णांक bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

शून्य matmul_backward(दशांश* dinp, दशांश* dweight, दशांश* dbias,
                     स्थिरांक दशांश* dout, स्थिरांक दशांश* inp, स्थिरांक दशांश* weight,
                     पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel साठी collapse(2)
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            स्थिरांक दशांश* dout_bt = dout + b * T * OC + t * OC;
            दशांश* dinp_bt = dinp + b * T * C + t * C;
            साठी (पूर्णांक o = 0; o < OC; o++) {
                स्थिरांक दशांश* wrow = weight + o*C;
                दशांश d = dout_bt[o];
                साठी (पूर्णांक i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel साठी
    साठी (पूर्णांक o = 0; o < OC; o++) {
        साठी (पूर्णांक b = 0; b < B; b++) {
            साठी (पूर्णांक t = 0; t < T; t++) {
                स्थिरांक दशांश* dout_bt = dout + b * T * OC + t * OC;
                स्थिरांक दशांश* inp_bt = inp + b * T * C + t * C;
                दशांश* dwrow = dweight + o*C;
                दशांश d = dout_bt[o];
                जर (dbias != रिक्त) { dbias[o] += d; }
                साठी (पूर्णांक i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

शून्य attention_forward(दशांश* out, दशांश* preatt, दशांश* att,
                       दशांश* inp,
                       पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    पूर्णांक C3 = C*3;
    पूर्णांक hs = C / NH; // head size
    दशांश scale = 1.0 / sqrtf(hs);

    #pragma omp parallel साठी collapse(3)
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            साठी (पूर्णांक h = 0; h < NH; h++) {
                दशांश* query_t = inp + b * T * C3 + t * C3 + h * hs;
                दशांश* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                दशांश* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                दशांश maxval = -10000.0f; // TODO something better
                साठी (पूर्णांक t2 = 0; t2 <= t; t2++) {
                    दशांश* किल्ली_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    दशांश val = 0.0f;
                    साठी (पूर्णांक i = 0; i < hs; i++) {
                        val += query_t[i] * किल्ली_t2[i];
                    }
                    val *= scale;
                    जर (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                दशांश expsum = 0.0f;
                साठी (पूर्णांक t2 = 0; t2 <= t; t2++) {
                    दशांश expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                दशांश expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                साठी (पूर्णांक t2 = 0; t2 < T; t2++) {
                    जर (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } अन्यथा {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                दशांश* out_bth = out + b * T * C + t * C + h * hs;
                साठी (पूर्णांक i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                साठी (पूर्णांक t2 = 0; t2 <= t; t2++) {
                    दशांश* मूल्य_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    दशांश att_btht2 = att_bth[t2];
                    साठी (पूर्णांक i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * मूल्य_t2[i];
                    }
                }
            }
        }
    }
}

शून्य attention_backward(दशांश* dinp, दशांश* dpreatt, दशांश* datt,
                        दशांश* dout, दशांश* inp, दशांश* att,
                        पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    पूर्णांक C3 = C*3;
    पूर्णांक hs = C / NH; // head size
    दशांश scale = 1.f / sqrtf(hs);

    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            साठी (पूर्णांक h = 0; h < NH; h++) {
                दशांश* att_bth = att + b*NH*T*T + h*T*T + t*T;
                दशांश* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                दशांश* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                दशांश* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                दशांश* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                दशांश* dout_bth = dout + b * T * C + t * C + h * hs;
                साठी (पूर्णांक t2 = 0; t2 <= t; t2++) {
                    दशांश* मूल्य_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    दशांश* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    साठी (पूर्णांक i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += मूल्य_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                साठी (पूर्णांक t2 = 0; t2 <= t; t2++) {
                    साठी (पूर्णांक t3 = 0; t3 <= t; t3++) {
                        दशांश indicator = t2 == t3 ? 1.0f : 0.0f;
                        दशांश local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                साठी (पूर्णांक t2 = 0; t2 <= t; t2++) {
                    दशांश* किल्ली_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    दशांश* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    साठी (पूर्णांक i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += किल्ली_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
शून्य gelu_forward(दशांश* out, दशांश* inp, पूर्णांक N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    साठी (पूर्णांक i = 0; i < N; i++) {
        दशांश x = inp[i];
        दशांश cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, ढकला)
#जर defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
शून्य gelu_backward(दशांश* dinp, दशांश* inp, दशांश* dout, पूर्णांक N) {
    साठी (पूर्णांक i = 0; i < N; i++) {
        दशांश x = inp[i];
        दशांश cube = 0.044715f * x * x * x;
        दशांश tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        दशांश tanh_out = tanhf(tanh_arg);
        दशांश coshf_out = coshf(tanh_arg);
        दशांश sech_out = 1.0f / (coshf_out * coshf_out);
        दशांश local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(काढा)

शून्य residual_forward(दशांश* out, दशांश* inp1, दशांश* inp2, पूर्णांक N) {
    साठी (पूर्णांक i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

शून्य residual_backward(दशांश* dinp1, दशांश* dinp2, दशांश* dout, पूर्णांक N) {
    साठी (पूर्णांक i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

शून्य softmax_forward(दशांश* probs, दशांश* logits, पूर्णांक B, पूर्णांक T, पूर्णांक V, पूर्णांक Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel साठी collapse(2)
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            // probs <- softmax(logits)
            दशांश* logits_bt = logits + b * T * Vp + t * Vp;
            दशांश* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            दशांश maxval = -10000.0f; // TODO something better
            साठी (पूर्णांक i = 0; i < V; i++) {
                जर (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            दशांश sum = 0.0f;
            साठी (पूर्णांक i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            साठी (पूर्णांक i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            साठी (पूर्णांक i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

शून्य crossentropy_forward(दशांश* losses,
                          दशांश* probs, पूर्णांक* targets,
                          पूर्णांक B, पूर्णांक T, पूर्णांक Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            // loss = -log(probs[target])
            दशांश* probs_bt = probs + b * T * Vp + t * Vp;
            पूर्णांक ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

शून्य crossentropy_softmax_backward(दशांश* dlogits,
                           दशांश* dlosses, दशांश* probs, पूर्णांक* targets,
                           पूर्णांक B, पूर्णांक T, पूर्णांक V, पूर्णांक Vp) {
    // backwards through both softmax and crossentropy
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            दशांश* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            दशांश* probs_bt = probs + b * T * Vp + t * Vp;
            दशांश dloss = dlosses[b * T + t];
            पूर्णांक ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            साठी (पूर्णांक i = 0; i < V; i++) {
                दशांश p = probs_bt[i];
                दशांश indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

प्रकारव्याख्या संरचना {
    पूर्णांक max_seq_len; // max sequence length, e.g. 1024
    पूर्णांक vocab_size; // vocab size, e.g. 50257
    पूर्णांक padded_vocab_size; // padded to e.g. %128==0, 50304
    पूर्णांक अंक_layers; // number of layers, e.g. 12
    पूर्णांक अंक_heads; // number of heads in attention, e.g. 12
    पूर्णांक channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define अंक_PARAMETER_TENSORS 16
प्रकारव्याख्या संरचना {
    दशांश* wte; // (V, C)
    दशांश* wpe; // (maxT, C)
    दशांश* ln1w; // (L, C)
    दशांश* ln1b; // (L, C)
    दशांश* qkvw; // (L, 3*C, C)
    दशांश* qkvb; // (L, 3*C)
    दशांश* attprojw; // (L, C, C)
    दशांश* attprojb; // (L, C)
    दशांश* ln2w; // (L, C)
    दशांश* ln2b; // (L, C)
    दशांश* fcw; // (L, 4*C, C)
    दशांश* fcb; // (L, 4*C)
    दशांश* fcprojw; // (L, C, 4*C)
    दशांश* fcprojb; // (L, C)
    दशांश* lnfw; // (C)
    दशांश* lnfb; // (C)
} ParameterTensors;

शून्य fill_in_parameter_sizes(माप_प्रकार* param_sizes, GPT2Config config) {
    माप_प्रकार Vp = config.padded_vocab_size;
    माप_प्रकार C = config.channels;
    माप_प्रकार maxT = config.max_seq_len;
    माप_प्रकार L = config.अंक_layers;
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
दशांश* malloc_and_point_parameters(ParameterTensors* params, माप_प्रकार* param_sizes) {
    माप_प्रकार अंक_parameters = 0;
    साठी (माप_प्रकार i = 0; i < अंक_PARAMETER_TENSORS; i++) {
        अंक_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    दशांश* params_memory = (दशांश*)mallocCheck(अंक_parameters * आकार(दशांश));
    // assign all the tensors
    दशांश** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    दशांश* params_memory_iterator = params_memory;
    साठी (माप_प्रकार i = 0; i < अंक_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    परत params_memory;
}

#define अंक_ACTIVATION_TENSORS 23
प्रकारव्याख्या संरचना {
    दशांश* encoded; // (B, T, C)
    दशांश* ln1; // (L, B, T, C)
    दशांश* ln1_mean; // (L, B, T)
    दशांश* ln1_rstd; // (L, B, T)
    दशांश* qkv; // (L, B, T, 3*C)
    दशांश* atty; // (L, B, T, C)
    दशांश* preatt; // (L, B, NH, T, T)
    दशांश* att; // (L, B, NH, T, T)
    दशांश* attproj; // (L, B, T, C)
    दशांश* residual2; // (L, B, T, C)
    दशांश* ln2; // (L, B, T, C)
    दशांश* ln2_mean; // (L, B, T)
    दशांश* ln2_rstd; // (L, B, T)
    दशांश* fch; // (L, B, T, 4*C)
    दशांश* fch_gelu; // (L, B, T, 4*C)
    दशांश* fcproj; // (L, B, T, C)
    दशांश* residual3; // (L, B, T, C)
    दशांश* lnf; // (B, T, C)
    दशांश* lnf_mean; // (B, T)
    दशांश* lnf_rstd; // (B, T)
    दशांश* logits; // (B, T, V)
    दशांश* probs; // (B, T, V)
    दशांश* losses; // (B, T)
} ActivationTensors;

शून्य fill_in_activation_sizes(माप_प्रकार* act_sizes, GPT2Config config, पूर्णांक B, पूर्णांक T) {
    माप_प्रकार C = config.channels;
    माप_प्रकार NH = config.अंक_heads;
    माप_प्रकार L = config.अंक_layers;
    माप_प्रकार Vp = config.padded_vocab_size;
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

दशांश* malloc_and_point_activations(ActivationTensors* acts, माप_प्रकार* act_sizes) {
    माप_प्रकार अंक_activations = 0;
    साठी (माप_प्रकार i = 0; i < अंक_ACTIVATION_TENSORS; i++) {
        अंक_activations += act_sizes[i];
    }
    दशांश* acts_memory = (दशांश*)mallocCheck(अंक_activations * आकार(दशांश));
    दशांश** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    दशांश* acts_memory_iterator = acts_memory;
    साठी (माप_प्रकार i = 0; i < अंक_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    परत acts_memory;
}

प्रकारव्याख्या संरचना {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    माप_प्रकार param_sizes[अंक_PARAMETER_TENSORS];
    दशांश* params_memory;
    माप_प्रकार अंक_parameters;
    // gradients of the weights
    ParameterTensors grads;
    दशांश* grads_memory;
    // buffers for the AdamW optimizer
    दशांश* m_memory;
    दशांश* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    माप_प्रकार act_sizes[अंक_ACTIVATION_TENSORS];
    दशांश* acts_memory;
    माप_प्रकार अंक_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    दशांश* grads_acts_memory;
    // other run state configuration
    पूर्णांक batch_size; // the batch size (B) of current forward pass
    पूर्णांक seq_len; // the sequence length (T) of current forward pass
    पूर्णांक* inputs; // the input tokens for the current forward pass
    पूर्णांक* targets; // the target tokens for the current forward pass
    दशांश mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

शून्य gpt2_build_from_checkpoint(GPT2 *model, स्थिरांक अक्षर* checkpoint_path) {

    // read in model from a checkpoint file
    खाता *model_file = fopenCheck(checkpoint_path, "rb");
    पूर्णांक model_header[256];
    freadCheck(model_header, आकार(पूर्णांक), 256, model_file);
    जर (model_header[0] != 20240326) { म_लिखो("Bad magic model file\n"); निकास(1); }
    जर (model_header[1] != 3) {
        म_लिखो("Bad version in model file\n");
        म_लिखो("---> HINT: try to re-run `python train_gpt2.py`\n");
        निकास(1);
    }

    // read in hyperparameters
    माप_प्रकार maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.अंक_layers = L = model_header[4];
    model->config.अंक_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    म_लिखो("[GPT-2]\n");
    म_लिखो("max_seq_len: %zu\n", maxT);
    म_लिखो("vocab_size: %zu\n", V);
    म_लिखो("padded_vocab_size: %zu\n", Vp);
    म_लिखो("num_layers: %zu\n", L);
    म_लिखो("num_heads: %zu\n", NH);
    म_लिखो("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    माप_प्रकार अंक_parameters = 0;
    साठी (माप_प्रकार i = 0; i < अंक_PARAMETER_TENSORS; i++) {
        अंक_parameters += model->param_sizes[i];
    }
    म_लिखो("num_parameters: %zu\n", अंक_parameters);
    model->अंक_parameters = अंक_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, आकार(दशांश), अंक_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = रिक्त;
    model->grads_memory = रिक्त;
    model->m_memory = रिक्त;
    model->v_memory = रिक्त;
    model->grads_acts_memory = रिक्त;
    model->inputs = रिक्त;
    model->targets = रिक्त;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

शून्य gpt2_forward(GPT2 *model, पूर्णांक* inputs, पूर्णांक* targets, माप_प्रकार B, माप_प्रकार T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    जर (model->params_memory == रिक्त) {
        म_लिखो("Error: model was not initialized properly.\n");
        निकास(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    माप_प्रकार V = model->config.vocab_size;
    माप_प्रकार Vp = model->config.padded_vocab_size;
    माप_प्रकार L = model->config.अंक_layers;
    माप_प्रकार NH = model->config.अंक_heads;
    माप_प्रकार C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    साठी(पूर्णांक i = 0; i < B * T; i++) {
        प्रतिज्ञा(0 <= inputs[i] && inputs[i] < V);
        जर (targets != रिक्त) {
            प्रतिज्ञा(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    जर(model->acts_memory == रिक्त) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        माप_प्रकार अंक_activations = 0;
        साठी (माप_प्रकार i = 0; i < अंक_ACTIVATION_TENSORS; i++) {
            अंक_activations += model->act_sizes[i];
        }
        म_लिखो("num_activations: %zu\n", अंक_activations);
        model->अंक_activations = अंक_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (पूर्णांक*)mallocCheck(B * T * आकार(पूर्णांक));
        model->targets = (पूर्णांक*)mallocCheck(B * T * आकार(पूर्णांक)); // might be unused if we never have targets but it's small
    } अन्यथा {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        जर (B != model->batch_size || T != model->seq_len) {
            म_लिखो("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (पूर्णांक)B, (पूर्णांक)T);
            निकास(निकास_त्रुटि);
        }
    }

    // cache the inputs/targets
    स_नकल(model->inputs, inputs, B * T * आकार(पूर्णांक));
    जर (targets != रिक्त) {
        स_नकल(model->targets, targets, B * T * आकार(पूर्णांक));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    दशांश* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    साठी (पूर्णांक l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        दशांश* l_ln1w = params.ln1w + l * C;
        दशांश* l_ln1b = params.ln1b + l * C;
        दशांश* l_qkvw = params.qkvw + l * 3*C * C;
        दशांश* l_qkvb = params.qkvb + l * 3*C;
        दशांश* l_attprojw = params.attprojw + l * C * C;
        दशांश* l_attprojb = params.attprojb + l * C;
        दशांश* l_ln2w = params.ln2w + l * C;
        दशांश* l_ln2b = params.ln2b + l * C;
        दशांश* l_fcw = params.fcw + l * 4*C * C;
        दशांश* l_fcb = params.fcb + l * 4*C;
        दशांश* l_fcprojw = params.fcprojw + l * C * 4*C;
        दशांश* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        दशांश* l_ln1 = acts.ln1 + l * B * T * C;
        दशांश* l_ln1_mean = acts.ln1_mean + l * B * T;
        दशांश* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        दशांश* l_qkv = acts.qkv + l * B * T * 3*C;
        दशांश* l_atty = acts.atty + l * B * T * C;
        दशांश* l_preatt = acts.preatt + l * B * NH * T * T;
        दशांश* l_att = acts.att + l * B * NH * T * T;
        दशांश* l_attproj = acts.attproj + l * B * T * C;
        दशांश* l_residual2 = acts.residual2 + l * B * T * C;
        दशांश* l_ln2 = acts.ln2 + l * B * T * C;
        दशांश* l_ln2_mean = acts.ln2_mean + l * B * T;
        दशांश* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        दशांश* l_fch = acts.fch + l * B * T * 4*C;
        दशांश* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        दशांश* l_fcproj = acts.fcproj + l * B * T * C;
        दशांश* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, रिक्त, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    जर (targets != रिक्त) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        दशांश mean_loss = 0.0f;
        साठी (पूर्णांक i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } अन्यथा {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

शून्य gpt2_zero_grad(GPT2 *model) {
    जर(model->grads_memory != रिक्त) { स_रखो(model->grads_memory, 0, model->अंक_parameters * आकार(दशांश)); }
    जर(model->grads_acts_memory != रिक्त) { स_रखो(model->grads_acts_memory, 0, model->अंक_activations * आकार(दशांश)); }
}

शून्य gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    जर (model->mean_loss == -1.0f) {
        म_लिखो("Error: must forward with targets before backward\n");
        निकास(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    जर (model->grads_memory == रिक्त) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    माप_प्रकार B = model->batch_size;
    माप_प्रकार T = model->seq_len;
    माप_प्रकार V = model->config.vocab_size;
    माप_प्रकार Vp = model->config.padded_vocab_size;
    माप_प्रकार L = model->config.अंक_layers;
    माप_प्रकार NH = model->config.अंक_heads;
    माप_प्रकार C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    दशांश dloss_mean = 1.0f / (B*T);
    साठी (पूर्णांक i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, रिक्त, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    दशांश* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    दशांश* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    साठी (पूर्णांक l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        दशांश* l_ln1w = params.ln1w + l * C;
        दशांश* l_qkvw = params.qkvw + l * 3*C * C;
        दशांश* l_attprojw = params.attprojw + l * C * C;
        दशांश* l_ln2w = params.ln2w + l * C;
        दशांश* l_fcw = params.fcw + l * 4*C * C;
        दशांश* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        दशांश* dl_ln1w = grads.ln1w + l * C;
        दशांश* dl_ln1b = grads.ln1b + l * C;
        दशांश* dl_qkvw = grads.qkvw + l * 3*C * C;
        दशांश* dl_qkvb = grads.qkvb + l * 3*C;
        दशांश* dl_attprojw = grads.attprojw + l * C * C;
        दशांश* dl_attprojb = grads.attprojb + l * C;
        दशांश* dl_ln2w = grads.ln2w + l * C;
        दशांश* dl_ln2b = grads.ln2b + l * C;
        दशांश* dl_fcw = grads.fcw + l * 4*C * C;
        दशांश* dl_fcb = grads.fcb + l * 4*C;
        दशांश* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        दशांश* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        दशांश* l_ln1 = acts.ln1 + l * B * T * C;
        दशांश* l_ln1_mean = acts.ln1_mean + l * B * T;
        दशांश* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        दशांश* l_qkv = acts.qkv + l * B * T * 3*C;
        दशांश* l_atty = acts.atty + l * B * T * C;
        दशांश* l_att = acts.att + l * B * NH * T * T;
        दशांश* l_residual2 = acts.residual2 + l * B * T * C;
        दशांश* l_ln2 = acts.ln2 + l * B * T * C;
        दशांश* l_ln2_mean = acts.ln2_mean + l * B * T;
        दशांश* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        दशांश* l_fch = acts.fch + l * B * T * 4*C;
        दशांश* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        दशांश* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        दशांश* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        दशांश* dl_atty = grads_acts.atty + l * B * T * C;
        दशांश* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        दशांश* dl_att = grads_acts.att + l * B * NH * T * T;
        दशांश* dl_attproj = grads_acts.attproj + l * B * T * C;
        दशांश* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        दशांश* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        दशांश* dl_fch = grads_acts.fch + l * B * T * 4*C;
        दशांश* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        दशांश* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        दशांश* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

शून्य gpt2_update(GPT2 *model, दशांश learning_rate, दशांश beta1, दशांश beta2, दशांश eps, दशांश weight_decay, पूर्णांक t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    जर (model->m_memory == रिक्त) {
        model->m_memory = (दशांश*)सुस्मृति(model->अंक_parameters, आकार(दशांश));
        model->v_memory = (दशांश*)सुस्मृति(model->अंक_parameters, आकार(दशांश));
    }

    साठी (माप_प्रकार i = 0; i < model->अंक_parameters; i++) {
        दशांश param = model->params_memory[i];
        दशांश grad = model->grads_memory[i];

        // update the first moment (momentum)
        दशांश m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        दशांश v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        दशांश m_hat = m / (1.0f - powf(beta1, t));
        दशांश v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

शून्य gpt2_free(GPT2 *model) {
    मुक्त(model->params_memory);
    मुक्त(model->grads_memory);
    मुक्त(model->m_memory);
    मुक्त(model->v_memory);
    मुक्त(model->acts_memory);
    मुक्त(model->grads_acts_memory);
    मुक्त(model->inputs);
    मुक्त(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

अचिन्हित पूर्णांक random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    परत (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
दशांश random_f32(uint64_t *state) { // random float32 in [0,1)
    परत (random_u32(state) >> 8) / 16777216.0f;
}

पूर्णांक sample_mult(दशांश* probabilities, पूर्णांक n, दशांश coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    दशांश cdf = 0.0f;
    साठी (पूर्णांक i = 0; i < n; i++) {
        cdf += probabilities[i];
        जर (coin < cdf) {
            परत i;
        }
    }
    परत n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
पूर्णांक main() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    स्थिरांक अक्षर* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    स्थिरांक अक्षर* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    स्थिरांक अक्षर* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    स्थिरांक अक्षर* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    स्थिरांक अक्षर* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    स्थिरांक अक्षर* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    पूर्णांक B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    पूर्णांक T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    म_लिखो("train dataset num_batches: %zu\n", train_loader.अंक_tokens / (B*T));
    म_लिखो("val dataset num_batches: %zu\n", val_loader.अंक_tokens / (B*T));
    पूर्णांक val_अंक_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    पूर्णांक* gen_tokens = (पूर्णांक*)mallocCheck(B * T * आकार(पूर्णांक));
    स्थिरांक पूर्णांक genT = 64; // number of steps of inference we will do

    // train
    संरचना timespec start, end;
    साठी (पूर्णांक step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        जर (step % 10 == 0) {
            दशांश val_loss = 0.0f;
            dataloader_reset(&val_loader);
            साठी (पूर्णांक i = 0; i < val_अंक_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_अंक_batches;
            म_लिखो("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        जर (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            साठी(पूर्णांक i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            म_लिखो("generating:\n---\n");
            साठी (पूर्णांक t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, रिक्त, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                दशांश* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                दशांश coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                पूर्णांक next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                जर (tokenizer.init_ok) {
                    स्थिरांक अक्षर* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } अन्यथा {
                    // fall back to printing the token id
                    म_लिखो("%d ", next_token);
                }
                ख_साफ(मानक_निकास);
            }
            म_लिखो("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुण time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        म_लिखो("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    मुक्त(gen_tokens);
    परत 0;
}
#endif
