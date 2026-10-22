/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <ctype.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <time.h>
#அந்தர்பாவ <string.h>
#அந்தர்பாவ <unistd.h>
#ifdef OMP
#அந்தர்பாவ <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#அந்தர்பாவ "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#அந்தர்பாவ "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#அந்தர்பாவ "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

வெறுமை encoder_forward(தசம* out,
                   முழு* inp, தசம* wte, தசம* wpe,
                   முழு B, முழு T, முழு C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            தசம* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            முழு ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            தசம* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            தசம* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            க்கு (முழு i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

வெறுமை encoder_backward(தசம* dwte, தசம* dwpe,
                      தசம* dout, முழு* inp,
                      முழு B, முழு T, முழு C) {
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            தசம* dout_bt = dout + b * T * C + t * C;
            முழு ix = inp[b * T + t];
            தசம* dwte_ix = dwte + ix * C;
            தசம* dwpe_t = dwpe + t * C;
            க்கு (முழு i = 0; i < C; i++) {
                தசம d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

வெறுமை layernorm_forward(தசம* out, தசம* mean, தசம* rstd,
                       தசம* inp, தசம* weight, தசம* bias,
                       முழு B, முழு T, முழு C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    தசம eps = 1e-5f;
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            தசம* x = inp + b * T * C + t * C;
            // calculate the mean
            தசம m = 0.0f;
            க்கு (முழு i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            தசம v = 0.0f;
            க்கு (முழு i = 0; i < C; i++) {
                தசம xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            தசம s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            தசம* out_bt = out + b * T * C + t * C;
            க்கு (முழு i = 0; i < C; i++) {
                தசம n = (s * (x[i] - m)); // normalize
                தசம o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

வெறுமை layernorm_backward(தசம* dinp, தசம* dweight, தசம* dbias,
                        தசம* dout, தசம* inp, தசம* weight, தசம* mean, தசம* rstd,
                        முழு B, முழு T, முழு C) {
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            தசம* dout_bt = dout + b * T * C + t * C;
            தசம* inp_bt = inp + b * T * C + t * C;
            தசம* dinp_bt = dinp + b * T * C + t * C;
            தசம mean_bt = mean[b * T + t];
            தசம rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            தசம dnorm_mean = 0.0f;
            தசம dnorm_norm_mean = 0.0f;
            க்கு (முழு i = 0; i < C; i++) {
                தசம norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                தசம dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            க்கு (முழு i = 0; i < C; i++) {
                தசம norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                தசம dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                தசம dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

வெறுமை matmul_forward_naive(தசம* out,
                         மாறிலி தசம* inp, மாறிலி தசம* weight, மாறிலி தசம* bias,
                         முழு B, முழு T, முழு C, முழு OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel க்கு collapse(2)
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            முழு bt = b * T + t;
            க்கு (முழு o = 0; o < OC; o++) {
                தசம val = (bias != வெற்று) ? bias[o] : 0.0f;
                க்கு (முழு i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

வெறுமை matmul_forward(தசம* out,
                    மாறிலி தசம* inp, மாறிலி தசம* weight, மாறிலி தசம* bias,
                    முழு B, முழு T, முழு C, முழு OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    மாறிலி முழு LOOP_UNROLL = 8;
    எனில் (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        திரும்பு;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel க்கு
    க்கு (முழு obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        க்கு (முழு o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            தசம result[LOOP_UNROLL];
            // initialize the bias, if it exists
            க்கு (முழு ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != வெற்று) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            க்கு (முழு i = 0; i < C; i++) {
                தசம w = weight[i + o * C];
                க்கு (முழு ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    முழு bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            க்கு (முழு ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                முழு bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

வெறுமை matmul_backward(தசம* dinp, தசம* dweight, தசம* dbias,
                     மாறிலி தசம* dout, மாறிலி தசம* inp, மாறிலி தசம* weight,
                     முழு B, முழு T, முழு C, முழு OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel க்கு collapse(2)
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            மாறிலி தசம* dout_bt = dout + b * T * OC + t * OC;
            தசம* dinp_bt = dinp + b * T * C + t * C;
            க்கு (முழு o = 0; o < OC; o++) {
                மாறிலி தசம* wrow = weight + o*C;
                தசம d = dout_bt[o];
                க்கு (முழு i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel க்கு
    க்கு (முழு o = 0; o < OC; o++) {
        க்கு (முழு b = 0; b < B; b++) {
            க்கு (முழு t = 0; t < T; t++) {
                மாறிலி தசம* dout_bt = dout + b * T * OC + t * OC;
                மாறிலி தசம* inp_bt = inp + b * T * C + t * C;
                தசம* dwrow = dweight + o*C;
                தசம d = dout_bt[o];
                எனில் (dbias != வெற்று) { dbias[o] += d; }
                க்கு (முழு i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

வெறுமை attention_forward(தசம* out, தசம* preatt, தசம* att,
                       தசம* inp,
                       முழு B, முழு T, முழு C, முழு NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    முழு C3 = C*3;
    முழு hs = C / NH; // head size
    தசம scale = 1.0 / sqrtf(hs);

    #pragma omp parallel க்கு collapse(3)
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            க்கு (முழு h = 0; h < NH; h++) {
                தசம* query_t = inp + b * T * C3 + t * C3 + h * hs;
                தசம* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                தசம* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                தசம maxval = -10000.0f; // TODO something better
                க்கு (முழு t2 = 0; t2 <= t; t2++) {
                    தசம* சாவி_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    தசம val = 0.0f;
                    க்கு (முழு i = 0; i < hs; i++) {
                        val += query_t[i] * சாவி_t2[i];
                    }
                    val *= scale;
                    எனில் (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                தசம expsum = 0.0f;
                க்கு (முழு t2 = 0; t2 <= t; t2++) {
                    தசம expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                தசம expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                க்கு (முழு t2 = 0; t2 < T; t2++) {
                    எனில் (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } இல்லையெனில் {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                தசம* out_bth = out + b * T * C + t * C + h * hs;
                க்கு (முழு i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                க்கு (முழு t2 = 0; t2 <= t; t2++) {
                    தசம* மதிப்பு_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    தசம att_btht2 = att_bth[t2];
                    க்கு (முழு i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * மதிப்பு_t2[i];
                    }
                }
            }
        }
    }
}

வெறுமை attention_backward(தசம* dinp, தசம* dpreatt, தசம* datt,
                        தசம* dout, தசம* inp, தசம* att,
                        முழு B, முழு T, முழு C, முழு NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    முழு C3 = C*3;
    முழு hs = C / NH; // head size
    தசம scale = 1.f / sqrtf(hs);

    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            க்கு (முழு h = 0; h < NH; h++) {
                தசம* att_bth = att + b*NH*T*T + h*T*T + t*T;
                தசம* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                தசம* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                தசம* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                தசம* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                தசம* dout_bth = dout + b * T * C + t * C + h * hs;
                க்கு (முழு t2 = 0; t2 <= t; t2++) {
                    தசம* மதிப்பு_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    தசம* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    க்கு (முழு i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += மதிப்பு_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                க்கு (முழு t2 = 0; t2 <= t; t2++) {
                    க்கு (முழு t3 = 0; t3 <= t; t3++) {
                        தசம indicator = t2 == t3 ? 1.0f : 0.0f;
                        தசம local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                க்கு (முழு t2 = 0; t2 <= t; t2++) {
                    தசம* சாவி_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    தசம* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    க்கு (முழு i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += சாவி_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
வெறுமை gelu_forward(தசம* out, தசம* inp, முழு N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    க்கு (முழு i = 0; i < N; i++) {
        தசம x = inp[i];
        தசம cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, தள்ளு)
#எனில் defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
வெறுமை gelu_backward(தசம* dinp, தசம* inp, தசம* dout, முழு N) {
    க்கு (முழு i = 0; i < N; i++) {
        தசம x = inp[i];
        தசம cube = 0.044715f * x * x * x;
        தசம tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        தசம tanh_out = tanhf(tanh_arg);
        தசம coshf_out = coshf(tanh_arg);
        தசம sech_out = 1.0f / (coshf_out * coshf_out);
        தசம local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(எடு)

வெறுமை residual_forward(தசம* out, தசம* inp1, தசம* inp2, முழு N) {
    க்கு (முழு i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

வெறுமை residual_backward(தசம* dinp1, தசம* dinp2, தசம* dout, முழு N) {
    க்கு (முழு i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

வெறுமை softmax_forward(தசம* probs, தசம* logits, முழு B, முழு T, முழு V, முழு Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel க்கு collapse(2)
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            // probs <- softmax(logits)
            தசம* logits_bt = logits + b * T * Vp + t * Vp;
            தசம* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            தசம maxval = -10000.0f; // TODO something better
            க்கு (முழு i = 0; i < V; i++) {
                எனில் (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            தசம sum = 0.0f;
            க்கு (முழு i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            க்கு (முழு i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            க்கு (முழு i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

வெறுமை crossentropy_forward(தசம* losses,
                          தசம* probs, முழு* targets,
                          முழு B, முழு T, முழு Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            // loss = -log(probs[target])
            தசம* probs_bt = probs + b * T * Vp + t * Vp;
            முழு ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

வெறுமை crossentropy_softmax_backward(தசம* dlogits,
                           தசம* dlosses, தசம* probs, முழு* targets,
                           முழு B, முழு T, முழு V, முழு Vp) {
    // backwards through both softmax and crossentropy
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            தசம* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            தசம* probs_bt = probs + b * T * Vp + t * Vp;
            தசம dloss = dlosses[b * T + t];
            முழு ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            க்கு (முழு i = 0; i < V; i++) {
                தசம p = probs_bt[i];
                தசம indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

வகைவரை கட்டமைப்பு {
    முழு max_seq_len; // max sequence length, e.g. 1024
    முழு vocab_size; // vocab size, e.g. 50257
    முழு padded_vocab_size; // padded to e.g. %128==0, 50304
    முழு இலக்கம்_layers; // number of layers, e.g. 12
    முழு இலக்கம்_heads; // number of heads in attention, e.g. 12
    முழு channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define இலக்கம்_PARAMETER_TENSORS 16
வகைவரை கட்டமைப்பு {
    தசம* wte; // (V, C)
    தசம* wpe; // (maxT, C)
    தசம* ln1w; // (L, C)
    தசம* ln1b; // (L, C)
    தசம* qkvw; // (L, 3*C, C)
    தசம* qkvb; // (L, 3*C)
    தசம* attprojw; // (L, C, C)
    தசம* attprojb; // (L, C)
    தசம* ln2w; // (L, C)
    தசம* ln2b; // (L, C)
    தசம* fcw; // (L, 4*C, C)
    தசம* fcb; // (L, 4*C)
    தசம* fcprojw; // (L, C, 4*C)
    தசம* fcprojb; // (L, C)
    தசம* lnfw; // (C)
    தசம* lnfb; // (C)
} ParameterTensors;

வெறுமை fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.இலக்கம்_layers;
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
தசம* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t இலக்கம்_parameters = 0;
    க்கு (size_t i = 0; i < இலக்கம்_PARAMETER_TENSORS; i++) {
        இலக்கம்_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    தசம* params_memory = (தசம*)mallocCheck(இலக்கம்_parameters * அளவு(தசம));
    // assign all the tensors
    தசம** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    தசம* params_memory_iterator = params_memory;
    க்கு (size_t i = 0; i < இலக்கம்_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    திரும்பு params_memory;
}

#define இலக்கம்_ACTIVATION_TENSORS 23
வகைவரை கட்டமைப்பு {
    தசம* encoded; // (B, T, C)
    தசம* ln1; // (L, B, T, C)
    தசம* ln1_mean; // (L, B, T)
    தசம* ln1_rstd; // (L, B, T)
    தசம* qkv; // (L, B, T, 3*C)
    தசம* atty; // (L, B, T, C)
    தசம* preatt; // (L, B, NH, T, T)
    தசம* att; // (L, B, NH, T, T)
    தசம* attproj; // (L, B, T, C)
    தசம* residual2; // (L, B, T, C)
    தசம* ln2; // (L, B, T, C)
    தசம* ln2_mean; // (L, B, T)
    தசம* ln2_rstd; // (L, B, T)
    தசம* fch; // (L, B, T, 4*C)
    தசம* fch_gelu; // (L, B, T, 4*C)
    தசம* fcproj; // (L, B, T, C)
    தசம* residual3; // (L, B, T, C)
    தசம* lnf; // (B, T, C)
    தசம* lnf_mean; // (B, T)
    தசம* lnf_rstd; // (B, T)
    தசம* logits; // (B, T, V)
    தசம* probs; // (B, T, V)
    தசம* losses; // (B, T)
} ActivationTensors;

வெறுமை fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, முழு B, முழு T) {
    size_t C = config.channels;
    size_t NH = config.இலக்கம்_heads;
    size_t L = config.இலக்கம்_layers;
    size_t Vp = config.padded_vocab_size;
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

தசம* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t இலக்கம்_activations = 0;
    க்கு (size_t i = 0; i < இலக்கம்_ACTIVATION_TENSORS; i++) {
        இலக்கம்_activations += act_sizes[i];
    }
    தசம* acts_memory = (தசம*)mallocCheck(இலக்கம்_activations * அளவு(தசம));
    தசம** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    தசம* acts_memory_iterator = acts_memory;
    க்கு (size_t i = 0; i < இலக்கம்_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    திரும்பு acts_memory;
}

வகைவரை கட்டமைப்பு {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[இலக்கம்_PARAMETER_TENSORS];
    தசம* params_memory;
    size_t இலக்கம்_parameters;
    // gradients of the weights
    ParameterTensors grads;
    தசம* grads_memory;
    // buffers for the AdamW optimizer
    தசம* m_memory;
    தசம* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[இலக்கம்_ACTIVATION_TENSORS];
    தசம* acts_memory;
    size_t இலக்கம்_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    தசம* grads_acts_memory;
    // other run state configuration
    முழு batch_size; // the batch size (B) of current forward pass
    முழு seq_len; // the sequence length (T) of current forward pass
    முழு* inputs; // the input tokens for the current forward pass
    முழு* targets; // the target tokens for the current forward pass
    தசம mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

வெறுமை gpt2_build_from_checkpoint(GPT2 *model, மாறிலி எழுத்து* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    முழு model_header[256];
    freadCheck(model_header, அளவு(முழு), 256, model_file);
    எனில் (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    எனில் (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.இலக்கம்_layers = L = model_header[4];
    model->config.இலக்கம்_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t இலக்கம்_parameters = 0;
    க்கு (size_t i = 0; i < இலக்கம்_PARAMETER_TENSORS; i++) {
        இலக்கம்_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", இலக்கம்_parameters);
    model->இலக்கம்_parameters = இலக்கம்_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, அளவு(தசம), இலக்கம்_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = வெற்று;
    model->grads_memory = வெற்று;
    model->m_memory = வெற்று;
    model->v_memory = வெற்று;
    model->grads_acts_memory = வெற்று;
    model->inputs = வெற்று;
    model->targets = வெற்று;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

வெறுமை gpt2_forward(GPT2 *model, முழு* inputs, முழு* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    எனில் (model->params_memory == வெற்று) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.இலக்கம்_layers;
    size_t NH = model->config.இலக்கம்_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    க்கு(முழு i = 0; i < B * T; i++) {
        ப்ரதிஜ்ஞா(0 <= inputs[i] && inputs[i] < V);
        எனில் (targets != வெற்று) {
            ப்ரதிஜ்ஞா(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    எனில்(model->acts_memory == வெற்று) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t இலக்கம்_activations = 0;
        க்கு (size_t i = 0; i < இலக்கம்_ACTIVATION_TENSORS; i++) {
            இலக்கம்_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", இலக்கம்_activations);
        model->இலக்கம்_activations = இலக்கம்_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (முழு*)mallocCheck(B * T * அளவு(முழு));
        model->targets = (முழு*)mallocCheck(B * T * அளவு(முழு)); // might be unused if we never have targets but it's small
    } இல்லையெனில் {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        எனில் (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (முழு)B, (முழு)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * அளவு(முழு));
    எனில் (targets != வெற்று) {
        memcpy(model->targets, targets, B * T * அளவு(முழு));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    தசம* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    க்கு (முழு l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        தசம* l_ln1w = params.ln1w + l * C;
        தசம* l_ln1b = params.ln1b + l * C;
        தசம* l_qkvw = params.qkvw + l * 3*C * C;
        தசம* l_qkvb = params.qkvb + l * 3*C;
        தசம* l_attprojw = params.attprojw + l * C * C;
        தசம* l_attprojb = params.attprojb + l * C;
        தசம* l_ln2w = params.ln2w + l * C;
        தசம* l_ln2b = params.ln2b + l * C;
        தசம* l_fcw = params.fcw + l * 4*C * C;
        தசம* l_fcb = params.fcb + l * 4*C;
        தசம* l_fcprojw = params.fcprojw + l * C * 4*C;
        தசம* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        தசம* l_ln1 = acts.ln1 + l * B * T * C;
        தசம* l_ln1_mean = acts.ln1_mean + l * B * T;
        தசம* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        தசம* l_qkv = acts.qkv + l * B * T * 3*C;
        தசம* l_atty = acts.atty + l * B * T * C;
        தசம* l_preatt = acts.preatt + l * B * NH * T * T;
        தசம* l_att = acts.att + l * B * NH * T * T;
        தசம* l_attproj = acts.attproj + l * B * T * C;
        தசம* l_residual2 = acts.residual2 + l * B * T * C;
        தசம* l_ln2 = acts.ln2 + l * B * T * C;
        தசம* l_ln2_mean = acts.ln2_mean + l * B * T;
        தசம* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        தசம* l_fch = acts.fch + l * B * T * 4*C;
        தசம* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        தசம* l_fcproj = acts.fcproj + l * B * T * C;
        தசம* l_residual3 = acts.residual3 + l * B * T * C;

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
    matmul_forward(acts.logits, acts.lnf, params.wte, வெற்று, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    எனில் (targets != வெற்று) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        தசம mean_loss = 0.0f;
        க்கு (முழு i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } இல்லையெனில் {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

வெறுமை gpt2_zero_grad(GPT2 *model) {
    எனில்(model->grads_memory != வெற்று) { memset(model->grads_memory, 0, model->இலக்கம்_parameters * அளவு(தசம)); }
    எனில்(model->grads_acts_memory != வெற்று) { memset(model->grads_acts_memory, 0, model->இலக்கம்_activations * அளவு(தசம)); }
}

வெறுமை gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    எனில் (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    எனில் (model->grads_memory == வெற்று) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.இலக்கம்_layers;
    size_t NH = model->config.இலக்கம்_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    தசம dloss_mean = 1.0f / (B*T);
    க்கு (முழு i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, வெற்று, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    தசம* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    தசம* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    க்கு (முழு l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        தசம* l_ln1w = params.ln1w + l * C;
        தசம* l_qkvw = params.qkvw + l * 3*C * C;
        தசம* l_attprojw = params.attprojw + l * C * C;
        தசம* l_ln2w = params.ln2w + l * C;
        தசம* l_fcw = params.fcw + l * 4*C * C;
        தசம* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        தசம* dl_ln1w = grads.ln1w + l * C;
        தசம* dl_ln1b = grads.ln1b + l * C;
        தசம* dl_qkvw = grads.qkvw + l * 3*C * C;
        தசம* dl_qkvb = grads.qkvb + l * 3*C;
        தசம* dl_attprojw = grads.attprojw + l * C * C;
        தசம* dl_attprojb = grads.attprojb + l * C;
        தசம* dl_ln2w = grads.ln2w + l * C;
        தசம* dl_ln2b = grads.ln2b + l * C;
        தசம* dl_fcw = grads.fcw + l * 4*C * C;
        தசம* dl_fcb = grads.fcb + l * 4*C;
        தசம* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        தசம* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        தசம* l_ln1 = acts.ln1 + l * B * T * C;
        தசம* l_ln1_mean = acts.ln1_mean + l * B * T;
        தசம* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        தசம* l_qkv = acts.qkv + l * B * T * 3*C;
        தசம* l_atty = acts.atty + l * B * T * C;
        தசம* l_att = acts.att + l * B * NH * T * T;
        தசம* l_residual2 = acts.residual2 + l * B * T * C;
        தசம* l_ln2 = acts.ln2 + l * B * T * C;
        தசம* l_ln2_mean = acts.ln2_mean + l * B * T;
        தசம* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        தசம* l_fch = acts.fch + l * B * T * 4*C;
        தசம* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        தசம* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        தசம* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        தசம* dl_atty = grads_acts.atty + l * B * T * C;
        தசம* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        தசம* dl_att = grads_acts.att + l * B * NH * T * T;
        தசம* dl_attproj = grads_acts.attproj + l * B * T * C;
        தசம* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        தசம* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        தசம* dl_fch = grads_acts.fch + l * B * T * 4*C;
        தசம* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        தசம* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        தசம* dl_residual3 = grads_acts.residual3 + l * B * T * C;

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

வெறுமை gpt2_update(GPT2 *model, தசம learning_rate, தசம beta1, தசம beta2, தசம eps, தசம weight_decay, முழு t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    எனில் (model->m_memory == வெற்று) {
        model->m_memory = (தசம*)calloc(model->இலக்கம்_parameters, அளவு(தசம));
        model->v_memory = (தசம*)calloc(model->இலக்கம்_parameters, அளவு(தசம));
    }

    க்கு (size_t i = 0; i < model->இலக்கம்_parameters; i++) {
        தசம param = model->params_memory[i];
        தசம grad = model->grads_memory[i];

        // update the first moment (momentum)
        தசம m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        தசம v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        தசம m_hat = m / (1.0f - powf(beta1, t));
        தசம v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

வெறுமை gpt2_free(GPT2 *model) {
    விடுவி(model->params_memory);
    விடுவி(model->grads_memory);
    விடுவி(model->m_memory);
    விடுவி(model->v_memory);
    விடுவி(model->acts_memory);
    விடுவி(model->grads_acts_memory);
    விடுவி(model->inputs);
    விடுவி(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned முழு random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    திரும்பு (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
தசம random_f32(uint64_t *state) { // random float32 in [0,1)
    திரும்பு (random_u32(state) >> 8) / 16777216.0f;
}

முழு sample_mult(தசம* probabilities, முழு n, தசம coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    தசம cdf = 0.0f;
    க்கு (முழு i = 0; i < n; i++) {
        cdf += probabilities[i];
        எனில் (coin < cdf) {
            திரும்பு i;
        }
    }
    திரும்பு n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
முழு முதன்மை() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    மாறிலி எழுத்து* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    மாறிலி எழுத்து* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    மாறிலி எழுத்து* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    மாறிலி எழுத்து* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    மாறிலி எழுத்து* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    மாறிலி எழுத்து* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    முழு B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    முழு T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.இலக்கம்_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.இலக்கம்_tokens / (B*T));
    முழு val_இலக்கம்_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    முழு* gen_tokens = (முழு*)mallocCheck(B * T * அளவு(முழு));
    மாறிலி முழு genT = 64; // number of steps of inference we will do

    // train
    கட்டமைப்பு timespec start, end;
    க்கு (முழு step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        எனில் (step % 10 == 0) {
            தசம val_loss = 0.0f;
            dataloader_reset(&val_loader);
            க்கு (முழு i = 0; i < val_இலக்கம்_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_இலக்கம்_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        எனில் (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            க்கு(முழு i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            க்கு (முழு t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, வெற்று, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                தசம* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                தசம coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                முழு next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                எனில் (tokenizer.init_ok) {
                    மாறிலி எழுத்து* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } இல்லையெனில் {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        இரட்டை time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    விடுவி(gen_tokens);
    திரும்பு 0;
}
#endif
