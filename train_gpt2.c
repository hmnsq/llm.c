/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <ctype.h>
#شامل <stdint.h>
#شامل <assert.h>
#شامل <math.h>
#شامل <time.h>
#شامل <string.h>
#شامل <unistd.h>
#ifdef OMP
#شامل <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#شامل "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#شامل "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#شامل "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

تهی encoder_forward(اعشاری* out,
                   صحیح* inp, اعشاری* wte, اعشاری* wpe,
                   صحیح B, صحیح T, صحیح C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            اعشاری* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            صحیح ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            اعشاری* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            اعشاری* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            برای (صحیح i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

تهی encoder_backward(اعشاری* dwte, اعشاری* dwpe,
                      اعشاری* dout, صحیح* inp,
                      صحیح B, صحیح T, صحیح C) {
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            اعشاری* dout_bt = dout + b * T * C + t * C;
            صحیح ix = inp[b * T + t];
            اعشاری* dwte_ix = dwte + ix * C;
            اعشاری* dwpe_t = dwpe + t * C;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

تهی layernorm_forward(اعشاری* out, اعشاری* mean, اعشاری* rstd,
                       اعشاری* inp, اعشاری* weight, اعشاری* bias,
                       صحیح B, صحیح T, صحیح C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    اعشاری eps = 1e-5f;
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            اعشاری* x = inp + b * T * C + t * C;
            // calculate the mean
            اعشاری m = 0.0f;
            برای (صحیح i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            اعشاری v = 0.0f;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            اعشاری s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            اعشاری* out_bt = out + b * T * C + t * C;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری n = (s * (x[i] - m)); // normalize
                اعشاری o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

تهی layernorm_backward(اعشاری* dinp, اعشاری* dweight, اعشاری* dbias,
                        اعشاری* dout, اعشاری* inp, اعشاری* weight, اعشاری* mean, اعشاری* rstd,
                        صحیح B, صحیح T, صحیح C) {
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            اعشاری* dout_bt = dout + b * T * C + t * C;
            اعشاری* inp_bt = inp + b * T * C + t * C;
            اعشاری* dinp_bt = dinp + b * T * C + t * C;
            اعشاری mean_bt = mean[b * T + t];
            اعشاری rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            اعشاری dnorm_mean = 0.0f;
            اعشاری dnorm_norm_mean = 0.0f;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                اعشاری dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            برای (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                اعشاری dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                اعشاری dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

تهی matmul_forward_naive(اعشاری* out,
                         ثابت اعشاری* inp, ثابت اعشاری* weight, ثابت اعشاری* bias,
                         صحیح B, صحیح T, صحیح C, صحیح OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel برای collapse(2)
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            صحیح bt = b * T + t;
            برای (صحیح o = 0; o < OC; o++) {
                اعشاری val = (bias != پوچ) ? bias[o] : 0.0f;
                برای (صحیح i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

تهی matmul_forward(اعشاری* out,
                    ثابت اعشاری* inp, ثابت اعشاری* weight, ثابت اعشاری* bias,
                    صحیح B, صحیح T, صحیح C, صحیح OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    ثابت صحیح LOOP_UNROLL = 8;
    اگر (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        بازگشت;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel برای
    برای (صحیح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        برای (صحیح o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            اعشاری result[LOOP_UNROLL];
            // initialize the bias, if it exists
            برای (صحیح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != پوچ) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            برای (صحیح i = 0; i < C; i++) {
                اعشاری w = weight[i + o * C];
                برای (صحیح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    صحیح bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            برای (صحیح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                صحیح bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

تهی matmul_backward(اعشاری* dinp, اعشاری* dweight, اعشاری* dbias,
                     ثابت اعشاری* dout, ثابت اعشاری* inp, ثابت اعشاری* weight,
                     صحیح B, صحیح T, صحیح C, صحیح OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel برای collapse(2)
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            ثابت اعشاری* dout_bt = dout + b * T * OC + t * OC;
            اعشاری* dinp_bt = dinp + b * T * C + t * C;
            برای (صحیح o = 0; o < OC; o++) {
                ثابت اعشاری* wrow = weight + o*C;
                اعشاری d = dout_bt[o];
                برای (صحیح i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel برای
    برای (صحیح o = 0; o < OC; o++) {
        برای (صحیح b = 0; b < B; b++) {
            برای (صحیح t = 0; t < T; t++) {
                ثابت اعشاری* dout_bt = dout + b * T * OC + t * OC;
                ثابت اعشاری* inp_bt = inp + b * T * C + t * C;
                اعشاری* dwrow = dweight + o*C;
                اعشاری d = dout_bt[o];
                اگر (dbias != پوچ) { dbias[o] += d; }
                برای (صحیح i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

تهی attention_forward(اعشاری* out, اعشاری* preatt, اعشاری* att,
                       اعشاری* inp,
                       صحیح B, صحیح T, صحیح C, صحیح NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    صحیح C3 = C*3;
    صحیح hs = C / NH; // head size
    اعشاری scale = 1.0 / sqrtf(hs);

    #pragma omp parallel برای collapse(3)
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            برای (صحیح h = 0; h < NH; h++) {
                اعشاری* query_t = inp + b * T * C3 + t * C3 + h * hs;
                اعشاری* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                اعشاری* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                اعشاری maxval = -10000.0f; // TODO something better
                برای (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* کلید_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    اعشاری val = 0.0f;
                    برای (صحیح i = 0; i < hs; i++) {
                        val += query_t[i] * کلید_t2[i];
                    }
                    val *= scale;
                    اگر (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                اعشاری expsum = 0.0f;
                برای (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                اعشاری expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                برای (صحیح t2 = 0; t2 < T; t2++) {
                    اگر (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } وگرنه {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                اعشاری* out_bth = out + b * T * C + t * C + h * hs;
                برای (صحیح i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                برای (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* مقدار_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    اعشاری att_btht2 = att_bth[t2];
                    برای (صحیح i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * مقدار_t2[i];
                    }
                }
            }
        }
    }
}

تهی attention_backward(اعشاری* dinp, اعشاری* dpreatt, اعشاری* datt,
                        اعشاری* dout, اعشاری* inp, اعشاری* att,
                        صحیح B, صحیح T, صحیح C, صحیح NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    صحیح C3 = C*3;
    صحیح hs = C / NH; // head size
    اعشاری scale = 1.f / sqrtf(hs);

    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            برای (صحیح h = 0; h < NH; h++) {
                اعشاری* att_bth = att + b*NH*T*T + h*T*T + t*T;
                اعشاری* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                اعشاری* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                اعشاری* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                اعشاری* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                اعشاری* dout_bth = dout + b * T * C + t * C + h * hs;
                برای (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* مقدار_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    اعشاری* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    برای (صحیح i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += مقدار_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                برای (صحیح t2 = 0; t2 <= t; t2++) {
                    برای (صحیح t3 = 0; t3 <= t; t3++) {
                        اعشاری indicator = t2 == t3 ? 1.0f : 0.0f;
                        اعشاری local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                برای (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* کلید_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    اعشاری* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    برای (صحیح i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += کلید_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
تهی gelu_forward(اعشاری* out, اعشاری* inp, صحیح N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    برای (صحیح i = 0; i < N; i++) {
        اعشاری x = inp[i];
        اعشاری cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, افزودن)
#اگر defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
تهی gelu_backward(اعشاری* dinp, اعشاری* inp, اعشاری* dout, صحیح N) {
    برای (صحیح i = 0; i < N; i++) {
        اعشاری x = inp[i];
        اعشاری cube = 0.044715f * x * x * x;
        اعشاری tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        اعشاری tanh_out = tanhf(tanh_arg);
        اعشاری coshf_out = coshf(tanh_arg);
        اعشاری sech_out = 1.0f / (coshf_out * coshf_out);
        اعشاری local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(برداشت)

تهی residual_forward(اعشاری* out, اعشاری* inp1, اعشاری* inp2, صحیح N) {
    برای (صحیح i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

تهی residual_backward(اعشاری* dinp1, اعشاری* dinp2, اعشاری* dout, صحیح N) {
    برای (صحیح i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

تهی softmax_forward(اعشاری* probs, اعشاری* logits, صحیح B, صحیح T, صحیح V, صحیح Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel برای collapse(2)
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            // probs <- softmax(logits)
            اعشاری* logits_bt = logits + b * T * Vp + t * Vp;
            اعشاری* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            اعشاری maxval = -10000.0f; // TODO something better
            برای (صحیح i = 0; i < V; i++) {
                اگر (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            اعشاری sum = 0.0f;
            برای (صحیح i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            برای (صحیح i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            برای (صحیح i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

تهی crossentropy_forward(اعشاری* losses,
                          اعشاری* probs, صحیح* targets,
                          صحیح B, صحیح T, صحیح Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            // loss = -log(probs[target])
            اعشاری* probs_bt = probs + b * T * Vp + t * Vp;
            صحیح ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

تهی crossentropy_softmax_backward(اعشاری* dlogits,
                           اعشاری* dlosses, اعشاری* probs, صحیح* targets,
                           صحیح B, صحیح T, صحیح V, صحیح Vp) {
    // backwards through both softmax and crossentropy
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            اعشاری* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            اعشاری* probs_bt = probs + b * T * Vp + t * Vp;
            اعشاری dloss = dlosses[b * T + t];
            صحیح ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            برای (صحیح i = 0; i < V; i++) {
                اعشاری p = probs_bt[i];
                اعشاری indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

تعریفنوع ساختار {
    صحیح max_seq_len; // max sequence length, e.g. 1024
    صحیح vocab_size; // vocab size, e.g. 50257
    صحیح padded_vocab_size; // padded to e.g. %128==0, 50304
    صحیح رقم_layers; // number of layers, e.g. 12
    صحیح رقم_heads; // number of heads in attention, e.g. 12
    صحیح channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define رقم_PARAMETER_TENSORS 16
تعریفنوع ساختار {
    اعشاری* wte; // (V, C)
    اعشاری* wpe; // (maxT, C)
    اعشاری* ln1w; // (L, C)
    اعشاری* ln1b; // (L, C)
    اعشاری* qkvw; // (L, 3*C, C)
    اعشاری* qkvb; // (L, 3*C)
    اعشاری* attprojw; // (L, C, C)
    اعشاری* attprojb; // (L, C)
    اعشاری* ln2w; // (L, C)
    اعشاری* ln2b; // (L, C)
    اعشاری* fcw; // (L, 4*C, C)
    اعشاری* fcb; // (L, 4*C)
    اعشاری* fcprojw; // (L, C, 4*C)
    اعشاری* fcprojb; // (L, C)
    اعشاری* lnfw; // (C)
    اعشاری* lnfb; // (C)
} ParameterTensors;

تهی fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.رقم_layers;
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
اعشاری* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t رقم_parameters = 0;
    برای (size_t i = 0; i < رقم_PARAMETER_TENSORS; i++) {
        رقم_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    اعشاری* params_memory = (اعشاری*)mallocCheck(رقم_parameters * اندازه(اعشاری));
    // assign all the tensors
    اعشاری** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    اعشاری* params_memory_iterator = params_memory;
    برای (size_t i = 0; i < رقم_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    بازگشت params_memory;
}

#define رقم_ACTIVATION_TENSORS 23
تعریفنوع ساختار {
    اعشاری* encoded; // (B, T, C)
    اعشاری* ln1; // (L, B, T, C)
    اعشاری* ln1_mean; // (L, B, T)
    اعشاری* ln1_rstd; // (L, B, T)
    اعشاری* qkv; // (L, B, T, 3*C)
    اعشاری* atty; // (L, B, T, C)
    اعشاری* preatt; // (L, B, NH, T, T)
    اعشاری* att; // (L, B, NH, T, T)
    اعشاری* attproj; // (L, B, T, C)
    اعشاری* residual2; // (L, B, T, C)
    اعشاری* ln2; // (L, B, T, C)
    اعشاری* ln2_mean; // (L, B, T)
    اعشاری* ln2_rstd; // (L, B, T)
    اعشاری* fch; // (L, B, T, 4*C)
    اعشاری* fch_gelu; // (L, B, T, 4*C)
    اعشاری* fcproj; // (L, B, T, C)
    اعشاری* residual3; // (L, B, T, C)
    اعشاری* lnf; // (B, T, C)
    اعشاری* lnf_mean; // (B, T)
    اعشاری* lnf_rstd; // (B, T)
    اعشاری* logits; // (B, T, V)
    اعشاری* probs; // (B, T, V)
    اعشاری* losses; // (B, T)
} ActivationTensors;

تهی fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, صحیح B, صحیح T) {
    size_t C = config.channels;
    size_t NH = config.رقم_heads;
    size_t L = config.رقم_layers;
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

اعشاری* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t رقم_activations = 0;
    برای (size_t i = 0; i < رقم_ACTIVATION_TENSORS; i++) {
        رقم_activations += act_sizes[i];
    }
    اعشاری* acts_memory = (اعشاری*)mallocCheck(رقم_activations * اندازه(اعشاری));
    اعشاری** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    اعشاری* acts_memory_iterator = acts_memory;
    برای (size_t i = 0; i < رقم_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    بازگشت acts_memory;
}

تعریفنوع ساختار {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[رقم_PARAMETER_TENSORS];
    اعشاری* params_memory;
    size_t رقم_parameters;
    // gradients of the weights
    ParameterTensors grads;
    اعشاری* grads_memory;
    // buffers for the AdamW optimizer
    اعشاری* m_memory;
    اعشاری* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[رقم_ACTIVATION_TENSORS];
    اعشاری* acts_memory;
    size_t رقم_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    اعشاری* grads_acts_memory;
    // other run state configuration
    صحیح batch_size; // the batch size (B) of current forward pass
    صحیح seq_len; // the sequence length (T) of current forward pass
    صحیح* inputs; // the input tokens for the current forward pass
    صحیح* targets; // the target tokens for the current forward pass
    اعشاری mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

تهی gpt2_build_from_checkpoint(GPT2 *model, ثابت نویسه* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    صحیح model_header[256];
    freadCheck(model_header, اندازه(صحیح), 256, model_file);
    اگر (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    اگر (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.رقم_layers = L = model_header[4];
    model->config.رقم_heads = NH = model_header[5];
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
    size_t رقم_parameters = 0;
    برای (size_t i = 0; i < رقم_PARAMETER_TENSORS; i++) {
        رقم_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", رقم_parameters);
    model->رقم_parameters = رقم_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, اندازه(اعشاری), رقم_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = پوچ;
    model->grads_memory = پوچ;
    model->m_memory = پوچ;
    model->v_memory = پوچ;
    model->grads_acts_memory = پوچ;
    model->inputs = پوچ;
    model->targets = پوچ;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

تهی gpt2_forward(GPT2 *model, صحیح* inputs, صحیح* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    اگر (model->params_memory == پوچ) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.رقم_layers;
    size_t NH = model->config.رقم_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    برای(صحیح i = 0; i < B * T; i++) {
        أكد(0 <= inputs[i] && inputs[i] < V);
        اگر (targets != پوچ) {
            أكد(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    اگر(model->acts_memory == پوچ) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t رقم_activations = 0;
        برای (size_t i = 0; i < رقم_ACTIVATION_TENSORS; i++) {
            رقم_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", رقم_activations);
        model->رقم_activations = رقم_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (صحیح*)mallocCheck(B * T * اندازه(صحیح));
        model->targets = (صحیح*)mallocCheck(B * T * اندازه(صحیح)); // might be unused if we never have targets but it's small
    } وگرنه {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        اگر (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (صحیح)B, (صحیح)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * اندازه(صحیح));
    اگر (targets != پوچ) {
        memcpy(model->targets, targets, B * T * اندازه(صحیح));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    اعشاری* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    برای (صحیح l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        اعشاری* l_ln1w = params.ln1w + l * C;
        اعشاری* l_ln1b = params.ln1b + l * C;
        اعشاری* l_qkvw = params.qkvw + l * 3*C * C;
        اعشاری* l_qkvb = params.qkvb + l * 3*C;
        اعشاری* l_attprojw = params.attprojw + l * C * C;
        اعشاری* l_attprojb = params.attprojb + l * C;
        اعشاری* l_ln2w = params.ln2w + l * C;
        اعشاری* l_ln2b = params.ln2b + l * C;
        اعشاری* l_fcw = params.fcw + l * 4*C * C;
        اعشاری* l_fcb = params.fcb + l * 4*C;
        اعشاری* l_fcprojw = params.fcprojw + l * C * 4*C;
        اعشاری* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        اعشاری* l_ln1 = acts.ln1 + l * B * T * C;
        اعشاری* l_ln1_mean = acts.ln1_mean + l * B * T;
        اعشاری* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        اعشاری* l_qkv = acts.qkv + l * B * T * 3*C;
        اعشاری* l_atty = acts.atty + l * B * T * C;
        اعشاری* l_preatt = acts.preatt + l * B * NH * T * T;
        اعشاری* l_att = acts.att + l * B * NH * T * T;
        اعشاری* l_attproj = acts.attproj + l * B * T * C;
        اعشاری* l_residual2 = acts.residual2 + l * B * T * C;
        اعشاری* l_ln2 = acts.ln2 + l * B * T * C;
        اعشاری* l_ln2_mean = acts.ln2_mean + l * B * T;
        اعشاری* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        اعشاری* l_fch = acts.fch + l * B * T * 4*C;
        اعشاری* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        اعشاری* l_fcproj = acts.fcproj + l * B * T * C;
        اعشاری* l_residual3 = acts.residual3 + l * B * T * C;

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
    matmul_forward(acts.logits, acts.lnf, params.wte, پوچ, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    اگر (targets != پوچ) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        اعشاری mean_loss = 0.0f;
        برای (صحیح i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } وگرنه {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

تهی gpt2_zero_grad(GPT2 *model) {
    اگر(model->grads_memory != پوچ) { memset(model->grads_memory, 0, model->رقم_parameters * اندازه(اعشاری)); }
    اگر(model->grads_acts_memory != پوچ) { memset(model->grads_acts_memory, 0, model->رقم_activations * اندازه(اعشاری)); }
}

تهی gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    اگر (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    اگر (model->grads_memory == پوچ) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.رقم_layers;
    size_t NH = model->config.رقم_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    اعشاری dloss_mean = 1.0f / (B*T);
    برای (صحیح i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, پوچ, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    اعشاری* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    اعشاری* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    برای (صحیح l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        اعشاری* l_ln1w = params.ln1w + l * C;
        اعشاری* l_qkvw = params.qkvw + l * 3*C * C;
        اعشاری* l_attprojw = params.attprojw + l * C * C;
        اعشاری* l_ln2w = params.ln2w + l * C;
        اعشاری* l_fcw = params.fcw + l * 4*C * C;
        اعشاری* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        اعشاری* dl_ln1w = grads.ln1w + l * C;
        اعشاری* dl_ln1b = grads.ln1b + l * C;
        اعشاری* dl_qkvw = grads.qkvw + l * 3*C * C;
        اعشاری* dl_qkvb = grads.qkvb + l * 3*C;
        اعشاری* dl_attprojw = grads.attprojw + l * C * C;
        اعشاری* dl_attprojb = grads.attprojb + l * C;
        اعشاری* dl_ln2w = grads.ln2w + l * C;
        اعشاری* dl_ln2b = grads.ln2b + l * C;
        اعشاری* dl_fcw = grads.fcw + l * 4*C * C;
        اعشاری* dl_fcb = grads.fcb + l * 4*C;
        اعشاری* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        اعشاری* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        اعشاری* l_ln1 = acts.ln1 + l * B * T * C;
        اعشاری* l_ln1_mean = acts.ln1_mean + l * B * T;
        اعشاری* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        اعشاری* l_qkv = acts.qkv + l * B * T * 3*C;
        اعشاری* l_atty = acts.atty + l * B * T * C;
        اعشاری* l_att = acts.att + l * B * NH * T * T;
        اعشاری* l_residual2 = acts.residual2 + l * B * T * C;
        اعشاری* l_ln2 = acts.ln2 + l * B * T * C;
        اعشاری* l_ln2_mean = acts.ln2_mean + l * B * T;
        اعشاری* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        اعشاری* l_fch = acts.fch + l * B * T * 4*C;
        اعشاری* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        اعشاری* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        اعشاری* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        اعشاری* dl_atty = grads_acts.atty + l * B * T * C;
        اعشاری* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        اعشاری* dl_att = grads_acts.att + l * B * NH * T * T;
        اعشاری* dl_attproj = grads_acts.attproj + l * B * T * C;
        اعشاری* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        اعشاری* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        اعشاری* dl_fch = grads_acts.fch + l * B * T * 4*C;
        اعشاری* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        اعشاری* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        اعشاری* dl_residual3 = grads_acts.residual3 + l * B * T * C;

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

تهی gpt2_update(GPT2 *model, اعشاری learning_rate, اعشاری beta1, اعشاری beta2, اعشاری eps, اعشاری weight_decay, صحیح t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    اگر (model->m_memory == پوچ) {
        model->m_memory = (اعشاری*)calloc(model->رقم_parameters, اندازه(اعشاری));
        model->v_memory = (اعشاری*)calloc(model->رقم_parameters, اندازه(اعشاری));
    }

    برای (size_t i = 0; i < model->رقم_parameters; i++) {
        اعشاری param = model->params_memory[i];
        اعشاری grad = model->grads_memory[i];

        // update the first moment (momentum)
        اعشاری m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        اعشاری v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        اعشاری m_hat = m / (1.0f - powf(beta1, t));
        اعشاری v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

تهی gpt2_free(GPT2 *model) {
    آزاد(model->params_memory);
    آزاد(model->grads_memory);
    آزاد(model->m_memory);
    آزاد(model->v_memory);
    آزاد(model->acts_memory);
    آزاد(model->grads_acts_memory);
    آزاد(model->inputs);
    آزاد(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned صحیح random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    بازگشت (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
اعشاری random_f32(uint64_t *state) { // random float32 in [0,1)
    بازگشت (random_u32(state) >> 8) / 16777216.0f;
}

صحیح sample_mult(اعشاری* probabilities, صحیح n, اعشاری coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    اعشاری cdf = 0.0f;
    برای (صحیح i = 0; i < n; i++) {
        cdf += probabilities[i];
        اگر (coin < cdf) {
            بازگشت i;
        }
    }
    بازگشت n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
صحیح اصلی() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    ثابت نویسه* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    ثابت نویسه* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    ثابت نویسه* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    ثابت نویسه* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    ثابت نویسه* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    ثابت نویسه* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    صحیح B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    صحیح T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.رقم_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.رقم_tokens / (B*T));
    صحیح val_رقم_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    صحیح* gen_tokens = (صحیح*)mallocCheck(B * T * اندازه(صحیح));
    ثابت صحیح genT = 64; // number of steps of inference we will do

    // train
    ساختار timespec start, end;
    برای (صحیح step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        اگر (step % 10 == 0) {
            اعشاری val_loss = 0.0f;
            dataloader_reset(&val_loader);
            برای (صحیح i = 0; i < val_رقم_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_رقم_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        اگر (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            برای(صحیح i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            برای (صحیح t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, پوچ, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                اعشاری* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                اعشاری coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                صحیح next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                اگر (tokenizer.init_ok) {
                    ثابت نویسه* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } وگرنه {
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
        دوگانه time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    آزاد(gen_tokens);
    بازگشت 0;
}
#endif
