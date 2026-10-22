/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <ctype.h>
#ضمن <stdint.h>
#ضمن <assert.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <string.h>
#ضمن <unistd.h>
#ifdef OMP
#ضمن <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#ضمن "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#ضمن "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#ضمن "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

فراغ encoder_forward(عائم* out,
                   صحيح* inp, عائم* wte, عائم* wpe,
                   صحيح B, صحيح T, صحيح C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            عائم* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            صحيح ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            عائم* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            عائم* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            لكل (صحيح i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

فراغ encoder_backward(عائم* dwte, عائم* dwpe,
                      عائم* dout, صحيح* inp,
                      صحيح B, صحيح T, صحيح C) {
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* dout_bt = dout + b * T * C + t * C;
            صحيح ix = inp[b * T + t];
            عائم* dwte_ix = dwte + ix * C;
            عائم* dwpe_t = dwpe + t * C;
            لكل (صحيح i = 0; i < C; i++) {
                عائم d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

فراغ layernorm_forward(عائم* out, عائم* mean, عائم* rstd,
                       عائم* inp, عائم* weight, عائم* bias,
                       صحيح B, صحيح T, صحيح C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    عائم eps = 1e-5f;
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            عائم* x = inp + b * T * C + t * C;
            // calculate the mean
            عائم m = 0.0f;
            لكل (صحيح i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            عائم v = 0.0f;
            لكل (صحيح i = 0; i < C; i++) {
                عائم xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            عائم s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            عائم* out_bt = out + b * T * C + t * C;
            لكل (صحيح i = 0; i < C; i++) {
                عائم n = (s * (x[i] - m)); // normalize
                عائم o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

فراغ layernorm_backward(عائم* dinp, عائم* dweight, عائم* dbias,
                        عائم* dout, عائم* inp, عائم* weight, عائم* mean, عائم* rstd,
                        صحيح B, صحيح T, صحيح C) {
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* dout_bt = dout + b * T * C + t * C;
            عائم* inp_bt = inp + b * T * C + t * C;
            عائم* dinp_bt = dinp + b * T * C + t * C;
            عائم mean_bt = mean[b * T + t];
            عائم rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            عائم dnorm_mean = 0.0f;
            عائم dnorm_norm_mean = 0.0f;
            لكل (صحيح i = 0; i < C; i++) {
                عائم norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                عائم dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            لكل (صحيح i = 0; i < C; i++) {
                عائم norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                عائم dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                عائم dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

فراغ matmul_forward_naive(عائم* out,
                         ثابت عائم* inp, ثابت عائم* weight, ثابت عائم* bias,
                         صحيح B, صحيح T, صحيح C, صحيح OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel لكل collapse(2)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            صحيح bt = b * T + t;
            لكل (صحيح o = 0; o < OC; o++) {
                عائم val = (bias != عدم) ? bias[o] : 0.0f;
                لكل (صحيح i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

فراغ matmul_forward(عائم* out,
                    ثابت عائم* inp, ثابت عائم* weight, ثابت عائم* bias,
                    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    ثابت صحيح LOOP_UNROLL = 8;
    إذا (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        أرجع;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel لكل
    لكل (صحيح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        لكل (صحيح o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            عائم result[LOOP_UNROLL];
            // initialize the bias, if it exists
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != عدم) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            لكل (صحيح i = 0; i < C; i++) {
                عائم w = weight[i + o * C];
                لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    صحيح bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                صحيح bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

فراغ matmul_backward(عائم* dinp, عائم* dweight, عائم* dbias,
                     ثابت عائم* dout, ثابت عائم* inp, ثابت عائم* weight,
                     صحيح B, صحيح T, صحيح C, صحيح OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel لكل collapse(2)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            ثابت عائم* dout_bt = dout + b * T * OC + t * OC;
            عائم* dinp_bt = dinp + b * T * C + t * C;
            لكل (صحيح o = 0; o < OC; o++) {
                ثابت عائم* wrow = weight + o*C;
                عائم d = dout_bt[o];
                لكل (صحيح i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel لكل
    لكل (صحيح o = 0; o < OC; o++) {
        لكل (صحيح b = 0; b < B; b++) {
            لكل (صحيح t = 0; t < T; t++) {
                ثابت عائم* dout_bt = dout + b * T * OC + t * OC;
                ثابت عائم* inp_bt = inp + b * T * C + t * C;
                عائم* dwrow = dweight + o*C;
                عائم d = dout_bt[o];
                إذا (dbias != عدم) { dbias[o] += d; }
                لكل (صحيح i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

فراغ attention_forward(عائم* out, عائم* preatt, عائم* att,
                       عائم* inp,
                       صحيح B, صحيح T, صحيح C, صحيح NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    صحيح C3 = C*3;
    صحيح hs = C / NH; // head size
    عائم scale = 1.0 / sqrtf(hs);

    #pragma omp parallel لكل collapse(3)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            لكل (صحيح h = 0; h < NH; h++) {
                عائم* query_t = inp + b * T * C3 + t * C3 + h * hs;
                عائم* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                عائم* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                عائم maxval = -10000.0f; // TODO something better
                لكل (صحيح t2 = 0; t2 <= t; t2++) {
                    عائم* کلید_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    عائم val = 0.0f;
                    لكل (صحيح i = 0; i < hs; i++) {
                        val += query_t[i] * کلید_t2[i];
                    }
                    val *= scale;
                    إذا (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                عائم expsum = 0.0f;
                لكل (صحيح t2 = 0; t2 <= t; t2++) {
                    عائم expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                عائم expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                لكل (صحيح t2 = 0; t2 < T; t2++) {
                    إذا (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } وإلا {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                عائم* out_bth = out + b * T * C + t * C + h * hs;
                لكل (صحيح i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                لكل (صحيح t2 = 0; t2 <= t; t2++) {
                    عائم* قدر_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    عائم att_btht2 = att_bth[t2];
                    لكل (صحيح i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * قدر_t2[i];
                    }
                }
            }
        }
    }
}

فراغ attention_backward(عائم* dinp, عائم* dpreatt, عائم* datt,
                        عائم* dout, عائم* inp, عائم* att,
                        صحيح B, صحيح T, صحيح C, صحيح NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    صحيح C3 = C*3;
    صحيح hs = C / NH; // head size
    عائم scale = 1.f / sqrtf(hs);

    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            لكل (صحيح h = 0; h < NH; h++) {
                عائم* att_bth = att + b*NH*T*T + h*T*T + t*T;
                عائم* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                عائم* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                عائم* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                عائم* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                عائم* dout_bth = dout + b * T * C + t * C + h * hs;
                لكل (صحيح t2 = 0; t2 <= t; t2++) {
                    عائم* قدر_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    عائم* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    لكل (صحيح i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += قدر_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                لكل (صحيح t2 = 0; t2 <= t; t2++) {
                    لكل (صحيح t3 = 0; t3 <= t; t3++) {
                        عائم indicator = t2 == t3 ? 1.0f : 0.0f;
                        عائم local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                لكل (صحيح t2 = 0; t2 <= t; t2++) {
                    عائم* کلید_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    عائم* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    لكل (صحيح i = 0; i < hs; i++) {
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
فراغ gelu_forward(عائم* out, عائم* inp, صحيح N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    لكل (صحيح i = 0; i < N; i++) {
        عائم x = inp[i];
        عائم cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, ڈالو)
#إذا defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
فراغ gelu_backward(عائم* dinp, عائم* inp, عائم* dout, صحيح N) {
    لكل (صحيح i = 0; i < N; i++) {
        عائم x = inp[i];
        عائم cube = 0.044715f * x * x * x;
        عائم tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        عائم tanh_out = tanhf(tanh_arg);
        عائم coshf_out = coshf(tanh_arg);
        عائم sech_out = 1.0f / (coshf_out * coshf_out);
        عائم local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(نکالو)

فراغ residual_forward(عائم* out, عائم* inp1, عائم* inp2, صحيح N) {
    لكل (صحيح i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

فراغ residual_backward(عائم* dinp1, عائم* dinp2, عائم* dout, صحيح N) {
    لكل (صحيح i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

فراغ softmax_forward(عائم* probs, عائم* logits, صحيح B, صحيح T, صحيح V, صحيح Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel لكل collapse(2)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            // probs <- softmax(logits)
            عائم* logits_bt = logits + b * T * Vp + t * Vp;
            عائم* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            عائم maxval = -10000.0f; // TODO something better
            لكل (صحيح i = 0; i < V; i++) {
                إذا (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            عائم sum = 0.0f;
            لكل (صحيح i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            لكل (صحيح i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            لكل (صحيح i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

فراغ crossentropy_forward(عائم* losses,
                          عائم* probs, صحيح* targets,
                          صحيح B, صحيح T, صحيح Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            // loss = -log(probs[target])
            عائم* probs_bt = probs + b * T * Vp + t * Vp;
            صحيح ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

فراغ crossentropy_softmax_backward(عائم* dlogits,
                           عائم* dlosses, عائم* probs, صحيح* targets,
                           صحيح B, صحيح T, صحيح V, صحيح Vp) {
    // backwards through both softmax and crossentropy
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            عائم* probs_bt = probs + b * T * Vp + t * Vp;
            عائم dloss = dlosses[b * T + t];
            صحيح ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            لكل (صحيح i = 0; i < V; i++) {
                عائم p = probs_bt[i];
                عائم indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

تعريفنوع بنية {
    صحيح max_seq_len; // max sequence length, e.g. 1024
    صحيح vocab_size; // vocab size, e.g. 50257
    صحيح padded_vocab_size; // padded to e.g. %128==0, 50304
    صحيح عدد_layers; // number of layers, e.g. 12
    صحيح عدد_heads; // number of heads in attention, e.g. 12
    صحيح channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define عدد_PARAMETER_TENSORS 16
تعريفنوع بنية {
    عائم* wte; // (V, C)
    عائم* wpe; // (maxT, C)
    عائم* ln1w; // (L, C)
    عائم* ln1b; // (L, C)
    عائم* qkvw; // (L, 3*C, C)
    عائم* qkvb; // (L, 3*C)
    عائم* attprojw; // (L, C, C)
    عائم* attprojb; // (L, C)
    عائم* ln2w; // (L, C)
    عائم* ln2b; // (L, C)
    عائم* fcw; // (L, 4*C, C)
    عائم* fcb; // (L, 4*C)
    عائم* fcprojw; // (L, C, 4*C)
    عائم* fcprojb; // (L, C)
    عائم* lnfw; // (C)
    عائم* lnfb; // (C)
} ParameterTensors;

فراغ fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.عدد_layers;
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
عائم* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t عدد_parameters = 0;
    لكل (size_t i = 0; i < عدد_PARAMETER_TENSORS; i++) {
        عدد_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    عائم* params_memory = (عائم*)mallocCheck(عدد_parameters * حجم(عائم));
    // assign all the tensors
    عائم** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    عائم* params_memory_iterator = params_memory;
    لكل (size_t i = 0; i < عدد_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    أرجع params_memory;
}

#define عدد_ACTIVATION_TENSORS 23
تعريفنوع بنية {
    عائم* encoded; // (B, T, C)
    عائم* ln1; // (L, B, T, C)
    عائم* ln1_mean; // (L, B, T)
    عائم* ln1_rstd; // (L, B, T)
    عائم* qkv; // (L, B, T, 3*C)
    عائم* atty; // (L, B, T, C)
    عائم* preatt; // (L, B, NH, T, T)
    عائم* att; // (L, B, NH, T, T)
    عائم* attproj; // (L, B, T, C)
    عائم* residual2; // (L, B, T, C)
    عائم* ln2; // (L, B, T, C)
    عائم* ln2_mean; // (L, B, T)
    عائم* ln2_rstd; // (L, B, T)
    عائم* fch; // (L, B, T, 4*C)
    عائم* fch_gelu; // (L, B, T, 4*C)
    عائم* fcproj; // (L, B, T, C)
    عائم* residual3; // (L, B, T, C)
    عائم* lnf; // (B, T, C)
    عائم* lnf_mean; // (B, T)
    عائم* lnf_rstd; // (B, T)
    عائم* logits; // (B, T, V)
    عائم* probs; // (B, T, V)
    عائم* losses; // (B, T)
} ActivationTensors;

فراغ fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, صحيح B, صحيح T) {
    size_t C = config.channels;
    size_t NH = config.عدد_heads;
    size_t L = config.عدد_layers;
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

عائم* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t عدد_activations = 0;
    لكل (size_t i = 0; i < عدد_ACTIVATION_TENSORS; i++) {
        عدد_activations += act_sizes[i];
    }
    عائم* acts_memory = (عائم*)mallocCheck(عدد_activations * حجم(عائم));
    عائم** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    عائم* acts_memory_iterator = acts_memory;
    لكل (size_t i = 0; i < عدد_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    أرجع acts_memory;
}

تعريفنوع بنية {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[عدد_PARAMETER_TENSORS];
    عائم* params_memory;
    size_t عدد_parameters;
    // gradients of the weights
    ParameterTensors grads;
    عائم* grads_memory;
    // buffers for the AdamW optimizer
    عائم* m_memory;
    عائم* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[عدد_ACTIVATION_TENSORS];
    عائم* acts_memory;
    size_t عدد_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    عائم* grads_acts_memory;
    // other run state configuration
    صحيح batch_size; // the batch size (B) of current forward pass
    صحيح seq_len; // the sequence length (T) of current forward pass
    صحيح* inputs; // the input tokens for the current forward pass
    صحيح* targets; // the target tokens for the current forward pass
    عائم mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

فراغ gpt2_build_from_checkpoint(GPT2 *model, ثابت حرف* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    صحيح model_header[256];
    freadCheck(model_header, حجم(صحيح), 256, model_file);
    إذا (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    إذا (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.عدد_layers = L = model_header[4];
    model->config.عدد_heads = NH = model_header[5];
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
    size_t عدد_parameters = 0;
    لكل (size_t i = 0; i < عدد_PARAMETER_TENSORS; i++) {
        عدد_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", عدد_parameters);
    model->عدد_parameters = عدد_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, حجم(عائم), عدد_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = عدم;
    model->grads_memory = عدم;
    model->m_memory = عدم;
    model->v_memory = عدم;
    model->grads_acts_memory = عدم;
    model->inputs = عدم;
    model->targets = عدم;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

فراغ gpt2_forward(GPT2 *model, صحيح* inputs, صحيح* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    إذا (model->params_memory == عدم) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.عدد_layers;
    size_t NH = model->config.عدد_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    لكل(صحيح i = 0; i < B * T; i++) {
        أكد(0 <= inputs[i] && inputs[i] < V);
        إذا (targets != عدم) {
            أكد(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    إذا(model->acts_memory == عدم) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t عدد_activations = 0;
        لكل (size_t i = 0; i < عدد_ACTIVATION_TENSORS; i++) {
            عدد_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", عدد_activations);
        model->عدد_activations = عدد_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (صحيح*)mallocCheck(B * T * حجم(صحيح));
        model->targets = (صحيح*)mallocCheck(B * T * حجم(صحيح)); // might be unused if we never have targets but it's small
    } وإلا {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        إذا (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (صحيح)B, (صحيح)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * حجم(صحيح));
    إذا (targets != عدم) {
        memcpy(model->targets, targets, B * T * حجم(صحيح));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    عائم* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    لكل (صحيح l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        عائم* l_ln1w = params.ln1w + l * C;
        عائم* l_ln1b = params.ln1b + l * C;
        عائم* l_qkvw = params.qkvw + l * 3*C * C;
        عائم* l_qkvb = params.qkvb + l * 3*C;
        عائم* l_attprojw = params.attprojw + l * C * C;
        عائم* l_attprojb = params.attprojb + l * C;
        عائم* l_ln2w = params.ln2w + l * C;
        عائم* l_ln2b = params.ln2b + l * C;
        عائم* l_fcw = params.fcw + l * 4*C * C;
        عائم* l_fcb = params.fcb + l * 4*C;
        عائم* l_fcprojw = params.fcprojw + l * C * 4*C;
        عائم* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        عائم* l_ln1 = acts.ln1 + l * B * T * C;
        عائم* l_ln1_mean = acts.ln1_mean + l * B * T;
        عائم* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        عائم* l_qkv = acts.qkv + l * B * T * 3*C;
        عائم* l_atty = acts.atty + l * B * T * C;
        عائم* l_preatt = acts.preatt + l * B * NH * T * T;
        عائم* l_att = acts.att + l * B * NH * T * T;
        عائم* l_attproj = acts.attproj + l * B * T * C;
        عائم* l_residual2 = acts.residual2 + l * B * T * C;
        عائم* l_ln2 = acts.ln2 + l * B * T * C;
        عائم* l_ln2_mean = acts.ln2_mean + l * B * T;
        عائم* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        عائم* l_fch = acts.fch + l * B * T * 4*C;
        عائم* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        عائم* l_fcproj = acts.fcproj + l * B * T * C;
        عائم* l_residual3 = acts.residual3 + l * B * T * C;

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
    matmul_forward(acts.logits, acts.lnf, params.wte, عدم, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    إذا (targets != عدم) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        عائم mean_loss = 0.0f;
        لكل (صحيح i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } وإلا {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

فراغ gpt2_zero_grad(GPT2 *model) {
    إذا(model->grads_memory != عدم) { memset(model->grads_memory, 0, model->عدد_parameters * حجم(عائم)); }
    إذا(model->grads_acts_memory != عدم) { memset(model->grads_acts_memory, 0, model->عدد_activations * حجم(عائم)); }
}

فراغ gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    إذا (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    إذا (model->grads_memory == عدم) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.عدد_layers;
    size_t NH = model->config.عدد_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    عائم dloss_mean = 1.0f / (B*T);
    لكل (صحيح i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, عدم, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    عائم* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    عائم* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    لكل (صحيح l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        عائم* l_ln1w = params.ln1w + l * C;
        عائم* l_qkvw = params.qkvw + l * 3*C * C;
        عائم* l_attprojw = params.attprojw + l * C * C;
        عائم* l_ln2w = params.ln2w + l * C;
        عائم* l_fcw = params.fcw + l * 4*C * C;
        عائم* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        عائم* dl_ln1w = grads.ln1w + l * C;
        عائم* dl_ln1b = grads.ln1b + l * C;
        عائم* dl_qkvw = grads.qkvw + l * 3*C * C;
        عائم* dl_qkvb = grads.qkvb + l * 3*C;
        عائم* dl_attprojw = grads.attprojw + l * C * C;
        عائم* dl_attprojb = grads.attprojb + l * C;
        عائم* dl_ln2w = grads.ln2w + l * C;
        عائم* dl_ln2b = grads.ln2b + l * C;
        عائم* dl_fcw = grads.fcw + l * 4*C * C;
        عائم* dl_fcb = grads.fcb + l * 4*C;
        عائم* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        عائم* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        عائم* l_ln1 = acts.ln1 + l * B * T * C;
        عائم* l_ln1_mean = acts.ln1_mean + l * B * T;
        عائم* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        عائم* l_qkv = acts.qkv + l * B * T * 3*C;
        عائم* l_atty = acts.atty + l * B * T * C;
        عائم* l_att = acts.att + l * B * NH * T * T;
        عائم* l_residual2 = acts.residual2 + l * B * T * C;
        عائم* l_ln2 = acts.ln2 + l * B * T * C;
        عائم* l_ln2_mean = acts.ln2_mean + l * B * T;
        عائم* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        عائم* l_fch = acts.fch + l * B * T * 4*C;
        عائم* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        عائم* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        عائم* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        عائم* dl_atty = grads_acts.atty + l * B * T * C;
        عائم* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        عائم* dl_att = grads_acts.att + l * B * NH * T * T;
        عائم* dl_attproj = grads_acts.attproj + l * B * T * C;
        عائم* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        عائم* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        عائم* dl_fch = grads_acts.fch + l * B * T * 4*C;
        عائم* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        عائم* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        عائم* dl_residual3 = grads_acts.residual3 + l * B * T * C;

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

فراغ gpt2_update(GPT2 *model, عائم learning_rate, عائم beta1, عائم beta2, عائم eps, عائم weight_decay, صحيح t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    إذا (model->m_memory == عدم) {
        model->m_memory = (عائم*)calloc(model->عدد_parameters, حجم(عائم));
        model->v_memory = (عائم*)calloc(model->عدد_parameters, حجم(عائم));
    }

    لكل (size_t i = 0; i < model->عدد_parameters; i++) {
        عائم param = model->params_memory[i];
        عائم grad = model->grads_memory[i];

        // update the first moment (momentum)
        عائم m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        عائم v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        عائم m_hat = m / (1.0f - powf(beta1, t));
        عائم v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

فراغ gpt2_free(GPT2 *model) {
    حرر(model->params_memory);
    حرر(model->grads_memory);
    حرر(model->m_memory);
    حرر(model->v_memory);
    حرر(model->acts_memory);
    حرر(model->grads_acts_memory);
    حرر(model->inputs);
    حرر(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned صحيح random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    أرجع (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
عائم random_f32(uint64_t *state) { // random float32 in [0,1)
    أرجع (random_u32(state) >> 8) / 16777216.0f;
}

صحيح sample_mult(عائم* probabilities, صحيح n, عائم coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    عائم cdf = 0.0f;
    لكل (صحيح i = 0; i < n; i++) {
        cdf += probabilities[i];
        إذا (coin < cdf) {
            أرجع i;
        }
    }
    أرجع n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
صحيح main() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    ثابت حرف* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    ثابت حرف* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    ثابت حرف* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    ثابت حرف* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    ثابت حرف* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    ثابت حرف* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    صحيح B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    صحيح T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.عدد_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.عدد_tokens / (B*T));
    صحيح val_عدد_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    صحيح* gen_tokens = (صحيح*)mallocCheck(B * T * حجم(صحيح));
    ثابت صحيح genT = 64; // number of steps of inference we will do

    // train
    بنية timespec start, end;
    لكل (صحيح step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        إذا (step % 10 == 0) {
            عائم val_loss = 0.0f;
            dataloader_reset(&val_loader);
            لكل (صحيح i = 0; i < val_عدد_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_عدد_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        إذا (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            لكل(صحيح i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            لكل (صحيح t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, عدم, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                عائم* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                عائم coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                صحيح next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                إذا (tokenizer.init_ok) {
                    ثابت حرف* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } وإلا {
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
        مزدوج time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    حرر(gen_tokens);
    أرجع 0;
}
#endif
