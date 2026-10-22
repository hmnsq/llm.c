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
#اگر_تعریف_شدہ OMP
#شامل <omp.h>
#اگر_ختم
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

خالی encoder_forward(اعشاری* برآمد,
                   صحیح* inp, اعشاری* wte, اعشاری* wpe,
                   صحیح B, صحیح T, صحیح C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            اعشاری* برآمد_bt = برآمد + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            صحیح ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            اعشاری* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            اعشاری* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            برائے (صحیح i = 0; i < C; i++) {
                برآمد_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

خالی encoder_backward(اعشاری* dwte, اعشاری* dwpe,
                      اعشاری* dout, صحیح* inp,
                      صحیح B, صحیح T, صحیح C) {
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            اعشاری* dout_bt = dout + b * T * C + t * C;
            صحیح ix = inp[b * T + t];
            اعشاری* dwte_ix = dwte + ix * C;
            اعشاری* dwpe_t = dwpe + t * C;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

خالی layernorm_forward(اعشاری* برآمد, اعشاری* اوسط, اعشاری* rstd,
                       اعشاری* inp, اعشاری* وزن, اعشاری* bias,
                       صحیح B, صحیح T, صحیح C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    اعشاری eps = 1e-5f;
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            اعشاری* x = inp + b * T * C + t * C;
            // calculate the mean
            اعشاری m = 0.0f;
            برائے (صحیح i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            اعشاری v = 0.0f;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            اعشاری s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            اعشاری* برآمد_bt = برآمد + b * T * C + t * C;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری n = (s * (x[i] - m)); // normalize
                اعشاری o = n * وزن[i] + bias[i]; // scale and shift
                برآمد_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            اوسط[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

خالی layernorm_backward(اعشاری* dinp, اعشاری* dweight, اعشاری* dbias,
                        اعشاری* dout, اعشاری* inp, اعشاری* وزن, اعشاری* اوسط, اعشاری* rstd,
                        صحیح B, صحیح T, صحیح C) {
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            اعشاری* dout_bt = dout + b * T * C + t * C;
            اعشاری* inp_bt = inp + b * T * C + t * C;
            اعشاری* dinp_bt = dinp + b * T * C + t * C;
            اعشاری اوسط_bt = اوسط[b * T + t];
            اعشاری rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            اعشاری dnorm_اوسط = 0.0f;
            اعشاری dnorm_norm_اوسط = 0.0f;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - اوسط_bt) * rstd_bt;
                اعشاری dnorm_i = وزن[i] * dout_bt[i];
                dnorm_اوسط += dnorm_i;
                dnorm_norm_اوسط += dnorm_i * norm_bti;
            }
            dnorm_اوسط = dnorm_اوسط / C;
            dnorm_norm_اوسط = dnorm_norm_اوسط / C;

            // now iterate again and accumulate all the gradients
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - اوسط_bt) * rstd_bt;
                اعشاری dnorm_i = وزن[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                اعشاری dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_اوسط; // term 2
                dval -= norm_bti * dnorm_norm_اوسط; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

خالی matmul_forward_naive(اعشاری* برآمد,
                         ثابت اعشاری* inp, ثابت اعشاری* وزن, ثابت اعشاری* bias,
                         صحیح B, صحیح T, صحیح C, صحیح OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #ہدایت omp parallel برائے collapse(2)
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            صحیح bt = b * T + t;
            برائے (صحیح o = 0; o < OC; o++) {
                اعشاری val = (bias != تہی) ? bias[o] : 0.0f;
                برائے (صحیح i = 0; i < C; i++) {
                    val += inp[bt * C + i] * وزن[o*C + i];
                }
                برآمد[bt * OC + o] = val;
            }
        }
    }
}

خالی matmul_forward(اعشاری* برآمد,
                    ثابت اعشاری* inp, ثابت اعشاری* وزن, ثابت اعشاری* bias,
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
        matmul_forward_naive(برآمد, inp, وزن, bias, B, T, C, OC);
        واپس;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #ہدایت omp parallel برائے
    برائے (صحیح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        برائے (صحیح o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            اعشاری نتیجہ[LOOP_UNROLL];
            // initialize the bias, if it exists
            برائے (صحیح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                نتیجہ[ibt] = (bias != تہی) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری w = وزن[i + o * C];
                برائے (صحیح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    صحیح bt = obt + ibt;
                    نتیجہ[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            برائے (صحیح ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                صحیح bt = obt + ibt;
                برآمد[bt * OC + o] = نتیجہ[ibt];
            }
        }
    }
}

خالی matmul_backward(اعشاری* dinp, اعشاری* dweight, اعشاری* dbias,
                     ثابت اعشاری* dout, ثابت اعشاری* inp, ثابت اعشاری* وزن,
                     صحیح B, صحیح T, صحیح C, صحیح OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #ہدایت omp parallel برائے collapse(2)
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            ثابت اعشاری* dout_bt = dout + b * T * OC + t * OC;
            اعشاری* dinp_bt = dinp + b * T * C + t * C;
            برائے (صحیح o = 0; o < OC; o++) {
                ثابت اعشاری* wrow = وزن + o*C;
                اعشاری d = dout_bt[o];
                برائے (صحیح i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #ہدایت omp parallel برائے
    برائے (صحیح o = 0; o < OC; o++) {
        برائے (صحیح b = 0; b < B; b++) {
            برائے (صحیح t = 0; t < T; t++) {
                ثابت اعشاری* dout_bt = dout + b * T * OC + t * OC;
                ثابت اعشاری* inp_bt = inp + b * T * C + t * C;
                اعشاری* dwrow = dweight + o*C;
                اعشاری d = dout_bt[o];
                اگر (dbias != تہی) { dbias[o] += d; }
                برائے (صحیح i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

خالی attention_forward(اعشاری* برآمد, اعشاری* preatt, اعشاری* att,
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

    #ہدایت omp parallel برائے collapse(3)
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            برائے (صحیح h = 0; h < NH; h++) {
                اعشاری* query_t = inp + b * T * C3 + t * C3 + h * hs;
                اعشاری* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                اعشاری* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                اعشاری maxval = -10000.0f; // TODO something better
                برائے (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* کلید_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    اعشاری val = 0.0f;
                    برائے (صحیح i = 0; i < hs; i++) {
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
                برائے (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                اعشاری expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                برائے (صحیح t2 = 0; t2 < T; t2++) {
                    اگر (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } ورنہ {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                اعشاری* برآمد_bth = برآمد + b * T * C + t * C + h * hs;
                برائے (صحیح i = 0; i < hs; i++) { برآمد_bth[i] = 0.0f; }
                برائے (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* قدر_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    اعشاری att_btht2 = att_bth[t2];
                    برائے (صحیح i = 0; i < hs; i++) {
                        برآمد_bth[i] += att_btht2 * قدر_t2[i];
                    }
                }
            }
        }
    }
}

خالی attention_backward(اعشاری* dinp, اعشاری* dpreatt, اعشاری* datt,
                        اعشاری* dout, اعشاری* inp, اعشاری* att,
                        صحیح B, صحیح T, صحیح C, صحیح NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    صحیح C3 = C*3;
    صحیح hs = C / NH; // head size
    اعشاری scale = 1.f / sqrtf(hs);

    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            برائے (صحیح h = 0; h < NH; h++) {
                اعشاری* att_bth = att + b*NH*T*T + h*T*T + t*T;
                اعشاری* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                اعشاری* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                اعشاری* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                اعشاری* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                اعشاری* dout_bth = dout + b * T * C + t * C + h * hs;
                برائے (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* قدر_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    اعشاری* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    برائے (صحیح i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += قدر_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                برائے (صحیح t2 = 0; t2 <= t; t2++) {
                    برائے (صحیح t3 = 0; t3 <= t; t3++) {
                        اعشاری indicator = t2 == t3 ? 1.0f : 0.0f;
                        اعشاری local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                برائے (صحیح t2 = 0; t2 <= t; t2++) {
                    اعشاری* کلید_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    اعشاری* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    برائے (صحیح i = 0; i < hs; i++) {
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

#تعریف_کرو GELU_SCALING_ضربی_عدد sqrtf(2.0f / M_PI)
خالی gelu_forward(اعشاری* برآمد, اعشاری* inp, صحیح N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    برائے (صحیح i = 0; i < N; i++) {
        اعشاری x = inp[i];
        اعشاری cube = 0.044715f * x * x * x;
        برآمد[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_ضربی_عدد * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#ہدایت float_control(precise, on, ڈالو)
#اگر defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#اگر_ختم
خالی gelu_backward(اعشاری* dinp, اعشاری* inp, اعشاری* dout, صحیح N) {
    برائے (صحیح i = 0; i < N; i++) {
        اعشاری x = inp[i];
        اعشاری cube = 0.044715f * x * x * x;
        اعشاری tanh_arg = GELU_SCALING_ضربی_عدد * (x + cube);
        اعشاری tanh_برآمد = tanhf(tanh_arg);
        اعشاری coshf_برآمد = coshf(tanh_arg);
        اعشاری sech_برآمد = 1.0f / (coshf_برآمد * coshf_برآمد);
        اعشاری local_grad = 0.5f * (1.0f + tanh_برآمد) + x * 0.5f * sech_برآمد * GELU_SCALING_ضربی_عدد * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#ہدایت float_control(نکالو)

خالی residual_forward(اعشاری* برآمد, اعشاری* inp1, اعشاری* inp2, صحیح N) {
    برائے (صحیح i = 0; i < N; i++) {
        برآمد[i] = inp1[i] + inp2[i];
    }
}

خالی residual_backward(اعشاری* dinp1, اعشاری* dinp2, اعشاری* dout, صحیح N) {
    برائے (صحیح i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

خالی softmax_forward(اعشاری* probs, اعشاری* logits, صحیح B, صحیح T, صحیح V, صحیح Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #ہدایت omp parallel برائے collapse(2)
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            // probs <- softmax(logits)
            اعشاری* logits_bt = logits + b * T * Vp + t * Vp;
            اعشاری* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            اعشاری maxval = -10000.0f; // TODO something better
            برائے (صحیح i = 0; i < V; i++) {
                اگر (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            اعشاری مجموع = 0.0f;
            برائے (صحیح i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                مجموع += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            برائے (صحیح i = 0; i < V; i++) {
                probs_bt[i] /= مجموع;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            برائے (صحیح i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

خالی crossentropy_forward(اعشاری* losses,
                          اعشاری* probs, صحیح* targets,
                          صحیح B, صحیح T, صحیح Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            // loss = -log(probs[target])
            اعشاری* probs_bt = probs + b * T * Vp + t * Vp;
            صحیح ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

خالی crossentropy_softmax_backward(اعشاری* dlogits,
                           اعشاری* dlosses, اعشاری* probs, صحیح* targets,
                           صحیح B, صحیح T, صحیح V, صحیح Vp) {
    // backwards through both softmax and crossentropy
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            اعشاری* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            اعشاری* probs_bt = probs + b * T * Vp + t * Vp;
            اعشاری dloss = dlosses[b * T + t];
            صحیح ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            برائے (صحیح i = 0; i < V; i++) {
                اعشاری p = probs_bt[i];
                اعشاری indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

قسمتعریف ساخت {
    صحیح max_seq_len; // max sequence length, e.g. 1024
    صحیح vocab_سائز; // vocab size, e.g. 50257
    صحیح padded_vocab_سائز; // padded to e.g. %128==0, 50304
    صحیح عدد_layers; // number of layers, e.g. 12
    صحیح عدد_heads; // number of heads in attention, e.g. 12
    صحیح channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#تعریف_کرو عدد_PARAMETER_TENSORS 16
قسمتعریف ساخت {
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

خالی fill_اندر_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_سائز;
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
اعشاری* malloc_and_نقطہ_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t عدد_parameters = 0;
    برائے (size_t i = 0; i < عدد_PARAMETER_TENSORS; i++) {
        عدد_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    اعشاری* params_حافظہ = (اعشاری*)mallocCheck(عدد_parameters * حجم(اعشاری));
    // assign all the tensors
    اعشاری** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    اعشاری* params_حافظہ_iterator = params_حافظہ;
    برائے (size_t i = 0; i < عدد_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_حافظہ_iterator;
        params_حافظہ_iterator += param_sizes[i];
    }
    واپس params_حافظہ;
}

#تعریف_کرو عدد_ACTIVATION_TENSORS 23
قسمتعریف ساخت {
    اعشاری* encoded; // (B, T, C)
    اعشاری* ln1; // (L, B, T, C)
    اعشاری* ln1_اوسط; // (L, B, T)
    اعشاری* ln1_rstd; // (L, B, T)
    اعشاری* qkv; // (L, B, T, 3*C)
    اعشاری* atty; // (L, B, T, C)
    اعشاری* preatt; // (L, B, NH, T, T)
    اعشاری* att; // (L, B, NH, T, T)
    اعشاری* attproj; // (L, B, T, C)
    اعشاری* residual2; // (L, B, T, C)
    اعشاری* ln2; // (L, B, T, C)
    اعشاری* ln2_اوسط; // (L, B, T)
    اعشاری* ln2_rstd; // (L, B, T)
    اعشاری* fch; // (L, B, T, 4*C)
    اعشاری* fch_gelu; // (L, B, T, 4*C)
    اعشاری* fcproj; // (L, B, T, C)
    اعشاری* residual3; // (L, B, T, C)
    اعشاری* lnf; // (B, T, C)
    اعشاری* lnf_اوسط; // (B, T)
    اعشاری* lnf_rstd; // (B, T)
    اعشاری* logits; // (B, T, V)
    اعشاری* probs; // (B, T, V)
    اعشاری* losses; // (B, T)
} ActivationTensors;

خالی fill_اندر_activation_sizes(size_t* act_sizes, GPT2Config config, صحیح B, صحیح T) {
    size_t C = config.channels;
    size_t NH = config.عدد_heads;
    size_t L = config.عدد_layers;
    size_t Vp = config.padded_vocab_سائز;
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

اعشاری* malloc_and_نقطہ_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t عدد_activations = 0;
    برائے (size_t i = 0; i < عدد_ACTIVATION_TENSORS; i++) {
        عدد_activations += act_sizes[i];
    }
    اعشاری* acts_حافظہ = (اعشاری*)mallocCheck(عدد_activations * حجم(اعشاری));
    اعشاری** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_اوسط, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_اوسط,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_اوسط, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    اعشاری* acts_حافظہ_iterator = acts_حافظہ;
    برائے (size_t i = 0; i < عدد_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_حافظہ_iterator;
        acts_حافظہ_iterator += act_sizes[i];
    }
    واپس acts_حافظہ;
}

قسمتعریف ساخت {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[عدد_PARAMETER_TENSORS];
    اعشاری* params_حافظہ;
    size_t عدد_parameters;
    // gradients of the weights
    ParameterTensors grads;
    اعشاری* grads_حافظہ;
    // buffers for the AdamW optimizer
    اعشاری* m_حافظہ;
    اعشاری* v_حافظہ;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[عدد_ACTIVATION_TENSORS];
    اعشاری* acts_حافظہ;
    size_t عدد_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    اعشاری* grads_acts_حافظہ;
    // other run state configuration
    صحیح batch_سائز; // the batch size (B) of current forward pass
    صحیح seq_len; // the sequence length (T) of current forward pass
    صحیح* inputs; // the input tokens for the current forward pass
    صحیح* targets; // the target tokens for the current forward pass
    اعشاری اوسط_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

خالی gpt2_build_from_checkpoint(GPT2 *model, ثابت حرف* checkpoint_راستہ) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_راستہ, "rb");
    صحیح model_سرخی[256];
    freadCheck(model_سرخی, حجم(صحیح), 256, model_file);
    اگر (model_سرخی[0] != 20240326) { لکھو("Bad magic model file\n"); باہر_نکلو(1); }
    اگر (model_سرخی[1] != 3) {
        لکھو("Bad version in model file\n");
        لکھو("---> HINT: try to re-run `python train_gpt2.py`\n");
        باہر_نکلو(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_سرخی[2];
    model->config.vocab_سائز = V = model_سرخی[3];
    model->config.عدد_layers = L = model_سرخی[4];
    model->config.عدد_heads = NH = model_سرخی[5];
    model->config.channels = C = model_سرخی[6];
    model->config.padded_vocab_سائز = Vp = model_سرخی[7];
    لکھو("[GPT-2]\n");
    لکھو("max_seq_len: %zu\n", maxT);
    لکھو("vocab_size: %zu\n", V);
    لکھو("padded_vocab_size: %zu\n", Vp);
    لکھو("num_layers: %zu\n", L);
    لکھو("num_heads: %zu\n", NH);
    لکھو("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_اندر_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t عدد_parameters = 0;
    برائے (size_t i = 0; i < عدد_PARAMETER_TENSORS; i++) {
        عدد_parameters += model->param_sizes[i];
    }
    لکھو("num_parameters: %zu\n", عدد_parameters);
    model->عدد_parameters = عدد_parameters;

    // read in all the parameters from file
    model->params_حافظہ = malloc_and_نقطہ_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_حافظہ, حجم(اعشاری), عدد_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_حافظہ = تہی;
    model->grads_حافظہ = تہی;
    model->m_حافظہ = تہی;
    model->v_حافظہ = تہی;
    model->grads_acts_حافظہ = تہی;
    model->inputs = تہی;
    model->targets = تہی;
    model->batch_سائز = 0;
    model->seq_len = 0;
    model->اوسط_loss = -1.0f; // -1.0f will designate no loss
}

خالی gpt2_forward(GPT2 *model, صحیح* inputs, صحیح* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    اگر (model->params_حافظہ == تہی) {
        لکھو("Error: model was not initialized properly.\n");
        باہر_نکلو(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_سائز;
    size_t Vp = model->config.padded_vocab_سائز;
    size_t L = model->config.عدد_layers;
    size_t NH = model->config.عدد_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    برائے(صحیح i = 0; i < B * T; i++) {
        أكد(0 <= inputs[i] && inputs[i] < V);
        اگر (targets != تہی) {
            أكد(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    اگر(model->acts_حافظہ == تہی) {
        // record the current B,T as well
        model->batch_سائز = B;
        model->seq_len = T;
        // and now allocate the space
        fill_اندر_activation_sizes(model->act_sizes, model->config, B, T);
        size_t عدد_activations = 0;
        برائے (size_t i = 0; i < عدد_ACTIVATION_TENSORS; i++) {
            عدد_activations += model->act_sizes[i];
        }
        لکھو("num_activations: %zu\n", عدد_activations);
        model->عدد_activations = عدد_activations;
        model->acts_حافظہ = malloc_and_نقطہ_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (صحیح*)mallocCheck(B * T * حجم(صحیح));
        model->targets = (صحیح*)mallocCheck(B * T * حجم(صحیح)); // might be unused if we never have targets but it's small
    } ورنہ {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        اگر (B != model->batch_سائز || T != model->seq_len) {
            لکھو("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_سائز, model->seq_len, (صحیح)B, (صحیح)T);
            باہر_نکلو(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    حافظہ_نقل(model->inputs, inputs, B * T * حجم(صحیح));
    اگر (targets != تہی) {
        حافظہ_نقل(model->targets, targets, B * T * حجم(صحیح));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    اعشاری* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    برائے (صحیح l = 0; l < L; l++) {

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
        اعشاری* l_ln1_اوسط = acts.ln1_اوسط + l * B * T;
        اعشاری* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        اعشاری* l_qkv = acts.qkv + l * B * T * 3*C;
        اعشاری* l_atty = acts.atty + l * B * T * C;
        اعشاری* l_preatt = acts.preatt + l * B * NH * T * T;
        اعشاری* l_att = acts.att + l * B * NH * T * T;
        اعشاری* l_attproj = acts.attproj + l * B * T * C;
        اعشاری* l_residual2 = acts.residual2 + l * B * T * C;
        اعشاری* l_ln2 = acts.ln2 + l * B * T * C;
        اعشاری* l_ln2_اوسط = acts.ln2_اوسط + l * B * T;
        اعشاری* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        اعشاری* l_fch = acts.fch + l * B * T * 4*C;
        اعشاری* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        اعشاری* l_fcproj = acts.fcproj + l * B * T * C;
        اعشاری* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_اوسط, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_اوسط, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_اوسط, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, تہی, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    اگر (targets != تہی) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        اعشاری اوسط_loss = 0.0f;
        برائے (صحیح i=0; i<B*T; i++) { اوسط_loss += model->acts.losses[i]; }
        اوسط_loss /= B*T;
        model->اوسط_loss = اوسط_loss;
    } ورنہ {
        // if we don't have targets, we don't have a loss
        model->اوسط_loss = -1.0f;
    }
}

خالی gpt2_zero_grad(GPT2 *model) {
    اگر(model->grads_حافظہ != تہی) { memset(model->grads_حافظہ, 0, model->عدد_parameters * حجم(اعشاری)); }
    اگر(model->grads_acts_حافظہ != تہی) { memset(model->grads_acts_حافظہ, 0, model->عدد_activations * حجم(اعشاری)); }
}

خالی gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    اگر (model->اوسط_loss == -1.0f) {
        لکھو("Error: must forward with targets before backward\n");
        باہر_نکلو(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    اگر (model->grads_حافظہ == تہی) {
        model->grads_حافظہ = malloc_and_نقطہ_parameters(&model->grads, model->param_sizes);
        model->grads_acts_حافظہ = malloc_and_نقطہ_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_سائز;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_سائز;
    size_t Vp = model->config.padded_vocab_سائز;
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
    اعشاری dloss_اوسط = 1.0f / (B*T);
    برائے (صحیح i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_اوسط; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, تہی, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    اعشاری* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    اعشاری* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_اوسط, acts.lnf_rstd, B, T, C);

    برائے (صحیح l = L-1; l >= 0; l--) {

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
        اعشاری* l_ln1_اوسط = acts.ln1_اوسط + l * B * T;
        اعشاری* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        اعشاری* l_qkv = acts.qkv + l * B * T * 3*C;
        اعشاری* l_atty = acts.atty + l * B * T * C;
        اعشاری* l_att = acts.att + l * B * NH * T * T;
        اعشاری* l_residual2 = acts.residual2 + l * B * T * C;
        اعشاری* l_ln2 = acts.ln2 + l * B * T * C;
        اعشاری* l_ln2_اوسط = acts.ln2_اوسط + l * B * T;
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
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_اوسط, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_اوسط, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

خالی gpt2_update(GPT2 *model, اعشاری learning_شرح, اعشاری beta1, اعشاری beta2, اعشاری eps, اعشاری وزن_decay, صحیح t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    اگر (model->m_حافظہ == تہی) {
        model->m_حافظہ = (اعشاری*)صفر_حافظہ_دو(model->عدد_parameters, حجم(اعشاری));
        model->v_حافظہ = (اعشاری*)صفر_حافظہ_دو(model->عدد_parameters, حجم(اعشاری));
    }

    برائے (size_t i = 0; i < model->عدد_parameters; i++) {
        اعشاری param = model->params_حافظہ[i];
        اعشاری grad = model->grads_حافظہ[i];

        // update the first moment (momentum)
        اعشاری m = beta1 * model->m_حافظہ[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        اعشاری v = beta2 * model->v_حافظہ[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        اعشاری m_hat = m / (1.0f - powf(beta1, t));
        اعشاری v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_حافظہ[i] = m;
        model->v_حافظہ[i] = v;
        model->params_حافظہ[i] -= learning_شرح * (m_hat / (sqrtf(v_hat) + eps) + وزن_decay * param);
    }
}

خالی gpt2_free(GPT2 *model) {
    آزاد(model->params_حافظہ);
    آزاد(model->grads_حافظہ);
    آزاد(model->m_حافظہ);
    آزاد(model->v_حافظہ);
    آزاد(model->acts_حافظہ);
    آزاد(model->grads_acts_حافظہ);
    آزاد(model->inputs);
    آزاد(model->targets);
}

#اگر_غیر_تعریف_شدہ TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

غیرعلامتی صحیح random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    واپس (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
اعشاری random_f32(uint64_t *state) { // random float32 in [0,1)
    واپس (random_u32(state) >> 8) / 16777216.0f;
}

صحیح sample_mult(اعشاری* probabilities, صحیح n, اعشاری coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    اعشاری cdf = 0.0f;
    برائے (صحیح i = 0; i < n; i++) {
        cdf += probabilities[i];
        اگر (coin < cdf) {
            واپس i;
        }
    }
    واپس n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
صحیح مین() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    ثابت حرف* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    ثابت حرف* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    ثابت حرف* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    ثابت حرف* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    ثابت حرف* train_علامات = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    ثابت حرف* val_علامات = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    صحیح B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    صحیح T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    ڈیٹا_Loader train_loader, val_loader;
    dataloader_init(&train_loader, train_علامات, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_علامات, B, T, 0, 1, 0);
    لکھو("train dataset num_batches: %zu\n", train_loader.عدد_علامات / (B*T));
    لکھو("val dataset num_batches: %zu\n", val_loader.عدد_علامات / (B*T));
    صحیح val_عدد_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    صحیح* gen_علامات = (صحیح*)mallocCheck(B * T * حجم(صحیح));
    ثابت صحیح genT = 64; // number of steps of inference we will do

    // train
    ساخت timespec آغاز, انتہا;
    برائے (صحیح قدم = 0; قدم <= 40; قدم++) {

        // once in a while estimate the validation loss
        اگر (قدم % 10 == 0) {
            اعشاری val_loss = 0.0f;
            dataloader_reset(&val_loader);
            برائے (صحیح i = 0; i < val_عدد_batches; i++) {
                dataloader_اگلا_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.اوسط_loss;
            }
            val_loss /= val_عدد_batches;
            لکھو("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        اگر (قدم > 0 && قدم % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            برائے(صحیح i = 0; i < B * T; ++i) {
                gen_علامات[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            لکھو("generating:\n---\n");
            برائے (صحیح t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_علامات, تہی, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                اعشاری* probs = model.acts.probs + (t-1) * model.config.padded_vocab_سائز;
                اعشاری coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                صحیح اگلا_token = sample_mult(probs, model.config.vocab_سائز, coin);
                gen_علامات[t] = اگلا_token;
                // print the generated token, either using the Tokenizer or a fallback
                اگر (tokenizer.init_ok) {
                    ثابت حرف* token_str = tokenizer_decode(&tokenizer, اگلا_token);
                    محفوظ_printf(token_str);
                } ورنہ {
                    // fall back to printing the token id
                    لکھو("%d ", اگلا_token);
                }
                fflush(stdout);
            }
            لکھو("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &آغاز);
        dataloader_اگلا_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, قدم+1);
        clock_gettime(CLOCK_MONOTONIC, &انتہا);
        دوہرا وقت_elapsed_s = (انتہا.tv_sec - آغاز.tv_sec) + (انتہا.tv_nsec - آغاز.tv_nsec) / 1e9;
        لکھو("step %d: train loss %f (took %f ms)\n", قدم, model.اوسط_loss, وقت_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    آزاد(gen_علامات);
    واپس 0;
}
#اگر_ختم
