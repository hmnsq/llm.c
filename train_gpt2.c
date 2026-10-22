/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <ctype.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <time.h>
#অন্তর্ভাব <string.h>
#অন্তর্ভাব <unistd.h>
#ifdef OMP
#অন্তর্ভাব <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#অন্তর্ভাব "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#অন্তর্ভাব "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#অন্তর্ভাব "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

শূন্য encoder_forward(দশমিক* out,
                   পূর্ণসংখ্যা* inp, দশমিক* wte, দশমিক* wpe,
                   পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            দশমিক* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            পূর্ণসংখ্যা ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            দশমিক* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            দশমিক* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

শূন্য encoder_backward(দশমিক* dwte, দশমিক* dwpe,
                      দশমিক* dout, পূর্ণসংখ্যা* inp,
                      পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C) {
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            দশমিক* dout_bt = dout + b * T * C + t * C;
            পূর্ণসংখ্যা ix = inp[b * T + t];
            দশমিক* dwte_ix = dwte + ix * C;
            দশমিক* dwpe_t = dwpe + t * C;
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

শূন্য layernorm_forward(দশমিক* out, দশমিক* mean, দশমিক* rstd,
                       দশমিক* inp, দশমিক* weight, দশমিক* bias,
                       পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    দশমিক eps = 1e-5f;
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            দশমিক* x = inp + b * T * C + t * C;
            // calculate the mean
            দশমিক m = 0.0f;
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            দশমিক v = 0.0f;
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            দশমিক s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            দশমিক* out_bt = out + b * T * C + t * C;
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক n = (s * (x[i] - m)); // normalize
                দশমিক o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

শূন্য layernorm_backward(দশমিক* dinp, দশমিক* dweight, দশমিক* dbias,
                        দশমিক* dout, দশমিক* inp, দশমিক* weight, দশমিক* mean, দশমিক* rstd,
                        পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C) {
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            দশমিক* dout_bt = dout + b * T * C + t * C;
            দশমিক* inp_bt = inp + b * T * C + t * C;
            দশমিক* dinp_bt = dinp + b * T * C + t * C;
            দশমিক mean_bt = mean[b * T + t];
            দশমিক rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            দশমিক dnorm_mean = 0.0f;
            দশমিক dnorm_norm_mean = 0.0f;
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                দশমিক dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                দশমিক dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                দশমিক dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

শূন্য matmul_forward_naive(দশমিক* out,
                         ধ্রুবক দশমিক* inp, ধ্রুবক দশমিক* weight, ধ্রুবক দশমিক* bias,
                         পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel জন্য collapse(2)
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            পূর্ণসংখ্যা bt = b * T + t;
            জন্য (পূর্ণসংখ্যা o = 0; o < OC; o++) {
                দশমিক val = (bias != ফাঁকা) ? bias[o] : 0.0f;
                জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

শূন্য matmul_forward(দশমিক* out,
                    ধ্রুবক দশমিক* inp, ধ্রুবক দশমিক* weight, ধ্রুবক দশমিক* bias,
                    পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    ধ্রুবক পূর্ণসংখ্যা LOOP_UNROLL = 8;
    যদি (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        ফেরত;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel জন্য
    জন্য (পূর্ণসংখ্যা obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        জন্য (পূর্ণসংখ্যা o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            দশমিক ফলাফল[LOOP_UNROLL];
            // initialize the bias, if it exists
            জন্য (পূর্ণসংখ্যা ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                ফলাফল[ibt] = (bias != ফাঁকা) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক w = weight[i + o * C];
                জন্য (পূর্ণসংখ্যা ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    পূর্ণসংখ্যা bt = obt + ibt;
                    ফলাফল[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            জন্য (পূর্ণসংখ্যা ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                পূর্ণসংখ্যা bt = obt + ibt;
                out[bt * OC + o] = ফলাফল[ibt];
            }
        }
    }
}

শূন্য matmul_backward(দশমিক* dinp, দশমিক* dweight, দশমিক* dbias,
                     ধ্রুবক দশমিক* dout, ধ্রুবক দশমিক* inp, ধ্রুবক দশমিক* weight,
                     পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel জন্য collapse(2)
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            ধ্রুবক দশমিক* dout_bt = dout + b * T * OC + t * OC;
            দশমিক* dinp_bt = dinp + b * T * C + t * C;
            জন্য (পূর্ণসংখ্যা o = 0; o < OC; o++) {
                ধ্রুবক দশমিক* wrow = weight + o*C;
                দশমিক d = dout_bt[o];
                জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel জন্য
    জন্য (পূর্ণসংখ্যা o = 0; o < OC; o++) {
        জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
            জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
                ধ্রুবক দশমিক* dout_bt = dout + b * T * OC + t * OC;
                ধ্রুবক দশমিক* inp_bt = inp + b * T * C + t * C;
                দশমিক* dwrow = dweight + o*C;
                দশমিক d = dout_bt[o];
                যদি (dbias != ফাঁকা) { dbias[o] += d; }
                জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

শূন্য attention_forward(দশমিক* out, দশমিক* preatt, দশমিক* att,
                       দশমিক* inp,
                       পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    পূর্ণসংখ্যা C3 = C*3;
    পূর্ণসংখ্যা hs = C / NH; // head size
    দশমিক scale = 1.0 / sqrtf(hs);

    #pragma omp parallel জন্য collapse(3)
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            জন্য (পূর্ণসংখ্যা h = 0; h < NH; h++) {
                দশমিক* query_t = inp + b * T * C3 + t * C3 + h * hs;
                দশমিক* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                দশমিক* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                দশমিক maxval = -10000.0f; // TODO something better
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 <= t; t2++) {
                    দশমিক* চাবি_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    দশমিক val = 0.0f;
                    জন্য (পূর্ণসংখ্যা i = 0; i < hs; i++) {
                        val += query_t[i] * চাবি_t2[i];
                    }
                    val *= scale;
                    যদি (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                দশমিক expsum = 0.0f;
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 <= t; t2++) {
                    দশমিক expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                দশমিক expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 < T; t2++) {
                    যদি (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } অন্যথা {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                দশমিক* out_bth = out + b * T * C + t * C + h * hs;
                জন্য (পূর্ণসংখ্যা i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 <= t; t2++) {
                    দশমিক* মান_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    দশমিক att_btht2 = att_bth[t2];
                    জন্য (পূর্ণসংখ্যা i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * মান_t2[i];
                    }
                }
            }
        }
    }
}

শূন্য attention_backward(দশমিক* dinp, দশমিক* dpreatt, দশমিক* datt,
                        দশমিক* dout, দশমিক* inp, দশমিক* att,
                        পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    পূর্ণসংখ্যা C3 = C*3;
    পূর্ণসংখ্যা hs = C / NH; // head size
    দশমিক scale = 1.f / sqrtf(hs);

    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            জন্য (পূর্ণসংখ্যা h = 0; h < NH; h++) {
                দশমিক* att_bth = att + b*NH*T*T + h*T*T + t*T;
                দশমিক* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                দশমিক* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                দশমিক* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                দশমিক* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                দশমিক* dout_bth = dout + b * T * C + t * C + h * hs;
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 <= t; t2++) {
                    দশমিক* মান_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    দশমিক* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    জন্য (পূর্ণসংখ্যা i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += মান_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 <= t; t2++) {
                    জন্য (পূর্ণসংখ্যা t3 = 0; t3 <= t; t3++) {
                        দশমিক indicator = t2 == t3 ? 1.0f : 0.0f;
                        দশমিক local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                জন্য (পূর্ণসংখ্যা t2 = 0; t2 <= t; t2++) {
                    দশমিক* চাবি_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    দশমিক* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    জন্য (পূর্ণসংখ্যা i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += চাবি_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
শূন্য gelu_forward(দশমিক* out, দশমিক* inp, পূর্ণসংখ্যা N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    জন্য (পূর্ণসংখ্যা i = 0; i < N; i++) {
        দশমিক x = inp[i];
        দশমিক cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, ঠেলো)
#যদি defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
শূন্য gelu_backward(দশমিক* dinp, দশমিক* inp, দশমিক* dout, পূর্ণসংখ্যা N) {
    জন্য (পূর্ণসংখ্যা i = 0; i < N; i++) {
        দশমিক x = inp[i];
        দশমিক cube = 0.044715f * x * x * x;
        দশমিক tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        দশমিক tanh_out = tanhf(tanh_arg);
        দশমিক coshf_out = coshf(tanh_arg);
        দশমিক sech_out = 1.0f / (coshf_out * coshf_out);
        দশমিক local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(বের_করো)

শূন্য residual_forward(দশমিক* out, দশমিক* inp1, দশমিক* inp2, পূর্ণসংখ্যা N) {
    জন্য (পূর্ণসংখ্যা i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

শূন্য residual_backward(দশমিক* dinp1, দশমিক* dinp2, দশমিক* dout, পূর্ণসংখ্যা N) {
    জন্য (পূর্ণসংখ্যা i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

শূন্য softmax_forward(দশমিক* probs, দশমিক* logits, পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা V, পূর্ণসংখ্যা Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel জন্য collapse(2)
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            // probs <- softmax(logits)
            দশমিক* logits_bt = logits + b * T * Vp + t * Vp;
            দশমিক* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            দশমিক maxval = -10000.0f; // TODO something better
            জন্য (পূর্ণসংখ্যা i = 0; i < V; i++) {
                যদি (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            দশমিক যোগ_2 = 0.0f;
            জন্য (পূর্ণসংখ্যা i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                যোগ_2 += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            জন্য (পূর্ণসংখ্যা i = 0; i < V; i++) {
                probs_bt[i] /= যোগ_2;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            জন্য (পূর্ণসংখ্যা i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

শূন্য crossentropy_forward(দশমিক* losses,
                          দশমিক* probs, পূর্ণসংখ্যা* targets,
                          পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            // loss = -log(probs[target])
            দশমিক* probs_bt = probs + b * T * Vp + t * Vp;
            পূর্ণসংখ্যা ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

শূন্য crossentropy_softmax_backward(দশমিক* dlogits,
                           দশমিক* dlosses, দশমিক* probs, পূর্ণসংখ্যা* targets,
                           পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা V, পূর্ণসংখ্যা Vp) {
    // backwards through both softmax and crossentropy
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            দশমিক* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            দশমিক* probs_bt = probs + b * T * Vp + t * Vp;
            দশমিক dloss = dlosses[b * T + t];
            পূর্ণসংখ্যা ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            জন্য (পূর্ণসংখ্যা i = 0; i < V; i++) {
                দশমিক p = probs_bt[i];
                দশমিক indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

প্রকারলেখ কাঠামো {
    পূর্ণসংখ্যা max_seq_len; // max sequence length, e.g. 1024
    পূর্ণসংখ্যা vocab_আকার; // vocab size, e.g. 50257
    পূর্ণসংখ্যা padded_vocab_আকার; // padded to e.g. %128==0, 50304
    পূর্ণসংখ্যা অঙ্ক_layers; // number of layers, e.g. 12
    পূর্ণসংখ্যা অঙ্ক_heads; // number of heads in attention, e.g. 12
    পূর্ণসংখ্যা channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define অঙ্ক_PARAMETER_TENSORS 16
প্রকারলেখ কাঠামো {
    দশমিক* wte; // (V, C)
    দশমিক* wpe; // (maxT, C)
    দশমিক* ln1w; // (L, C)
    দশমিক* ln1b; // (L, C)
    দশমিক* qkvw; // (L, 3*C, C)
    দশমিক* qkvb; // (L, 3*C)
    দশমিক* attprojw; // (L, C, C)
    দশমিক* attprojb; // (L, C)
    দশমিক* ln2w; // (L, C)
    দশমিক* ln2b; // (L, C)
    দশমিক* fcw; // (L, 4*C, C)
    দশমিক* fcb; // (L, 4*C)
    দশমিক* fcprojw; // (L, C, 4*C)
    দশমিক* fcprojb; // (L, C)
    দশমিক* lnfw; // (C)
    দশমিক* lnfb; // (C)
} ParameterTensors;

শূন্য fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_আকার;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.অঙ্ক_layers;
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
দশমিক* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t অঙ্ক_parameters = 0;
    জন্য (size_t i = 0; i < অঙ্ক_PARAMETER_TENSORS; i++) {
        অঙ্ক_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    দশমিক* params_memory = (দশমিক*)mallocCheck(অঙ্ক_parameters * আকার(দশমিক));
    // assign all the tensors
    দশমিক** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    দশমিক* params_memory_iterator = params_memory;
    জন্য (size_t i = 0; i < অঙ্ক_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    ফেরত params_memory;
}

#define অঙ্ক_ACTIVATION_TENSORS 23
প্রকারলেখ কাঠামো {
    দশমিক* encoded; // (B, T, C)
    দশমিক* ln1; // (L, B, T, C)
    দশমিক* ln1_mean; // (L, B, T)
    দশমিক* ln1_rstd; // (L, B, T)
    দশমিক* qkv; // (L, B, T, 3*C)
    দশমিক* atty; // (L, B, T, C)
    দশমিক* preatt; // (L, B, NH, T, T)
    দশমিক* att; // (L, B, NH, T, T)
    দশমিক* attproj; // (L, B, T, C)
    দশমিক* residual2; // (L, B, T, C)
    দশমিক* ln2; // (L, B, T, C)
    দশমিক* ln2_mean; // (L, B, T)
    দশমিক* ln2_rstd; // (L, B, T)
    দশমিক* fch; // (L, B, T, 4*C)
    দশমিক* fch_gelu; // (L, B, T, 4*C)
    দশমিক* fcproj; // (L, B, T, C)
    দশমিক* residual3; // (L, B, T, C)
    দশমিক* lnf; // (B, T, C)
    দশমিক* lnf_mean; // (B, T)
    দশমিক* lnf_rstd; // (B, T)
    দশমিক* logits; // (B, T, V)
    দশমিক* probs; // (B, T, V)
    দশমিক* losses; // (B, T)
} ActivationTensors;

শূন্য fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, পূর্ণসংখ্যা B, পূর্ণসংখ্যা T) {
    size_t C = config.channels;
    size_t NH = config.অঙ্ক_heads;
    size_t L = config.অঙ্ক_layers;
    size_t Vp = config.padded_vocab_আকার;
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

দশমিক* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t অঙ্ক_activations = 0;
    জন্য (size_t i = 0; i < অঙ্ক_ACTIVATION_TENSORS; i++) {
        অঙ্ক_activations += act_sizes[i];
    }
    দশমিক* acts_memory = (দশমিক*)mallocCheck(অঙ্ক_activations * আকার(দশমিক));
    দশমিক** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    দশমিক* acts_memory_iterator = acts_memory;
    জন্য (size_t i = 0; i < অঙ্ক_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    ফেরত acts_memory;
}

প্রকারলেখ কাঠামো {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[অঙ্ক_PARAMETER_TENSORS];
    দশমিক* params_memory;
    size_t অঙ্ক_parameters;
    // gradients of the weights
    ParameterTensors grads;
    দশমিক* grads_memory;
    // buffers for the AdamW optimizer
    দশমিক* m_memory;
    দশমিক* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[অঙ্ক_ACTIVATION_TENSORS];
    দশমিক* acts_memory;
    size_t অঙ্ক_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    দশমিক* grads_acts_memory;
    // other run state configuration
    পূর্ণসংখ্যা batch_আকার; // the batch size (B) of current forward pass
    পূর্ণসংখ্যা seq_len; // the sequence length (T) of current forward pass
    পূর্ণসংখ্যা* inputs; // the input tokens for the current forward pass
    পূর্ণসংখ্যা* targets; // the target tokens for the current forward pass
    দশমিক mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

শূন্য gpt2_build_from_checkpoint(GPT2 *model, ধ্রুবক অক্ষর* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    পূর্ণসংখ্যা model_header[256];
    freadCheck(model_header, আকার(পূর্ণসংখ্যা), 256, model_file);
    যদি (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    যদি (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_আকার = V = model_header[3];
    model->config.অঙ্ক_layers = L = model_header[4];
    model->config.অঙ্ক_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_আকার = Vp = model_header[7];
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
    size_t অঙ্ক_parameters = 0;
    জন্য (size_t i = 0; i < অঙ্ক_PARAMETER_TENSORS; i++) {
        অঙ্ক_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", অঙ্ক_parameters);
    model->অঙ্ক_parameters = অঙ্ক_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, আকার(দশমিক), অঙ্ক_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = ফাঁকা;
    model->grads_memory = ফাঁকা;
    model->m_memory = ফাঁকা;
    model->v_memory = ফাঁকা;
    model->grads_acts_memory = ফাঁকা;
    model->inputs = ফাঁকা;
    model->targets = ফাঁকা;
    model->batch_আকার = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

শূন্য gpt2_forward(GPT2 *model, পূর্ণসংখ্যা* inputs, পূর্ণসংখ্যা* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    যদি (model->params_memory == ফাঁকা) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_আকার;
    size_t Vp = model->config.padded_vocab_আকার;
    size_t L = model->config.অঙ্ক_layers;
    size_t NH = model->config.অঙ্ক_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    জন্য(পূর্ণসংখ্যা i = 0; i < B * T; i++) {
        প্রতিজ্ঞা(0 <= inputs[i] && inputs[i] < V);
        যদি (targets != ফাঁকা) {
            প্রতিজ্ঞা(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    যদি(model->acts_memory == ফাঁকা) {
        // record the current B,T as well
        model->batch_আকার = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t অঙ্ক_activations = 0;
        জন্য (size_t i = 0; i < অঙ্ক_ACTIVATION_TENSORS; i++) {
            অঙ্ক_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", অঙ্ক_activations);
        model->অঙ্ক_activations = অঙ্ক_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (পূর্ণসংখ্যা*)mallocCheck(B * T * আকার(পূর্ণসংখ্যা));
        model->targets = (পূর্ণসংখ্যা*)mallocCheck(B * T * আকার(পূর্ণসংখ্যা)); // might be unused if we never have targets but it's small
    } অন্যথা {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        যদি (B != model->batch_আকার || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_আকার, model->seq_len, (পূর্ণসংখ্যা)B, (পূর্ণসংখ্যা)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * আকার(পূর্ণসংখ্যা));
    যদি (targets != ফাঁকা) {
        memcpy(model->targets, targets, B * T * আকার(পূর্ণসংখ্যা));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    দশমিক* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    জন্য (পূর্ণসংখ্যা l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        দশমিক* l_ln1w = params.ln1w + l * C;
        দশমিক* l_ln1b = params.ln1b + l * C;
        দশমিক* l_qkvw = params.qkvw + l * 3*C * C;
        দশমিক* l_qkvb = params.qkvb + l * 3*C;
        দশমিক* l_attprojw = params.attprojw + l * C * C;
        দশমিক* l_attprojb = params.attprojb + l * C;
        দশমিক* l_ln2w = params.ln2w + l * C;
        দশমিক* l_ln2b = params.ln2b + l * C;
        দশমিক* l_fcw = params.fcw + l * 4*C * C;
        দশমিক* l_fcb = params.fcb + l * 4*C;
        দশমিক* l_fcprojw = params.fcprojw + l * C * 4*C;
        দশমিক* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        দশমিক* l_ln1 = acts.ln1 + l * B * T * C;
        দশমিক* l_ln1_mean = acts.ln1_mean + l * B * T;
        দশমিক* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        দশমিক* l_qkv = acts.qkv + l * B * T * 3*C;
        দশমিক* l_atty = acts.atty + l * B * T * C;
        দশমিক* l_preatt = acts.preatt + l * B * NH * T * T;
        দশমিক* l_att = acts.att + l * B * NH * T * T;
        দশমিক* l_attproj = acts.attproj + l * B * T * C;
        দশমিক* l_residual2 = acts.residual2 + l * B * T * C;
        দশমিক* l_ln2 = acts.ln2 + l * B * T * C;
        দশমিক* l_ln2_mean = acts.ln2_mean + l * B * T;
        দশমিক* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        দশমিক* l_fch = acts.fch + l * B * T * 4*C;
        দশমিক* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        দশমিক* l_fcproj = acts.fcproj + l * B * T * C;
        দশমিক* l_residual3 = acts.residual3 + l * B * T * C;

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
    matmul_forward(acts.logits, acts.lnf, params.wte, ফাঁকা, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    যদি (targets != ফাঁকা) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        দশমিক mean_loss = 0.0f;
        জন্য (পূর্ণসংখ্যা i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } অন্যথা {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

শূন্য gpt2_zero_grad(GPT2 *model) {
    যদি(model->grads_memory != ফাঁকা) { memset(model->grads_memory, 0, model->অঙ্ক_parameters * আকার(দশমিক)); }
    যদি(model->grads_acts_memory != ফাঁকা) { memset(model->grads_acts_memory, 0, model->অঙ্ক_activations * আকার(দশমিক)); }
}

শূন্য gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    যদি (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    যদি (model->grads_memory == ফাঁকা) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_আকার;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_আকার;
    size_t Vp = model->config.padded_vocab_আকার;
    size_t L = model->config.অঙ্ক_layers;
    size_t NH = model->config.অঙ্ক_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    দশমিক dloss_mean = 1.0f / (B*T);
    জন্য (পূর্ণসংখ্যা i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, ফাঁকা, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    দশমিক* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    দশমিক* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    জন্য (পূর্ণসংখ্যা l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        দশমিক* l_ln1w = params.ln1w + l * C;
        দশমিক* l_qkvw = params.qkvw + l * 3*C * C;
        দশমিক* l_attprojw = params.attprojw + l * C * C;
        দশমিক* l_ln2w = params.ln2w + l * C;
        দশমিক* l_fcw = params.fcw + l * 4*C * C;
        দশমিক* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        দশমিক* dl_ln1w = grads.ln1w + l * C;
        দশমিক* dl_ln1b = grads.ln1b + l * C;
        দশমিক* dl_qkvw = grads.qkvw + l * 3*C * C;
        দশমিক* dl_qkvb = grads.qkvb + l * 3*C;
        দশমিক* dl_attprojw = grads.attprojw + l * C * C;
        দশমিক* dl_attprojb = grads.attprojb + l * C;
        দশমিক* dl_ln2w = grads.ln2w + l * C;
        দশমিক* dl_ln2b = grads.ln2b + l * C;
        দশমিক* dl_fcw = grads.fcw + l * 4*C * C;
        দশমিক* dl_fcb = grads.fcb + l * 4*C;
        দশমিক* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        দশমিক* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        দশমিক* l_ln1 = acts.ln1 + l * B * T * C;
        দশমিক* l_ln1_mean = acts.ln1_mean + l * B * T;
        দশমিক* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        দশমিক* l_qkv = acts.qkv + l * B * T * 3*C;
        দশমিক* l_atty = acts.atty + l * B * T * C;
        দশমিক* l_att = acts.att + l * B * NH * T * T;
        দশমিক* l_residual2 = acts.residual2 + l * B * T * C;
        দশমিক* l_ln2 = acts.ln2 + l * B * T * C;
        দশমিক* l_ln2_mean = acts.ln2_mean + l * B * T;
        দশমিক* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        দশমিক* l_fch = acts.fch + l * B * T * 4*C;
        দশমিক* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        দশমিক* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        দশমিক* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        দশমিক* dl_atty = grads_acts.atty + l * B * T * C;
        দশমিক* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        দশমিক* dl_att = grads_acts.att + l * B * NH * T * T;
        দশমিক* dl_attproj = grads_acts.attproj + l * B * T * C;
        দশমিক* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        দশমিক* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        দশমিক* dl_fch = grads_acts.fch + l * B * T * 4*C;
        দশমিক* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        দশমিক* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        দশমিক* dl_residual3 = grads_acts.residual3 + l * B * T * C;

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

শূন্য gpt2_update(GPT2 *model, দশমিক learning_rate, দশমিক beta1, দশমিক beta2, দশমিক eps, দশমিক weight_decay, পূর্ণসংখ্যা t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    যদি (model->m_memory == ফাঁকা) {
        model->m_memory = (দশমিক*)calloc(model->অঙ্ক_parameters, আকার(দশমিক));
        model->v_memory = (দশমিক*)calloc(model->অঙ্ক_parameters, আকার(দশমিক));
    }

    জন্য (size_t i = 0; i < model->অঙ্ক_parameters; i++) {
        দশমিক param = model->params_memory[i];
        দশমিক grad = model->grads_memory[i];

        // update the first moment (momentum)
        দশমিক m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        দশমিক v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        দশমিক m_hat = m / (1.0f - powf(beta1, t));
        দশমিক v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

শূন্য gpt2_free(GPT2 *model) {
    মুক্ত(model->params_memory);
    মুক্ত(model->grads_memory);
    মুক্ত(model->m_memory);
    মুক্ত(model->v_memory);
    মুক্ত(model->acts_memory);
    মুক্ত(model->grads_acts_memory);
    মুক্ত(model->inputs);
    মুক্ত(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned পূর্ণসংখ্যা random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ফেরত (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
দশমিক random_f32(uint64_t *state) { // random float32 in [0,1)
    ফেরত (random_u32(state) >> 8) / 16777216.0f;
}

পূর্ণসংখ্যা sample_mult(দশমিক* probabilities, পূর্ণসংখ্যা n, দশমিক coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    দশমিক cdf = 0.0f;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        cdf += probabilities[i];
        যদি (coin < cdf) {
            ফেরত i;
        }
    }
    ফেরত n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
পূর্ণসংখ্যা প্রধান() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    ধ্রুবক অক্ষর* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    ধ্রুবক অক্ষর* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    ধ্রুবক অক্ষর* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    ধ্রুবক অক্ষর* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    ধ্রুবক অক্ষর* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    ধ্রুবক অক্ষর* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    পূর্ণসংখ্যা B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    পূর্ণসংখ্যা T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    তথ্য_Loader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.অঙ্ক_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.অঙ্ক_tokens / (B*T));
    পূর্ণসংখ্যা val_অঙ্ক_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    পূর্ণসংখ্যা* gen_tokens = (পূর্ণসংখ্যা*)mallocCheck(B * T * আকার(পূর্ণসংখ্যা));
    ধ্রুবক পূর্ণসংখ্যা genT = 64; // number of steps of inference we will do

    // train
    কাঠামো timespec start, end;
    জন্য (পূর্ণসংখ্যা step = 0; step <= 40; step++) {

        // once in a while estimate the validation loss
        যদি (step % 10 == 0) {
            দশমিক val_loss = 0.0f;
            dataloader_reset(&val_loader);
            জন্য (পূর্ণসংখ্যা i = 0; i < val_অঙ্ক_batches; i++) {
                dataloader_পরবর্তী_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_অঙ্ক_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        যদি (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            জন্য(পূর্ণসংখ্যা i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            জন্য (পূর্ণসংখ্যা t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, ফাঁকা, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                দশমিক* probs = model.acts.probs + (t-1) * model.config.padded_vocab_আকার;
                দশমিক coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                পূর্ণসংখ্যা পরবর্তী_token = sample_mult(probs, model.config.vocab_আকার, coin);
                gen_tokens[t] = পরবর্তী_token;
                // print the generated token, either using the Tokenizer or a fallback
                যদি (tokenizer.init_ok) {
                    ধ্রুবক অক্ষর* token_str = tokenizer_decode(&tokenizer, পরবর্তী_token);
                    safe_printf(token_str);
                } অন্যথা {
                    // fall back to printing the token id
                    printf("%d ", পরবর্তী_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_পরবর্তী_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        দ্বিগুণ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    মুক্ত(gen_tokens);
    ফেরত 0;
}
#endif
