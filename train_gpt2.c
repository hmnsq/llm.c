/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <ctype.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <time.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <unistd.h>
#నిర్వచితమైతే OMP
#అన్తర్భావ <omp.h>
#అయితే_ముగింపు
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#అన్తర్భావ "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#అన్తర్భావ "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#అన్తర్భావ "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

శూన్య encoder_forward(దశాంశం* బహిర్గతం,
                   పూర్ణాంకం* inp, దశాంశం* wte, దశాంశం* wpe,
                   పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            దశాంశం* బహిర్గతం_bt = బహిర్గతం + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            పూర్ణాంకం ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            దశాంశం* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            దశాంశం* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                బహిర్గతం_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

శూన్య encoder_backward(దశాంశం* dwte, దశాంశం* dwpe,
                      దశాంశం* dout, పూర్ణాంకం* inp,
                      పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C) {
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            దశాంశం* dout_bt = dout + b * T * C + t * C;
            పూర్ణాంకం ix = inp[b * T + t];
            దశాంశం* dwte_ix = dwte + ix * C;
            దశాంశం* dwpe_t = dwpe + t * C;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

శూన్య layernorm_forward(దశాంశం* బహిర్గతం, దశాంశం* సగటు, దశాంశం* rstd,
                       దశాంశం* inp, దశాంశం* బరువు, దశాంశం* bias,
                       పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    దశాంశం eps = 1e-5f;
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            దశాంశం* x = inp + b * T * C + t * C;
            // calculate the mean
            దశాంశం m = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            దశాంశం v = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            దశాంశం s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            దశాంశం* బహిర్గతం_bt = బహిర్గతం + b * T * C + t * C;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం n = (s * (x[i] - m)); // normalize
                దశాంశం o = n * బరువు[i] + bias[i]; // scale and shift
                బహిర్గతం_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            సగటు[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

శూన్య layernorm_backward(దశాంశం* dinp, దశాంశం* dweight, దశాంశం* dbias,
                        దశాంశం* dout, దశాంశం* inp, దశాంశం* బరువు, దశాంశం* సగటు, దశాంశం* rstd,
                        పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C) {
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            దశాంశం* dout_bt = dout + b * T * C + t * C;
            దశాంశం* inp_bt = inp + b * T * C + t * C;
            దశాంశం* dinp_bt = dinp + b * T * C + t * C;
            దశాంశం సగటు_bt = సగటు[b * T + t];
            దశాంశం rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            దశాంశం dnorm_సగటు = 0.0f;
            దశాంశం dnorm_norm_సగటు = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం norm_bti = (inp_bt[i] - సగటు_bt) * rstd_bt;
                దశాంశం dnorm_i = బరువు[i] * dout_bt[i];
                dnorm_సగటు += dnorm_i;
                dnorm_norm_సగటు += dnorm_i * norm_bti;
            }
            dnorm_సగటు = dnorm_సగటు / C;
            dnorm_norm_సగటు = dnorm_norm_సగటు / C;

            // now iterate again and accumulate all the gradients
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం norm_bti = (inp_bt[i] - సగటు_bt) * rstd_bt;
                దశాంశం dnorm_i = బరువు[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                దశాంశం dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_సగటు; // term 2
                dval -= norm_bti * dnorm_norm_సగటు; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

శూన్య matmul_forward_naive(దశాంశం* బహిర్గతం,
                         స్థిరరాశి దశాంశం* inp, స్థిరరాశి దశాంశం* బరువు, స్థిరరాశి దశాంశం* bias,
                         పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #సూచన omp parallel కొరకు collapse(2)
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            పూర్ణాంకం bt = b * T + t;
            కొరకు (పూర్ణాంకం o = 0; o < OC; o++) {
                దశాంశం val = (bias != ఖాళీ) ? bias[o] : 0.0f;
                కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                    val += inp[bt * C + i] * బరువు[o*C + i];
                }
                బహిర్గతం[bt * OC + o] = val;
            }
        }
    }
}

శూన్య matmul_forward(దశాంశం* బహిర్గతం,
                    స్థిరరాశి దశాంశం* inp, స్థిరరాశి దశాంశం* బరువు, స్థిరరాశి దశాంశం* bias,
                    పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    స్థిరరాశి పూర్ణాంకం LOOP_UNROLL = 8;
    ఒకవేళ (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(బహిర్గతం, inp, బరువు, bias, B, T, C, OC);
        తిరిగి;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #సూచన omp parallel కొరకు
    కొరకు (పూర్ణాంకం obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        కొరకు (పూర్ణాంకం o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            దశాంశం ఫలితం[LOOP_UNROLL];
            // initialize the bias, if it exists
            కొరకు (పూర్ణాంకం ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                ఫలితం[ibt] = (bias != ఖాళీ) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం w = బరువు[i + o * C];
                కొరకు (పూర్ణాంకం ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    పూర్ణాంకం bt = obt + ibt;
                    ఫలితం[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            కొరకు (పూర్ణాంకం ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                పూర్ణాంకం bt = obt + ibt;
                బహిర్గతం[bt * OC + o] = ఫలితం[ibt];
            }
        }
    }
}

శూన్య matmul_backward(దశాంశం* dinp, దశాంశం* dweight, దశాంశం* dbias,
                     స్థిరరాశి దశాంశం* dout, స్థిరరాశి దశాంశం* inp, స్థిరరాశి దశాంశం* బరువు,
                     పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #సూచన omp parallel కొరకు collapse(2)
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            స్థిరరాశి దశాంశం* dout_bt = dout + b * T * OC + t * OC;
            దశాంశం* dinp_bt = dinp + b * T * C + t * C;
            కొరకు (పూర్ణాంకం o = 0; o < OC; o++) {
                స్థిరరాశి దశాంశం* wrow = బరువు + o*C;
                దశాంశం d = dout_bt[o];
                కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #సూచన omp parallel కొరకు
    కొరకు (పూర్ణాంకం o = 0; o < OC; o++) {
        కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
            కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
                స్థిరరాశి దశాంశం* dout_bt = dout + b * T * OC + t * OC;
                స్థిరరాశి దశాంశం* inp_bt = inp + b * T * C + t * C;
                దశాంశం* dwrow = dweight + o*C;
                దశాంశం d = dout_bt[o];
                ఒకవేళ (dbias != ఖాళీ) { dbias[o] += d; }
                కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

శూన్య attention_forward(దశాంశం* బహిర్గతం, దశాంశం* preatt, దశాంశం* att,
                       దశాంశం* inp,
                       పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    పూర్ణాంకం C3 = C*3;
    పూర్ణాంకం hs = C / NH; // head size
    దశాంశం scale = 1.0 / sqrtf(hs);

    #సూచన omp parallel కొరకు collapse(3)
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            కొరకు (పూర్ణాంకం h = 0; h < NH; h++) {
                దశాంశం* query_t = inp + b * T * C3 + t * C3 + h * hs;
                దశాంశం* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                దశాంశం* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                దశాంశం maxval = -10000.0f; // TODO something better
                కొరకు (పూర్ణాంకం t2 = 0; t2 <= t; t2++) {
                    దశాంశం* కీలకం_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    దశాంశం val = 0.0f;
                    కొరకు (పూర్ణాంకం i = 0; i < hs; i++) {
                        val += query_t[i] * కీలకం_t2[i];
                    }
                    val *= scale;
                    ఒకవేళ (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                దశాంశం expsum = 0.0f;
                కొరకు (పూర్ణాంకం t2 = 0; t2 <= t; t2++) {
                    దశాంశం expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                దశాంశం expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                కొరకు (పూర్ణాంకం t2 = 0; t2 < T; t2++) {
                    ఒకవేళ (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } లేకపోతే {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                దశాంశం* బహిర్గతం_bth = బహిర్గతం + b * T * C + t * C + h * hs;
                కొరకు (పూర్ణాంకం i = 0; i < hs; i++) { బహిర్గతం_bth[i] = 0.0f; }
                కొరకు (పూర్ణాంకం t2 = 0; t2 <= t; t2++) {
                    దశాంశం* విలువ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    దశాంశం att_btht2 = att_bth[t2];
                    కొరకు (పూర్ణాంకం i = 0; i < hs; i++) {
                        బహిర్గతం_bth[i] += att_btht2 * విలువ_t2[i];
                    }
                }
            }
        }
    }
}

శూన్య attention_backward(దశాంశం* dinp, దశాంశం* dpreatt, దశాంశం* datt,
                        దశాంశం* dout, దశాంశం* inp, దశాంశం* att,
                        పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    పూర్ణాంకం C3 = C*3;
    పూర్ణాంకం hs = C / NH; // head size
    దశాంశం scale = 1.f / sqrtf(hs);

    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            కొరకు (పూర్ణాంకం h = 0; h < NH; h++) {
                దశాంశం* att_bth = att + b*NH*T*T + h*T*T + t*T;
                దశాంశం* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                దశాంశం* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                దశాంశం* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                దశాంశం* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                దశాంశం* dout_bth = dout + b * T * C + t * C + h * hs;
                కొరకు (పూర్ణాంకం t2 = 0; t2 <= t; t2++) {
                    దశాంశం* విలువ_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    దశాంశం* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    కొరకు (పూర్ణాంకం i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += విలువ_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                కొరకు (పూర్ణాంకం t2 = 0; t2 <= t; t2++) {
                    కొరకు (పూర్ణాంకం t3 = 0; t3 <= t; t3++) {
                        దశాంశం indicator = t2 == t3 ? 1.0f : 0.0f;
                        దశాంశం local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                కొరకు (పూర్ణాంకం t2 = 0; t2 <= t; t2++) {
                    దశాంశం* కీలకం_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    దశాంశం* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    కొరకు (పూర్ణాంకం i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += కీలకం_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#నిర్వచించు GELU_SCALING_గుణకం sqrtf(2.0f / M_PI)
శూన్య gelu_forward(దశాంశం* బహిర్గతం, దశాంశం* inp, పూర్ణాంకం N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    కొరకు (పూర్ణాంకం i = 0; i < N; i++) {
        దశాంశం x = inp[i];
        దశాంశం cube = 0.044715f * x * x * x;
        బహిర్గతం[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_గుణకం * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#సూచన float_control(precise, on, నెట్టు)
#ఒకవేళ defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#అయితే_ముగింపు
శూన్య gelu_backward(దశాంశం* dinp, దశాంశం* inp, దశాంశం* dout, పూర్ణాంకం N) {
    కొరకు (పూర్ణాంకం i = 0; i < N; i++) {
        దశాంశం x = inp[i];
        దశాంశం cube = 0.044715f * x * x * x;
        దశాంశం tanh_arg = GELU_SCALING_గుణకం * (x + cube);
        దశాంశం tanh_బహిర్గతం = tanhf(tanh_arg);
        దశాంశం coshf_బహిర్గతం = coshf(tanh_arg);
        దశాంశం sech_బహిర్గతం = 1.0f / (coshf_బహిర్గతం * coshf_బహిర్గతం);
        దశాంశం local_grad = 0.5f * (1.0f + tanh_బహిర్గతం) + x * 0.5f * sech_బహిర్గతం * GELU_SCALING_గుణకం * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#సూచన float_control(తీయి)

శూన్య residual_forward(దశాంశం* బహిర్గతం, దశాంశం* inp1, దశాంశం* inp2, పూర్ణాంకం N) {
    కొరకు (పూర్ణాంకం i = 0; i < N; i++) {
        బహిర్గతం[i] = inp1[i] + inp2[i];
    }
}

శూన్య residual_backward(దశాంశం* dinp1, దశాంశం* dinp2, దశాంశం* dout, పూర్ణాంకం N) {
    కొరకు (పూర్ణాంకం i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

శూన్య softmax_forward(దశాంశం* probs, దశాంశం* logits, పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం V, పూర్ణాంకం Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #సూచన omp parallel కొరకు collapse(2)
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            // probs <- softmax(logits)
            దశాంశం* logits_bt = logits + b * T * Vp + t * Vp;
            దశాంశం* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            దశాంశం maxval = -10000.0f; // TODO something better
            కొరకు (పూర్ణాంకం i = 0; i < V; i++) {
                ఒకవేళ (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            దశాంశం కూడిక = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                కూడిక += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            కొరకు (పూర్ణాంకం i = 0; i < V; i++) {
                probs_bt[i] /= కూడిక;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            కొరకు (పూర్ణాంకం i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

శూన్య crossentropy_forward(దశాంశం* losses,
                          దశాంశం* probs, పూర్ణాంకం* targets,
                          పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            // loss = -log(probs[target])
            దశాంశం* probs_bt = probs + b * T * Vp + t * Vp;
            పూర్ణాంకం ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

శూన్య crossentropy_softmax_backward(దశాంశం* dlogits,
                           దశాంశం* dlosses, దశాంశం* probs, పూర్ణాంకం* targets,
                           పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం V, పూర్ణాంకం Vp) {
    // backwards through both softmax and crossentropy
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            దశాంశం* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            దశాంశం* probs_bt = probs + b * T * Vp + t * Vp;
            దశాంశం dloss = dlosses[b * T + t];
            పూర్ణాంకం ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            కొరకు (పూర్ణాంకం i = 0; i < V; i++) {
                దశాంశం p = probs_bt[i];
                దశాంశం indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

రకనిర్వచనం నిర్మాణం {
    పూర్ణాంకం max_seq_len; // max sequence length, e.g. 1024
    పూర్ణాంకం vocab_కొలత; // vocab size, e.g. 50257
    పూర్ణాంకం padded_vocab_కొలత; // padded to e.g. %128==0, 50304
    పూర్ణాంకం అంకె_layers; // number of layers, e.g. 12
    పూర్ణాంకం అంకె_heads; // number of heads in attention, e.g. 12
    పూర్ణాంకం channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#నిర్వచించు అంకె_PARAMETER_TENSORS 16
రకనిర్వచనం నిర్మాణం {
    దశాంశం* wte; // (V, C)
    దశాంశం* wpe; // (maxT, C)
    దశాంశం* ln1w; // (L, C)
    దశాంశం* ln1b; // (L, C)
    దశాంశం* qkvw; // (L, 3*C, C)
    దశాంశం* qkvb; // (L, 3*C)
    దశాంశం* attprojw; // (L, C, C)
    దశాంశం* attprojb; // (L, C)
    దశాంశం* ln2w; // (L, C)
    దశాంశం* ln2b; // (L, C)
    దశాంశం* fcw; // (L, 4*C, C)
    దశాంశం* fcb; // (L, 4*C)
    దశాంశం* fcprojw; // (L, C, 4*C)
    దశాంశం* fcprojb; // (L, C)
    దశాంశం* lnfw; // (C)
    దశాంశం* lnfb; // (C)
} ParameterTensors;

శూన్య fill_లోపల_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_కొలత;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.అంకె_layers;
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
దశాంశం* malloc_and_బిందువు_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t అంకె_parameters = 0;
    కొరకు (size_t i = 0; i < అంకె_PARAMETER_TENSORS; i++) {
        అంకె_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    దశాంశం* params_జ్ఞాపకం = (దశాంశం*)mallocCheck(అంకె_parameters * పరిమాణం(దశాంశం));
    // assign all the tensors
    దశాంశం** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    దశాంశం* params_జ్ఞాపకం_iterator = params_జ్ఞాపకం;
    కొరకు (size_t i = 0; i < అంకె_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_జ్ఞాపకం_iterator;
        params_జ్ఞాపకం_iterator += param_sizes[i];
    }
    తిరిగి params_జ్ఞాపకం;
}

#నిర్వచించు అంకె_ACTIVATION_TENSORS 23
రకనిర్వచనం నిర్మాణం {
    దశాంశం* encoded; // (B, T, C)
    దశాంశం* ln1; // (L, B, T, C)
    దశాంశం* ln1_సగటు; // (L, B, T)
    దశాంశం* ln1_rstd; // (L, B, T)
    దశాంశం* qkv; // (L, B, T, 3*C)
    దశాంశం* atty; // (L, B, T, C)
    దశాంశం* preatt; // (L, B, NH, T, T)
    దశాంశం* att; // (L, B, NH, T, T)
    దశాంశం* attproj; // (L, B, T, C)
    దశాంశం* residual2; // (L, B, T, C)
    దశాంశం* ln2; // (L, B, T, C)
    దశాంశం* ln2_సగటు; // (L, B, T)
    దశాంశం* ln2_rstd; // (L, B, T)
    దశాంశం* fch; // (L, B, T, 4*C)
    దశాంశం* fch_gelu; // (L, B, T, 4*C)
    దశాంశం* fcproj; // (L, B, T, C)
    దశాంశం* residual3; // (L, B, T, C)
    దశాంశం* lnf; // (B, T, C)
    దశాంశం* lnf_సగటు; // (B, T)
    దశాంశం* lnf_rstd; // (B, T)
    దశాంశం* logits; // (B, T, V)
    దశాంశం* probs; // (B, T, V)
    దశాంశం* losses; // (B, T)
} ActivationTensors;

శూన్య fill_లోపల_activation_sizes(size_t* act_sizes, GPT2Config config, పూర్ణాంకం B, పూర్ణాంకం T) {
    size_t C = config.channels;
    size_t NH = config.అంకె_heads;
    size_t L = config.అంకె_layers;
    size_t Vp = config.padded_vocab_కొలత;
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

దశాంశం* malloc_and_బిందువు_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t అంకె_activations = 0;
    కొరకు (size_t i = 0; i < అంకె_ACTIVATION_TENSORS; i++) {
        అంకె_activations += act_sizes[i];
    }
    దశాంశం* acts_జ్ఞాపకం = (దశాంశం*)mallocCheck(అంకె_activations * పరిమాణం(దశాంశం));
    దశాంశం** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_సగటు, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_సగటు,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_సగటు, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    దశాంశం* acts_జ్ఞాపకం_iterator = acts_జ్ఞాపకం;
    కొరకు (size_t i = 0; i < అంకె_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_జ్ఞాపకం_iterator;
        acts_జ్ఞాపకం_iterator += act_sizes[i];
    }
    తిరిగి acts_జ్ఞాపకం;
}

రకనిర్వచనం నిర్మాణం {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[అంకె_PARAMETER_TENSORS];
    దశాంశం* params_జ్ఞాపకం;
    size_t అంకె_parameters;
    // gradients of the weights
    ParameterTensors grads;
    దశాంశం* grads_జ్ఞాపకం;
    // buffers for the AdamW optimizer
    దశాంశం* m_జ్ఞాపకం;
    దశాంశం* v_జ్ఞాపకం;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[అంకె_ACTIVATION_TENSORS];
    దశాంశం* acts_జ్ఞాపకం;
    size_t అంకె_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    దశాంశం* grads_acts_జ్ఞాపకం;
    // other run state configuration
    పూర్ణాంకం batch_కొలత; // the batch size (B) of current forward pass
    పూర్ణాంకం seq_len; // the sequence length (T) of current forward pass
    పూర్ణాంకం* inputs; // the input tokens for the current forward pass
    పూర్ణాంకం* targets; // the target tokens for the current forward pass
    దశాంశం సగటు_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

శూన్య gpt2_build_from_checkpoint(GPT2 *model, స్థిరరాశి అక్షరం* checkpoint_మార్గం) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_మార్గం, "rb");
    పూర్ణాంకం model_శీర్షిక[256];
    freadCheck(model_శీర్షిక, పరిమాణం(పూర్ణాంకం), 256, model_file);
    ఒకవేళ (model_శీర్షిక[0] != 20240326) { రాయి("Bad magic model file\n"); నిష్క్రమించు(1); }
    ఒకవేళ (model_శీర్షిక[1] != 3) {
        రాయి("Bad version in model file\n");
        రాయి("---> HINT: try to re-run `python train_gpt2.py`\n");
        నిష్క్రమించు(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_శీర్షిక[2];
    model->config.vocab_కొలత = V = model_శీర్షిక[3];
    model->config.అంకె_layers = L = model_శీర్షిక[4];
    model->config.అంకె_heads = NH = model_శీర్షిక[5];
    model->config.channels = C = model_శీర్షిక[6];
    model->config.padded_vocab_కొలత = Vp = model_శీర్షిక[7];
    రాయి("[GPT-2]\n");
    రాయి("max_seq_len: %zu\n", maxT);
    రాయి("vocab_size: %zu\n", V);
    రాయి("padded_vocab_size: %zu\n", Vp);
    రాయి("num_layers: %zu\n", L);
    రాయి("num_heads: %zu\n", NH);
    రాయి("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_లోపల_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t అంకె_parameters = 0;
    కొరకు (size_t i = 0; i < అంకె_PARAMETER_TENSORS; i++) {
        అంకె_parameters += model->param_sizes[i];
    }
    రాయి("num_parameters: %zu\n", అంకె_parameters);
    model->అంకె_parameters = అంకె_parameters;

    // read in all the parameters from file
    model->params_జ్ఞాపకం = malloc_and_బిందువు_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_జ్ఞాపకం, పరిమాణం(దశాంశం), అంకె_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_జ్ఞాపకం = ఖాళీ;
    model->grads_జ్ఞాపకం = ఖాళీ;
    model->m_జ్ఞాపకం = ఖాళీ;
    model->v_జ్ఞాపకం = ఖాళీ;
    model->grads_acts_జ్ఞాపకం = ఖాళీ;
    model->inputs = ఖాళీ;
    model->targets = ఖాళీ;
    model->batch_కొలత = 0;
    model->seq_len = 0;
    model->సగటు_loss = -1.0f; // -1.0f will designate no loss
}

శూన్య gpt2_forward(GPT2 *model, పూర్ణాంకం* inputs, పూర్ణాంకం* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    ఒకవేళ (model->params_జ్ఞాపకం == ఖాళీ) {
        రాయి("Error: model was not initialized properly.\n");
        నిష్క్రమించు(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_కొలత;
    size_t Vp = model->config.padded_vocab_కొలత;
    size_t L = model->config.అంకె_layers;
    size_t NH = model->config.అంకె_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    కొరకు(పూర్ణాంకం i = 0; i < B * T; i++) {
        ప్రతిజ్ఞా(0 <= inputs[i] && inputs[i] < V);
        ఒకవేళ (targets != ఖాళీ) {
            ప్రతిజ్ఞా(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    ఒకవేళ(model->acts_జ్ఞాపకం == ఖాళీ) {
        // record the current B,T as well
        model->batch_కొలత = B;
        model->seq_len = T;
        // and now allocate the space
        fill_లోపల_activation_sizes(model->act_sizes, model->config, B, T);
        size_t అంకె_activations = 0;
        కొరకు (size_t i = 0; i < అంకె_ACTIVATION_TENSORS; i++) {
            అంకె_activations += model->act_sizes[i];
        }
        రాయి("num_activations: %zu\n", అంకె_activations);
        model->అంకె_activations = అంకె_activations;
        model->acts_జ్ఞాపకం = malloc_and_బిందువు_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (పూర్ణాంకం*)mallocCheck(B * T * పరిమాణం(పూర్ణాంకం));
        model->targets = (పూర్ణాంకం*)mallocCheck(B * T * పరిమాణం(పూర్ణాంకం)); // might be unused if we never have targets but it's small
    } లేకపోతే {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        ఒకవేళ (B != model->batch_కొలత || T != model->seq_len) {
            రాయి("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_కొలత, model->seq_len, (పూర్ణాంకం)B, (పూర్ణాంకం)T);
            నిష్క్రమించు(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    జ్ఞాపకం_నకలు(model->inputs, inputs, B * T * పరిమాణం(పూర్ణాంకం));
    ఒకవేళ (targets != ఖాళీ) {
        జ్ఞాపకం_నకలు(model->targets, targets, B * T * పరిమాణం(పూర్ణాంకం));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    దశాంశం* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    కొరకు (పూర్ణాంకం l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        దశాంశం* l_ln1w = params.ln1w + l * C;
        దశాంశం* l_ln1b = params.ln1b + l * C;
        దశాంశం* l_qkvw = params.qkvw + l * 3*C * C;
        దశాంశం* l_qkvb = params.qkvb + l * 3*C;
        దశాంశం* l_attprojw = params.attprojw + l * C * C;
        దశాంశం* l_attprojb = params.attprojb + l * C;
        దశాంశం* l_ln2w = params.ln2w + l * C;
        దశాంశం* l_ln2b = params.ln2b + l * C;
        దశాంశం* l_fcw = params.fcw + l * 4*C * C;
        దశాంశం* l_fcb = params.fcb + l * 4*C;
        దశాంశం* l_fcprojw = params.fcprojw + l * C * 4*C;
        దశాంశం* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        దశాంశం* l_ln1 = acts.ln1 + l * B * T * C;
        దశాంశం* l_ln1_సగటు = acts.ln1_సగటు + l * B * T;
        దశాంశం* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        దశాంశం* l_qkv = acts.qkv + l * B * T * 3*C;
        దశాంశం* l_atty = acts.atty + l * B * T * C;
        దశాంశం* l_preatt = acts.preatt + l * B * NH * T * T;
        దశాంశం* l_att = acts.att + l * B * NH * T * T;
        దశాంశం* l_attproj = acts.attproj + l * B * T * C;
        దశాంశం* l_residual2 = acts.residual2 + l * B * T * C;
        దశాంశం* l_ln2 = acts.ln2 + l * B * T * C;
        దశాంశం* l_ln2_సగటు = acts.ln2_సగటు + l * B * T;
        దశాంశం* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        దశాంశం* l_fch = acts.fch + l * B * T * 4*C;
        దశాంశం* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        దశాంశం* l_fcproj = acts.fcproj + l * B * T * C;
        దశాంశం* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_సగటు, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_సగటు, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_సగటు, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, ఖాళీ, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    ఒకవేళ (targets != ఖాళీ) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        దశాంశం సగటు_loss = 0.0f;
        కొరకు (పూర్ణాంకం i=0; i<B*T; i++) { సగటు_loss += model->acts.losses[i]; }
        సగటు_loss /= B*T;
        model->సగటు_loss = సగటు_loss;
    } లేకపోతే {
        // if we don't have targets, we don't have a loss
        model->సగటు_loss = -1.0f;
    }
}

శూన్య gpt2_zero_grad(GPT2 *model) {
    ఒకవేళ(model->grads_జ్ఞాపకం != ఖాళీ) { memset(model->grads_జ్ఞాపకం, 0, model->అంకె_parameters * పరిమాణం(దశాంశం)); }
    ఒకవేళ(model->grads_acts_జ్ఞాపకం != ఖాళీ) { memset(model->grads_acts_జ్ఞాపకం, 0, model->అంకె_activations * పరిమాణం(దశాంశం)); }
}

శూన్య gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    ఒకవేళ (model->సగటు_loss == -1.0f) {
        రాయి("Error: must forward with targets before backward\n");
        నిష్క్రమించు(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    ఒకవేళ (model->grads_జ్ఞాపకం == ఖాళీ) {
        model->grads_జ్ఞాపకం = malloc_and_బిందువు_parameters(&model->grads, model->param_sizes);
        model->grads_acts_జ్ఞాపకం = malloc_and_బిందువు_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_కొలత;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_కొలత;
    size_t Vp = model->config.padded_vocab_కొలత;
    size_t L = model->config.అంకె_layers;
    size_t NH = model->config.అంకె_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    దశాంశం dloss_సగటు = 1.0f / (B*T);
    కొరకు (పూర్ణాంకం i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_సగటు; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, ఖాళీ, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    దశాంశం* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    దశాంశం* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_సగటు, acts.lnf_rstd, B, T, C);

    కొరకు (పూర్ణాంకం l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        దశాంశం* l_ln1w = params.ln1w + l * C;
        దశాంశం* l_qkvw = params.qkvw + l * 3*C * C;
        దశాంశం* l_attprojw = params.attprojw + l * C * C;
        దశాంశం* l_ln2w = params.ln2w + l * C;
        దశాంశం* l_fcw = params.fcw + l * 4*C * C;
        దశాంశం* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        దశాంశం* dl_ln1w = grads.ln1w + l * C;
        దశాంశం* dl_ln1b = grads.ln1b + l * C;
        దశాంశం* dl_qkvw = grads.qkvw + l * 3*C * C;
        దశాంశం* dl_qkvb = grads.qkvb + l * 3*C;
        దశాంశం* dl_attprojw = grads.attprojw + l * C * C;
        దశాంశం* dl_attprojb = grads.attprojb + l * C;
        దశాంశం* dl_ln2w = grads.ln2w + l * C;
        దశాంశం* dl_ln2b = grads.ln2b + l * C;
        దశాంశం* dl_fcw = grads.fcw + l * 4*C * C;
        దశాంశం* dl_fcb = grads.fcb + l * 4*C;
        దశాంశం* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        దశాంశం* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        దశాంశం* l_ln1 = acts.ln1 + l * B * T * C;
        దశాంశం* l_ln1_సగటు = acts.ln1_సగటు + l * B * T;
        దశాంశం* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        దశాంశం* l_qkv = acts.qkv + l * B * T * 3*C;
        దశాంశం* l_atty = acts.atty + l * B * T * C;
        దశాంశం* l_att = acts.att + l * B * NH * T * T;
        దశాంశం* l_residual2 = acts.residual2 + l * B * T * C;
        దశాంశం* l_ln2 = acts.ln2 + l * B * T * C;
        దశాంశం* l_ln2_సగటు = acts.ln2_సగటు + l * B * T;
        దశాంశం* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        దశాంశం* l_fch = acts.fch + l * B * T * 4*C;
        దశాంశం* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        దశాంశం* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        దశాంశం* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        దశాంశం* dl_atty = grads_acts.atty + l * B * T * C;
        దశాంశం* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        దశాంశం* dl_att = grads_acts.att + l * B * NH * T * T;
        దశాంశం* dl_attproj = grads_acts.attproj + l * B * T * C;
        దశాంశం* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        దశాంశం* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        దశాంశం* dl_fch = grads_acts.fch + l * B * T * 4*C;
        దశాంశం* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        దశాంశం* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        దశాంశం* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_సగటు, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_సగటు, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

శూన్య gpt2_update(GPT2 *model, దశాంశం learning_రేటు, దశాంశం beta1, దశాంశం beta2, దశాంశం eps, దశాంశం బరువు_decay, పూర్ణాంకం t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    ఒకవేళ (model->m_జ్ఞాపకం == ఖాళీ) {
        model->m_జ్ఞాపకం = (దశాంశం*)సున్నా_జ్ఞాపకం_ఇవ్వు(model->అంకె_parameters, పరిమాణం(దశాంశం));
        model->v_జ్ఞాపకం = (దశాంశం*)సున్నా_జ్ఞాపకం_ఇవ్వు(model->అంకె_parameters, పరిమాణం(దశాంశం));
    }

    కొరకు (size_t i = 0; i < model->అంకె_parameters; i++) {
        దశాంశం param = model->params_జ్ఞాపకం[i];
        దశాంశం grad = model->grads_జ్ఞాపకం[i];

        // update the first moment (momentum)
        దశాంశం m = beta1 * model->m_జ్ఞాపకం[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        దశాంశం v = beta2 * model->v_జ్ఞాపకం[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        దశాంశం m_hat = m / (1.0f - powf(beta1, t));
        దశాంశం v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_జ్ఞాపకం[i] = m;
        model->v_జ్ఞాపకం[i] = v;
        model->params_జ్ఞాపకం[i] -= learning_రేటు * (m_hat / (sqrtf(v_hat) + eps) + బరువు_decay * param);
    }
}

శూన్య gpt2_free(GPT2 *model) {
    విడుదల(model->params_జ్ఞాపకం);
    విడుదల(model->grads_జ్ఞాపకం);
    విడుదల(model->m_జ్ఞాపకం);
    విడుదల(model->v_జ్ఞాపకం);
    విడుదల(model->acts_జ్ఞాపకం);
    విడుదల(model->grads_acts_జ్ఞాపకం);
    విడుదల(model->inputs);
    విడుదల(model->targets);
}

#నిర్వచితం_కాకపోతే TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

గుర్తులేని పూర్ణాంకం random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    తిరిగి (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
దశాంశం random_f32(uint64_t *state) { // random float32 in [0,1)
    తిరిగి (random_u32(state) >> 8) / 16777216.0f;
}

పూర్ణాంకం sample_mult(దశాంశం* probabilities, పూర్ణాంకం n, దశాంశం coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    దశాంశం cdf = 0.0f;
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        cdf += probabilities[i];
        ఒకవేళ (coin < cdf) {
            తిరిగి i;
        }
    }
    తిరిగి n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
పూర్ణాంకం ప్రధాన() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    స్థిరరాశి అక్షరం* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    స్థిరరాశి అక్షరం* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    స్థిరరాశి అక్షరం* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    స్థిరరాశి అక్షరం* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    స్థిరరాశి అక్షరం* train_టోకెన్లు = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    స్థిరరాశి అక్షరం* val_టోకెన్లు = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    పూర్ణాంకం B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    పూర్ణాంకం T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    దత్తాంశం_Loader train_loader, val_loader;
    dataloader_init(&train_loader, train_టోకెన్లు, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_టోకెన్లు, B, T, 0, 1, 0);
    రాయి("train dataset num_batches: %zu\n", train_loader.అంకె_టోకెన్లు / (B*T));
    రాయి("val dataset num_batches: %zu\n", val_loader.అంకె_టోకెన్లు / (B*T));
    పూర్ణాంకం val_అంకె_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    పూర్ణాంకం* gen_టోకెన్లు = (పూర్ణాంకం*)mallocCheck(B * T * పరిమాణం(పూర్ణాంకం));
    స్థిరరాశి పూర్ణాంకం genT = 64; // number of steps of inference we will do

    // train
    నిర్మాణం timespec ప్రారంభం, చివర;
    కొరకు (పూర్ణాంకం అడుగు = 0; అడుగు <= 40; అడుగు++) {

        // once in a while estimate the validation loss
        ఒకవేళ (అడుగు % 10 == 0) {
            దశాంశం val_loss = 0.0f;
            dataloader_reset(&val_loader);
            కొరకు (పూర్ణాంకం i = 0; i < val_అంకె_batches; i++) {
                dataloader_తదుపరి_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.సగటు_loss;
            }
            val_loss /= val_అంకె_batches;
            రాయి("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        ఒకవేళ (అడుగు > 0 && అడుగు % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            కొరకు(పూర్ణాంకం i = 0; i < B * T; ++i) {
                gen_టోకెన్లు[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            రాయి("generating:\n---\n");
            కొరకు (పూర్ణాంకం t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_టోకెన్లు, ఖాళీ, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                దశాంశం* probs = model.acts.probs + (t-1) * model.config.padded_vocab_కొలత;
                దశాంశం coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                పూర్ణాంకం తదుపరి_token = sample_mult(probs, model.config.vocab_కొలత, coin);
                gen_టోకెన్లు[t] = తదుపరి_token;
                // print the generated token, either using the Tokenizer or a fallback
                ఒకవేళ (tokenizer.init_ok) {
                    స్థిరరాశి అక్షరం* token_str = tokenizer_decode(&tokenizer, తదుపరి_token);
                    సురక్షిత_printf(token_str);
                } లేకపోతే {
                    // fall back to printing the token id
                    రాయి("%d ", తదుపరి_token);
                }
                fflush(stdout);
            }
            రాయి("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &ప్రారంభం);
        dataloader_తదుపరి_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, అడుగు+1);
        clock_gettime(CLOCK_MONOTONIC, &చివర);
        ద్విగుణ సమయం_elapsed_s = (చివర.tv_sec - ప్రారంభం.tv_sec) + (చివర.tv_nsec - ప్రారంభం.tv_nsec) / 1e9;
        రాయి("step %d: train loss %f (took %f ms)\n", అడుగు, model.సగటు_loss, సమయం_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    విడుదల(gen_టోకెన్లు);
    తిరిగి 0;
}
#అయితే_ముగింపు
