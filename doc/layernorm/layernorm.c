// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <math.h>

ശൂന്യം layernorm_forward(ദശാംശം* out, ദശാംശം* mean, ദശാംശം* rstd,
                       ദശാംശം* inp, ദശാംശം* weight, ദശാംശം* bias,
                       പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ C) {
    ദശാംശം eps = 1e-5f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < B; b++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ദശാംശം* x = inp + b * T * C + t * C;
            // calculate the mean
            ദശാംശം m = 0.0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ദശാംശം v = 0.0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                ദശാംശം xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            ദശാംശം s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ദശാംശം* out_bt = out + b * T * C + t * C;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                ദശാംശം n = (s * (x[i] - m)); // normalized output
                ദശാംശം o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ശൂന്യം layernorm_backward(ദശാംശം* dinp, ദശാംശം* dweight, ദശാംശം* dbias,
                        ദശാംശം* dout, ദശാംശം* inp, ദശാംശം* weight, ദശാംശം* mean, ദശാംശം* rstd,
                        പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ C) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < B; b++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < T; t++) {
            ദശാംശം* dout_bt = dout + b * T * C + t * C;
            ദശാംശം* inp_bt = inp + b * T * C + t * C;
            ദശാംശം* dinp_bt = dinp + b * T * C + t * C;
            ദശാംശം mean_bt = mean[b * T + t];
            ദശാംശം rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ദശാംശം dnorm_mean = 0.0f;
            ദശാംശം dnorm_norm_mean = 0.0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                ദശാംശം norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ദശാംശം dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                ദശാംശം norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ദശാംശം dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ദശാംശം dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
പൂർണ്ണസംഖ്യ check_tensor(ദശാംശം *a, ദശാംശം *b, പൂർണ്ണസംഖ്യ n, അക്ഷരം* label) {
    പൂർണ്ണസംഖ്യ ok = 1;
    printf("%s\n", label);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        എങ്കിൽ (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } അല്ലെങ്കിൽ {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    തിരികെ ok;
}

പൂർണ്ണസംഖ്യ പ്രധാന() {

    പൂർണ്ണസംഖ്യ B = 2; // batch
    പൂർണ്ണസംഖ്യ T = 3; // time / sequence length
    പൂർണ്ണസംഖ്യ C = 4; // number of channels

    ദശാംശം* x = (ദശാംശം*) malloc(B * T * C * വലുപ്പം(ദശാംശം));
    ദശാംശം* w = (ദശാംശം*) malloc(C * വലുപ്പം(ദശാംശം));
    ദശാംശം* b = (ദശാംശം*) malloc(C * വലുപ്പം(ദശാംശം));
    ദശാംശം* out = (ദശാംശം*) malloc(B * T * C * വലുപ്പം(ദശാംശം));
    ദശാംശം* mean = (ദശാംശം*) malloc(B * T * വലുപ്പം(ദശാംശം));
    ദശാംശം* rstd = (ദശാംശം*) malloc(B * T * വലുപ്പം(ദശാംശം));
    ദശാംശം* dout = (ദശാംശം*) malloc(B * T * C * വലുപ്പം(ദശാംശം));
    ദശാംശം* dx = (ദശാംശം*) malloc(B * T * C * വലുപ്പം(ദശാംശം));
    ദശാംശം* dw = (ദശാംശം*) malloc(C * വലുപ്പം(ദശാംശം));
    ദശാംശം* db = (ദശാംശം*) malloc(C * വലുപ്പം(ദശാംശം));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    എങ്കിൽ (file == ശൂന്യ) {
        printf("Error opening file\n");
        തിരികെ 1;
    }
    fread(x, വലുപ്പം(ദശാംശം), B * T * C, file);
    fread(w, വലുപ്പം(ദശാംശം), C, file);
    fread(b, വലുപ്പം(ദശാംശം), C, file);
    fread(out, വലുപ്പം(ദശാംശം), B * T * C, file);
    fread(mean, വലുപ്പം(ദശാംശം), B * T, file);
    fread(rstd, വലുപ്പം(ദശാംശം), B * T, file);
    fread(dout, വലുപ്പം(ദശാംശം), B * T * C, file);
    fread(dx, വലുപ്പം(ദശാംശം), B * T * C, file);
    fread(dw, വലുപ്പം(ദശാംശം), C, file);
    fread(db, വലുപ്പം(ദശാംശം), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    ദശാംശം* c_out = (ദശാംശം*) malloc(B * T * C * വലുപ്പം(ദശാംശം));
    ദശാംശം* c_mean = (ദശാംശം*) malloc(B * T * വലുപ്പം(ദശാംശം));
    ദശാംശം* c_rstd = (ദശാംശം*) malloc(B * T * വലുപ്പം(ദശാംശം));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    ദശാംശം* c_dx = (ദശാംശം*) calloc(B * T * C, വലുപ്പം(ദശാംശം));
    ദശാംശം* c_dw = (ദശാംശം*) calloc(B * T, വലുപ്പം(ദശാംശം));
    ദശാംശം* c_db = (ദശാംശം*) calloc(B * T, വലുപ്പം(ദശാംശം));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    സ്വതന്ത്രം(x);
    സ്വതന്ത്രം(w);
    സ്വതന്ത്രം(b);
    സ്വതന്ത്രം(out);
    സ്വതന്ത്രം(mean);
    സ്വതന്ത്രം(rstd);
    സ്വതന്ത്രം(dout);
    സ്വതന്ത്രം(dx);
    സ്വതന്ത്രം(dw);
    സ്വതന്ത്രം(db);
    തിരികെ 0;
}
