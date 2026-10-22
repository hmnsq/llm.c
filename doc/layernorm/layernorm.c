// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <math.h>

શૂન્ય layernorm_forward(દશાંશ* out, દશાંશ* mean, દશાંશ* rstd,
                       દશાંશ* inp, દશાંશ* weight, દશાંશ* bias,
                       પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક C) {
    દશાંશ eps = 1e-5f;
    માટે (પૂર્ણાંક b = 0; b < B; b++) {
        માટે (પૂર્ણાંક t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            દશાંશ* x = inp + b * T * C + t * C;
            // calculate the mean
            દશાંશ m = 0.0f;
            માટે (પૂર્ણાંક i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            દશાંશ v = 0.0f;
            માટે (પૂર્ણાંક i = 0; i < C; i++) {
                દશાંશ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            દશાંશ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            દશાંશ* out_bt = out + b * T * C + t * C;
            માટે (પૂર્ણાંક i = 0; i < C; i++) {
                દશાંશ n = (s * (x[i] - m)); // normalized output
                દશાંશ o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

શૂન્ય layernorm_backward(દશાંશ* dinp, દશાંશ* dweight, દશાંશ* dbias,
                        દશાંશ* dout, દશાંશ* inp, દશાંશ* weight, દશાંશ* mean, દશાંશ* rstd,
                        પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક C) {
    માટે (પૂર્ણાંક b = 0; b < B; b++) {
        માટે (પૂર્ણાંક t = 0; t < T; t++) {
            દશાંશ* dout_bt = dout + b * T * C + t * C;
            દશાંશ* inp_bt = inp + b * T * C + t * C;
            દશાંશ* dinp_bt = dinp + b * T * C + t * C;
            દશાંશ mean_bt = mean[b * T + t];
            દશાંશ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            દશાંશ dnorm_mean = 0.0f;
            દશાંશ dnorm_norm_mean = 0.0f;
            માટે (પૂર્ણાંક i = 0; i < C; i++) {
                દશાંશ norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                દશાંશ dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            માટે (પૂર્ણાંક i = 0; i < C; i++) {
                દશાંશ norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                દશાંશ dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                દશાંશ dval = 0.0f;
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
પૂર્ણાંક check_tensor(દશાંશ *a, દશાંશ *b, પૂર્ણાંક n, અક્ષર* label) {
    પૂર્ણાંક ok = 1;
    printf("%s\n", label);
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        જો (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } નહીંતર {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    પાછા ok;
}

પૂર્ણાંક મુખ્ય() {

    પૂર્ણાંક B = 2; // batch
    પૂર્ણાંક T = 3; // time / sequence length
    પૂર્ણાંક C = 4; // number of channels

    દશાંશ* x = (દશાંશ*) malloc(B * T * C * કદ(દશાંશ));
    દશાંશ* w = (દશાંશ*) malloc(C * કદ(દશાંશ));
    દશાંશ* b = (દશાંશ*) malloc(C * કદ(દશાંશ));
    દશાંશ* out = (દશાંશ*) malloc(B * T * C * કદ(દશાંશ));
    દશાંશ* mean = (દશાંશ*) malloc(B * T * કદ(દશાંશ));
    દશાંશ* rstd = (દશાંશ*) malloc(B * T * કદ(દશાંશ));
    દશાંશ* dout = (દશાંશ*) malloc(B * T * C * કદ(દશાંશ));
    દશાંશ* dx = (દશાંશ*) malloc(B * T * C * કદ(દશાંશ));
    દશાંશ* dw = (દશાંશ*) malloc(C * કદ(દશાંશ));
    દશાંશ* db = (દશાંશ*) malloc(C * કદ(દશાંશ));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    જો (file == ખાલી) {
        printf("Error opening file\n");
        પાછા 1;
    }
    fread(x, કદ(દશાંશ), B * T * C, file);
    fread(w, કદ(દશાંશ), C, file);
    fread(b, કદ(દશાંશ), C, file);
    fread(out, કદ(દશાંશ), B * T * C, file);
    fread(mean, કદ(દશાંશ), B * T, file);
    fread(rstd, કદ(દશાંશ), B * T, file);
    fread(dout, કદ(દશાંશ), B * T * C, file);
    fread(dx, કદ(દશાંશ), B * T * C, file);
    fread(dw, કદ(દશાંશ), C, file);
    fread(db, કદ(દશાંશ), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    દશાંશ* c_out = (દશાંશ*) malloc(B * T * C * કદ(દશાંશ));
    દશાંશ* c_mean = (દશાંશ*) malloc(B * T * કદ(દશાંશ));
    દશાંશ* c_rstd = (દશાંશ*) malloc(B * T * કદ(દશાંશ));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    દશાંશ* c_dx = (દશાંશ*) calloc(B * T * C, કદ(દશાંશ));
    દશાંશ* c_dw = (દશાંશ*) calloc(B * T, કદ(દશાંશ));
    દશાંશ* c_db = (દશાંશ*) calloc(B * T, કદ(દશાંશ));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    મુક્ત(x);
    મુક્ત(w);
    મુક્ત(b);
    મુક્ત(out);
    મુક્ત(mean);
    મુક્ત(rstd);
    મુક્ત(dout);
    મુક્ત(dx);
    મુક્ત(dw);
    મુક્ત(db);
    પાછા 0;
}
