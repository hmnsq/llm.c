// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <math.h>

ଶୂନ୍ୟ layernorm_forward(ଦଶମିକ* out, ଦଶମିକ* mean, ଦଶମିକ* rstd,
                       ଦଶମିକ* inp, ଦଶମିକ* weight, ଦଶମିକ* bias,
                       ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C) {
    ଦଶମିକ eps = 1e-5f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < B; b++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ଦଶମିକ* x = inp + b * T * C + t * C;
            // calculate the mean
            ଦଶମିକ m = 0.0f;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ଦଶମିକ v = 0.0f;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                ଦଶମିକ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            ଦଶମିକ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ଦଶମିକ* out_bt = out + b * T * C + t * C;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                ଦଶମିକ n = (s * (x[i] - m)); // normalized output
                ଦଶମିକ o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ଶୂନ୍ୟ layernorm_backward(ଦଶମିକ* dinp, ଦଶମିକ* dweight, ଦଶମିକ* dbias,
                        ଦଶମିକ* dout, ଦଶମିକ* inp, ଦଶମିକ* weight, ଦଶମିକ* mean, ଦଶମିକ* rstd,
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < B; b++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < T; t++) {
            ଦଶମିକ* dout_bt = dout + b * T * C + t * C;
            ଦଶମିକ* inp_bt = inp + b * T * C + t * C;
            ଦଶମିକ* dinp_bt = dinp + b * T * C + t * C;
            ଦଶମିକ mean_bt = mean[b * T + t];
            ଦଶମିକ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ଦଶମିକ dnorm_mean = 0.0f;
            ଦଶମିକ dnorm_norm_mean = 0.0f;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                ଦଶମିକ norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ଦଶମିକ dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                ଦଶମିକ norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ଦଶମିକ dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ଦଶମିକ dval = 0.0f;
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
ପୂର୍ଣ୍ଣସଂଖ୍ୟା check_tensor(ଦଶମିକ *a, ଦଶମିକ *b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅକ୍ଷର* label) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ok = 1;
    printf("%s\n", label);
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ଯଦି (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } ଅନ୍ୟଥା {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    ଫେରନ୍ତୁ ok;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୁଖ୍ୟ() {

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B = 2; // batch
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା T = 3; // time / sequence length
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା C = 4; // number of channels

    ଦଶମିକ* x = (ଦଶମିକ*) malloc(B * T * C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* w = (ଦଶମିକ*) malloc(C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* b = (ଦଶମିକ*) malloc(C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* out = (ଦଶମିକ*) malloc(B * T * C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* mean = (ଦଶମିକ*) malloc(B * T * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* rstd = (ଦଶମିକ*) malloc(B * T * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* dout = (ଦଶମିକ*) malloc(B * T * C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* dx = (ଦଶମିକ*) malloc(B * T * C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* dw = (ଦଶମିକ*) malloc(C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* db = (ଦଶମିକ*) malloc(C * ଆକାର(ଦଶମିକ));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    ଯଦି (file == ଖାଲି) {
        printf("Error opening file\n");
        ଫେରନ୍ତୁ 1;
    }
    fread(x, ଆକାର(ଦଶମିକ), B * T * C, file);
    fread(w, ଆକାର(ଦଶମିକ), C, file);
    fread(b, ଆକାର(ଦଶମିକ), C, file);
    fread(out, ଆକାର(ଦଶମିକ), B * T * C, file);
    fread(mean, ଆକାର(ଦଶମିକ), B * T, file);
    fread(rstd, ଆକାର(ଦଶମିକ), B * T, file);
    fread(dout, ଆକାର(ଦଶମିକ), B * T * C, file);
    fread(dx, ଆକାର(ଦଶମିକ), B * T * C, file);
    fread(dw, ଆକାର(ଦଶମିକ), C, file);
    fread(db, ଆକାର(ଦଶମିକ), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    ଦଶମିକ* c_out = (ଦଶମିକ*) malloc(B * T * C * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* c_mean = (ଦଶମିକ*) malloc(B * T * ଆକାର(ଦଶମିକ));
    ଦଶମିକ* c_rstd = (ଦଶମିକ*) malloc(B * T * ଆକାର(ଦଶମିକ));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    ଦଶମିକ* c_dx = (ଦଶମିକ*) calloc(B * T * C, ଆକାର(ଦଶମିକ));
    ଦଶମିକ* c_dw = (ଦଶମିକ*) calloc(B * T, ଆକାର(ଦଶମିକ));
    ଦଶମିକ* c_db = (ଦଶମିକ*) calloc(B * T, ଆକାର(ଦଶମିକ));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    ମୁକ୍ତ(x);
    ମୁକ୍ତ(w);
    ମୁକ୍ତ(b);
    ମୁକ୍ତ(out);
    ମୁକ୍ତ(mean);
    ମୁକ୍ତ(rstd);
    ମୁକ୍ତ(dout);
    ମୁକ୍ତ(dx);
    ମୁକ୍ତ(dw);
    ମୁକ୍ତ(db);
    ଫେରନ୍ତୁ 0;
}
