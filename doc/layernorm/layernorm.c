// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <math.h>

ಶೂನ್ಯ layernorm_forward(ಭಿನ್ನ* out, ಭಿನ್ನ* mean, ಭಿನ್ನ* rstd,
                       ಭಿನ್ನ* inp, ಭಿನ್ನ* weight, ಭಿನ್ನ* bias,
                       ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ C) {
    ಭಿನ್ನ eps = 1e-5f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < B; b++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ಭಿನ್ನ* x = inp + b * T * C + t * C;
            // calculate the mean
            ಭಿನ್ನ m = 0.0f;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ಭಿನ್ನ v = 0.0f;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                ಭಿನ್ನ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            ಭಿನ್ನ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ಭಿನ್ನ* out_bt = out + b * T * C + t * C;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                ಭಿನ್ನ n = (s * (x[i] - m)); // normalized output
                ಭಿನ್ನ o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ಶೂನ್ಯ layernorm_backward(ಭಿನ್ನ* dinp, ಭಿನ್ನ* dweight, ಭಿನ್ನ* dbias,
                        ಭಿನ್ನ* dout, ಭಿನ್ನ* inp, ಭಿನ್ನ* weight, ಭಿನ್ನ* mean, ಭಿನ್ನ* rstd,
                        ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ C) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < B; b++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < T; t++) {
            ಭಿನ್ನ* dout_bt = dout + b * T * C + t * C;
            ಭಿನ್ನ* inp_bt = inp + b * T * C + t * C;
            ಭಿನ್ನ* dinp_bt = dinp + b * T * C + t * C;
            ಭಿನ್ನ mean_bt = mean[b * T + t];
            ಭಿನ್ನ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ಭಿನ್ನ dnorm_mean = 0.0f;
            ಭಿನ್ನ dnorm_norm_mean = 0.0f;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                ಭಿನ್ನ norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ಭಿನ್ನ dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                ಭಿನ್ನ norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ಭಿನ್ನ dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ಭಿನ್ನ dval = 0.0f;
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
ಪೂರ್ಣಾಙ್ಕ check_tensor(ಭಿನ್ನ *a, ಭಿನ್ನ *b, ಪೂರ್ಣಾಙ್ಕ n, ವರ್ಣ* label) {
    ಪೂರ್ಣಾಙ್ಕ ok = 1;
    printf("%s\n", label);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        ಯದಿ (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } ಅನ್ಯಥಾ {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    ಪ್ರತ್ಯಾಗಮ ok;
}

ಪೂರ್ಣಾಙ್ಕ main() {

    ಪೂರ್ಣಾಙ್ಕ B = 2; // batch
    ಪೂರ್ಣಾಙ್ಕ T = 3; // time / sequence length
    ಪೂರ್ಣಾಙ್ಕ C = 4; // number of channels

    ಭಿನ್ನ* x = (ಭಿನ್ನ*) malloc(B * T * C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* w = (ಭಿನ್ನ*) malloc(C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* b = (ಭಿನ್ನ*) malloc(C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* out = (ಭಿನ್ನ*) malloc(B * T * C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* mean = (ಭಿನ್ನ*) malloc(B * T * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* rstd = (ಭಿನ್ನ*) malloc(B * T * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* dout = (ಭಿನ್ನ*) malloc(B * T * C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* dx = (ಭಿನ್ನ*) malloc(B * T * C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* dw = (ಭಿನ್ನ*) malloc(C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* db = (ಭಿನ್ನ*) malloc(C * ಪರಿಮಾಣ(ಭಿನ್ನ));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    ಯದಿ (file == ರಿಕ್ತ) {
        printf("Error opening file\n");
        ಪ್ರತ್ಯಾಗಮ 1;
    }
    fread(x, ಪರಿಮಾಣ(ಭಿನ್ನ), B * T * C, file);
    fread(w, ಪರಿಮಾಣ(ಭಿನ್ನ), C, file);
    fread(b, ಪರಿಮಾಣ(ಭಿನ್ನ), C, file);
    fread(out, ಪರಿಮಾಣ(ಭಿನ್ನ), B * T * C, file);
    fread(mean, ಪರಿಮಾಣ(ಭಿನ್ನ), B * T, file);
    fread(rstd, ಪರಿಮಾಣ(ಭಿನ್ನ), B * T, file);
    fread(dout, ಪರಿಮಾಣ(ಭಿನ್ನ), B * T * C, file);
    fread(dx, ಪರಿಮಾಣ(ಭಿನ್ನ), B * T * C, file);
    fread(dw, ಪರಿಮಾಣ(ಭಿನ್ನ), C, file);
    fread(db, ಪರಿಮಾಣ(ಭಿನ್ನ), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    ಭಿನ್ನ* c_out = (ಭಿನ್ನ*) malloc(B * T * C * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* c_mean = (ಭಿನ್ನ*) malloc(B * T * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* c_rstd = (ಭಿನ್ನ*) malloc(B * T * ಪರಿಮಾಣ(ಭಿನ್ನ));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    ಭಿನ್ನ* c_dx = (ಭಿನ್ನ*) calloc(B * T * C, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* c_dw = (ಭಿನ್ನ*) calloc(B * T, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* c_db = (ಭಿನ್ನ*) calloc(B * T, ಪರಿಮಾಣ(ಭಿನ್ನ));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    ಮುಕ್ತ(x);
    ಮುಕ್ತ(w);
    ಮುಕ್ತ(b);
    ಮುಕ್ತ(out);
    ಮುಕ್ತ(mean);
    ಮುಕ್ತ(rstd);
    ಮುಕ್ತ(dout);
    ಮುಕ್ತ(dx);
    ಮುಕ್ತ(dw);
    ಮುಕ್ತ(db);
    ಪ್ರತ್ಯಾಗಮ 0;
}
