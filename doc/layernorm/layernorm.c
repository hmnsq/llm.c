// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <math.h>

فراغ layernorm_forward(عائم* out, عائم* mean, عائم* rstd,
                       عائم* inp, عائم* weight, عائم* bias,
                       صحيح B, صحيح T, صحيح C) {
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
            // calculate the rstd
            عائم s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            عائم* out_bt = out + b * T * C + t * C;
            لكل (صحيح i = 0; i < C; i++) {
                عائم n = (s * (x[i] - m)); // normalized output
                عائم o = n * weight[i] + bias[i]; // scale and shift it
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

// poor man's tensor checker
صحيح check_tensor(عائم *a, عائم *b, صحيح n, حرف* label) {
    صحيح ok = 1;
    printf("%s\n", label);
    لكل (صحيح i = 0; i < n; i++) {
        إذا (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } وإلا {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    إرجاع ok;
}

صحيح الرئيسية() {

    صحيح B = 2; // batch
    صحيح T = 3; // time / sequence length
    صحيح C = 4; // number of channels

    عائم* x = (عائم*) malloc(B * T * C * حجم(عائم));
    عائم* w = (عائم*) malloc(C * حجم(عائم));
    عائم* b = (عائم*) malloc(C * حجم(عائم));
    عائم* out = (عائم*) malloc(B * T * C * حجم(عائم));
    عائم* mean = (عائم*) malloc(B * T * حجم(عائم));
    عائم* rstd = (عائم*) malloc(B * T * حجم(عائم));
    عائم* dout = (عائم*) malloc(B * T * C * حجم(عائم));
    عائم* dx = (عائم*) malloc(B * T * C * حجم(عائم));
    عائم* dw = (عائم*) malloc(C * حجم(عائم));
    عائم* db = (عائم*) malloc(C * حجم(عائم));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    إذا (file == عدم) {
        printf("Error opening file\n");
        إرجاع 1;
    }
    fread(x, حجم(عائم), B * T * C, file);
    fread(w, حجم(عائم), C, file);
    fread(b, حجم(عائم), C, file);
    fread(out, حجم(عائم), B * T * C, file);
    fread(mean, حجم(عائم), B * T, file);
    fread(rstd, حجم(عائم), B * T, file);
    fread(dout, حجم(عائم), B * T * C, file);
    fread(dx, حجم(عائم), B * T * C, file);
    fread(dw, حجم(عائم), C, file);
    fread(db, حجم(عائم), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    عائم* c_out = (عائم*) malloc(B * T * C * حجم(عائم));
    عائم* c_mean = (عائم*) malloc(B * T * حجم(عائم));
    عائم* c_rstd = (عائم*) malloc(B * T * حجم(عائم));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    عائم* c_dx = (عائم*) calloc(B * T * C, حجم(عائم));
    عائم* c_dw = (عائم*) calloc(B * T, حجم(عائم));
    عائم* c_db = (عائم*) calloc(B * T, حجم(عائم));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    حرر(x);
    حرر(w);
    حرر(b);
    حرر(out);
    حرر(mean);
    حرر(rstd);
    حرر(dout);
    حرر(dx);
    حرر(dw);
    حرر(db);
    إرجاع 0;
}
