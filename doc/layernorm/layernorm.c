// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <math.h>

শূন্য layernorm_forward(ভিন্ন* out, ভিন্ন* mean, ভিন্ন* rstd,
                       ভিন্ন* inp, ভিন্ন* weight, ভিন্ন* bias,
                       পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক C) {
    ভিন্ন eps = 1e-5f;
    বাবে (পূর্ণাঙ্ক b = 0; b < B; b++) {
        বাবে (পূর্ণাঙ্ক t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ভিন্ন* x = inp + b * T * C + t * C;
            // calculate the mean
            ভিন্ন m = 0.0f;
            বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ভিন্ন v = 0.0f;
            বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                ভিন্ন xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            ভিন্ন s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ভিন্ন* out_bt = out + b * T * C + t * C;
            বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                ভিন্ন n = (s * (x[i] - m)); // normalized output
                ভিন্ন o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

শূন্য layernorm_backward(ভিন্ন* dinp, ভিন্ন* dweight, ভিন্ন* dbias,
                        ভিন্ন* dout, ভিন্ন* inp, ভিন্ন* weight, ভিন্ন* mean, ভিন্ন* rstd,
                        পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক C) {
    বাবে (পূর্ণাঙ্ক b = 0; b < B; b++) {
        বাবে (পূর্ণাঙ্ক t = 0; t < T; t++) {
            ভিন্ন* dout_bt = dout + b * T * C + t * C;
            ভিন্ন* inp_bt = inp + b * T * C + t * C;
            ভিন্ন* dinp_bt = dinp + b * T * C + t * C;
            ভিন্ন mean_bt = mean[b * T + t];
            ভিন্ন rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ভিন্ন dnorm_mean = 0.0f;
            ভিন্ন dnorm_norm_mean = 0.0f;
            বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                ভিন্ন norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ভিন্ন dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                ভিন্ন norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                ভিন্ন dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ভিন্ন dval = 0.0f;
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
পূর্ণাঙ্ক check_tensor(ভিন্ন *a, ভিন্ন *b, পূর্ণাঙ্ক n, ৱর্ণ* label) {
    পূর্ণাঙ্ক ok = 1;
    printf("%s\n", label);
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        যদি (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } অন্যথা {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    ঘূৰাই ok;
}

পূর্ণাঙ্ক main() {

    পূর্ণাঙ্ক B = 2; // batch
    পূর্ণাঙ্ক T = 3; // time / sequence length
    পূর্ণাঙ্ক C = 4; // number of channels

    ভিন্ন* x = (ভিন্ন*) malloc(B * T * C * পরিমাণ(ভিন্ন));
    ভিন্ন* w = (ভিন্ন*) malloc(C * পরিমাণ(ভিন্ন));
    ভিন্ন* b = (ভিন্ন*) malloc(C * পরিমাণ(ভিন্ন));
    ভিন্ন* out = (ভিন্ন*) malloc(B * T * C * পরিমাণ(ভিন্ন));
    ভিন্ন* mean = (ভিন্ন*) malloc(B * T * পরিমাণ(ভিন্ন));
    ভিন্ন* rstd = (ভিন্ন*) malloc(B * T * পরিমাণ(ভিন্ন));
    ভিন্ন* dout = (ভিন্ন*) malloc(B * T * C * পরিমাণ(ভিন্ন));
    ভিন্ন* dx = (ভিন্ন*) malloc(B * T * C * পরিমাণ(ভিন্ন));
    ভিন্ন* dw = (ভিন্ন*) malloc(C * পরিমাণ(ভিন্ন));
    ভিন্ন* db = (ভিন্ন*) malloc(C * পরিমাণ(ভিন্ন));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    যদি (file == রিক্ত) {
        printf("Error opening file\n");
        ঘূৰাই 1;
    }
    fread(x, পরিমাণ(ভিন্ন), B * T * C, file);
    fread(w, পরিমাণ(ভিন্ন), C, file);
    fread(b, পরিমাণ(ভিন্ন), C, file);
    fread(out, পরিমাণ(ভিন্ন), B * T * C, file);
    fread(mean, পরিমাণ(ভিন্ন), B * T, file);
    fread(rstd, পরিমাণ(ভিন্ন), B * T, file);
    fread(dout, পরিমাণ(ভিন্ন), B * T * C, file);
    fread(dx, পরিমাণ(ভিন্ন), B * T * C, file);
    fread(dw, পরিমাণ(ভিন্ন), C, file);
    fread(db, পরিমাণ(ভিন্ন), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    ভিন্ন* c_out = (ভিন্ন*) malloc(B * T * C * পরিমাণ(ভিন্ন));
    ভিন্ন* c_mean = (ভিন্ন*) malloc(B * T * পরিমাণ(ভিন্ন));
    ভিন্ন* c_rstd = (ভিন্ন*) malloc(B * T * পরিমাণ(ভিন্ন));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    ভিন্ন* c_dx = (ভিন্ন*) calloc(B * T * C, পরিমাণ(ভিন্ন));
    ভিন্ন* c_dw = (ভিন্ন*) calloc(B * T, পরিমাণ(ভিন্ন));
    ভিন্ন* c_db = (ভিন্ন*) calloc(B * T, পরিমাণ(ভিন্ন));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    মুক্ত(x);
    মুক্ত(w);
    মুক্ত(b);
    মুক্ত(out);
    মুক্ত(mean);
    মুক্ত(rstd);
    মুক্ত(dout);
    মুক্ত(dx);
    মুক্ত(dw);
    মুক্ত(db);
    ঘূৰাই 0;
}
