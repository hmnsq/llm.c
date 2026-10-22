// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <math.h>

تهی layernorm_forward(اعشاری* out, اعشاری* mean, اعشاری* rstd,
                       اعشاری* inp, اعشاری* weight, اعشاری* bias,
                       صحیح B, صحیح T, صحیح C) {
    اعشاری eps = 1e-5f;
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            اعشاری* x = inp + b * T * C + t * C;
            // calculate the mean
            اعشاری m = 0.0f;
            برای (صحیح i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            اعشاری v = 0.0f;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            اعشاری s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            اعشاری* out_bt = out + b * T * C + t * C;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری n = (s * (x[i] - m)); // normalized output
                اعشاری o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

تهی layernorm_backward(اعشاری* dinp, اعشاری* dweight, اعشاری* dbias,
                        اعشاری* dout, اعشاری* inp, اعشاری* weight, اعشاری* mean, اعشاری* rstd,
                        صحیح B, صحیح T, صحیح C) {
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            اعشاری* dout_bt = dout + b * T * C + t * C;
            اعشاری* inp_bt = inp + b * T * C + t * C;
            اعشاری* dinp_bt = dinp + b * T * C + t * C;
            اعشاری mean_bt = mean[b * T + t];
            اعشاری rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            اعشاری dnorm_mean = 0.0f;
            اعشاری dnorm_norm_mean = 0.0f;
            برای (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                اعشاری dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            برای (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                اعشاری dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                اعشاری dval = 0.0f;
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
صحیح check_tensor(اعشاری *a, اعشاری *b, صحیح n, نویسه* label) {
    صحیح ok = 1;
    printf("%s\n", label);
    برای (صحیح i = 0; i < n; i++) {
        اگر (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } وگرنه {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    بازگشت ok;
}

صحیح اصلی() {

    صحیح B = 2; // batch
    صحیح T = 3; // time / sequence length
    صحیح C = 4; // number of channels

    اعشاری* x = (اعشاری*) malloc(B * T * C * اندازه(اعشاری));
    اعشاری* w = (اعشاری*) malloc(C * اندازه(اعشاری));
    اعشاری* b = (اعشاری*) malloc(C * اندازه(اعشاری));
    اعشاری* out = (اعشاری*) malloc(B * T * C * اندازه(اعشاری));
    اعشاری* mean = (اعشاری*) malloc(B * T * اندازه(اعشاری));
    اعشاری* rstd = (اعشاری*) malloc(B * T * اندازه(اعشاری));
    اعشاری* dout = (اعشاری*) malloc(B * T * C * اندازه(اعشاری));
    اعشاری* dx = (اعشاری*) malloc(B * T * C * اندازه(اعشاری));
    اعشاری* dw = (اعشاری*) malloc(C * اندازه(اعشاری));
    اعشاری* db = (اعشاری*) malloc(C * اندازه(اعشاری));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    اگر (file == پوچ) {
        printf("Error opening file\n");
        بازگشت 1;
    }
    fread(x, اندازه(اعشاری), B * T * C, file);
    fread(w, اندازه(اعشاری), C, file);
    fread(b, اندازه(اعشاری), C, file);
    fread(out, اندازه(اعشاری), B * T * C, file);
    fread(mean, اندازه(اعشاری), B * T, file);
    fread(rstd, اندازه(اعشاری), B * T, file);
    fread(dout, اندازه(اعشاری), B * T * C, file);
    fread(dx, اندازه(اعشاری), B * T * C, file);
    fread(dw, اندازه(اعشاری), C, file);
    fread(db, اندازه(اعشاری), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    اعشاری* c_out = (اعشاری*) malloc(B * T * C * اندازه(اعشاری));
    اعشاری* c_mean = (اعشاری*) malloc(B * T * اندازه(اعشاری));
    اعشاری* c_rstd = (اعشاری*) malloc(B * T * اندازه(اعشاری));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    اعشاری* c_dx = (اعشاری*) calloc(B * T * C, اندازه(اعشاری));
    اعشاری* c_dw = (اعشاری*) calloc(B * T, اندازه(اعشاری));
    اعشاری* c_db = (اعشاری*) calloc(B * T, اندازه(اعشاری));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    آزاد(x);
    آزاد(w);
    آزاد(b);
    آزاد(out);
    آزاد(mean);
    آزاد(rstd);
    آزاد(dout);
    آزاد(dx);
    آزاد(dw);
    آزاد(db);
    بازگشت 0;
}
