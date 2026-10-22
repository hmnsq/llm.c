// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <math.h>

শূন্য layernorm_forward(দশমিক* out, দশমিক* mean, দশমিক* rstd,
                       দশমিক* inp, দশমিক* weight, দশমিক* bias,
                       পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C) {
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
            // calculate the rstd
            দশমিক s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            দশমিক* out_bt = out + b * T * C + t * C;
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক n = (s * (x[i] - m)); // normalized output
                দশমিক o = n * weight[i] + bias[i]; // scale and shift it
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

// poor man's tensor checker
পূর্ণসংখ্যা check_tensor(দশমিক *a, দশমিক *b, পূর্ণসংখ্যা n, অক্ষর* label) {
    পূর্ণসংখ্যা ok = 1;
    printf("%s\n", label);
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        যদি (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } অন্যথা {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    ফেরত ok;
}

পূর্ণসংখ্যা প্রধান() {

    পূর্ণসংখ্যা B = 2; // batch
    পূর্ণসংখ্যা T = 3; // time / sequence length
    পূর্ণসংখ্যা C = 4; // number of channels

    দশমিক* x = (দশমিক*) malloc(B * T * C * আকার(দশমিক));
    দশমিক* w = (দশমিক*) malloc(C * আকার(দশমিক));
    দশমিক* b = (দশমিক*) malloc(C * আকার(দশমিক));
    দশমিক* out = (দশমিক*) malloc(B * T * C * আকার(দশমিক));
    দশমিক* mean = (দশমিক*) malloc(B * T * আকার(দশমিক));
    দশমিক* rstd = (দশমিক*) malloc(B * T * আকার(দশমিক));
    দশমিক* dout = (দশমিক*) malloc(B * T * C * আকার(দশমিক));
    দশমিক* dx = (দশমিক*) malloc(B * T * C * আকার(দশমিক));
    দশমিক* dw = (দশমিক*) malloc(C * আকার(দশমিক));
    দশমিক* db = (দশমিক*) malloc(C * আকার(দশমিক));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    যদি (file == ফাঁকা) {
        printf("Error opening file\n");
        ফেরত 1;
    }
    fread(x, আকার(দশমিক), B * T * C, file);
    fread(w, আকার(দশমিক), C, file);
    fread(b, আকার(দশমিক), C, file);
    fread(out, আকার(দশমিক), B * T * C, file);
    fread(mean, আকার(দশমিক), B * T, file);
    fread(rstd, আকার(দশমিক), B * T, file);
    fread(dout, আকার(দশমিক), B * T * C, file);
    fread(dx, আকার(দশমিক), B * T * C, file);
    fread(dw, আকার(দশমিক), C, file);
    fread(db, আকার(দশমিক), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    দশমিক* c_out = (দশমিক*) malloc(B * T * C * আকার(দশমিক));
    দশমিক* c_mean = (দশমিক*) malloc(B * T * আকার(দশমিক));
    দশমিক* c_rstd = (দশমিক*) malloc(B * T * আকার(দশমিক));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    দশমিক* c_dx = (দশমিক*) calloc(B * T * C, আকার(দশমিক));
    দশমিক* c_dw = (দশমিক*) calloc(B * T, আকার(দশমিক));
    দশমিক* c_db = (দশমিক*) calloc(B * T, আকার(দশমিক));
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
    ফেরত 0;
}
