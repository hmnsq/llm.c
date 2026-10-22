// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <math.h>

فراغ layernorm_forward(عائم* بیرون, عائم* میانگین, عائم* rstd,
                       عائم* inp, عائم* وزن, عائم* bias,
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
            عائم* بیرون_bt = بیرون + b * T * C + t * C;
            لكل (صحيح i = 0; i < C; i++) {
                عائم n = (s * (x[i] - m)); // normalized output
                عائم o = n * وزن[i] + bias[i]; // scale and shift it
                بیرون_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            میانگین[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

فراغ layernorm_backward(عائم* dinp, عائم* dweight, عائم* dbias,
                        عائم* dout, عائم* inp, عائم* وزن, عائم* میانگین, عائم* rstd,
                        صحيح B, صحيح T, صحيح C) {
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* dout_bt = dout + b * T * C + t * C;
            عائم* inp_bt = inp + b * T * C + t * C;
            عائم* dinp_bt = dinp + b * T * C + t * C;
            عائم میانگین_bt = میانگین[b * T + t];
            عائم rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            عائم dnorm_میانگین = 0.0f;
            عائم dnorm_norm_میانگین = 0.0f;
            لكل (صحيح i = 0; i < C; i++) {
                عائم norm_bti = (inp_bt[i] - میانگین_bt) * rstd_bt;
                عائم dnorm_i = وزن[i] * dout_bt[i];
                dnorm_میانگین += dnorm_i;
                dnorm_norm_میانگین += dnorm_i * norm_bti;
            }
            dnorm_میانگین = dnorm_میانگین / C;
            dnorm_norm_میانگین = dnorm_norm_میانگین / C;

            // now iterate again and accumulate all the gradients
            لكل (صحيح i = 0; i < C; i++) {
                عائم norm_bti = (inp_bt[i] - میانگین_bt) * rstd_bt;
                عائم dnorm_i = وزن[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                عائم dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_میانگین; // term 2
                dval -= norm_bti * dnorm_norm_میانگین; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
صحيح check_tensor(عائم *a, عائم *b, صحيح n, حرف* label) {
    صحيح ok = 1;
    چاپ_کن("%s\n", label);
    لكل (صحيح i = 0; i < n; i++) {
        إذا (fabs(a[i] - b[i]) <= 1e-5) {
            چاپ_کن("OK ");
        } وإلا {
            چاپ_کن("NOT OK ");
            ok = 0;
        }
        چاپ_کن("%f %f\n", a[i], b[i]);
    }
    أرجع ok;
}

صحيح اصلی() {

    صحيح B = 2; // batch
    صحيح T = 3; // time / sequence length
    صحيح C = 4; // number of channels

    عائم* x = (عائم*) حافظه_بده(B * T * C * حجم(عائم));
    عائم* w = (عائم*) حافظه_بده(C * حجم(عائم));
    عائم* b = (عائم*) حافظه_بده(C * حجم(عائم));
    عائم* بیرون = (عائم*) حافظه_بده(B * T * C * حجم(عائم));
    عائم* میانگین = (عائم*) حافظه_بده(B * T * حجم(عائم));
    عائم* rstd = (عائم*) حافظه_بده(B * T * حجم(عائم));
    عائم* dout = (عائم*) حافظه_بده(B * T * C * حجم(عائم));
    عائم* dx = (عائم*) حافظه_بده(B * T * C * حجم(عائم));
    عائم* dw = (عائم*) حافظه_بده(C * حجم(عائم));
    عائم* db = (عائم*) حافظه_بده(C * حجم(عائم));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    إذا (file == عدم) {
        چاپ_کن("Error opening file\n");
        أرجع 1;
    }
    fread(x, حجم(عائم), B * T * C, file);
    fread(w, حجم(عائم), C, file);
    fread(b, حجم(عائم), C, file);
    fread(بیرون, حجم(عائم), B * T * C, file);
    fread(میانگین, حجم(عائم), B * T, file);
    fread(rstd, حجم(عائم), B * T, file);
    fread(dout, حجم(عائم), B * T * C, file);
    fread(dx, حجم(عائم), B * T * C, file);
    fread(dw, حجم(عائم), C, file);
    fread(db, حجم(عائم), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    عائم* c_بیرون = (عائم*) حافظه_بده(B * T * C * حجم(عائم));
    عائم* c_میانگین = (عائم*) حافظه_بده(B * T * حجم(عائم));
    عائم* c_rstd = (عائم*) حافظه_بده(B * T * حجم(عائم));
    layernorm_forward(c_بیرون, c_میانگین, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(بیرون, c_بیرون, B*T*C, "out");
    check_tensor(میانگین, c_میانگین, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    عائم* c_dx = (عائم*) حافظه_صفر_بده(B * T * C, حجم(عائم));
    عائم* c_dw = (عائم*) حافظه_صفر_بده(B * T, حجم(عائم));
    عائم* c_db = (عائم*) حافظه_صفر_بده(B * T, حجم(عائم));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_میانگین, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    حرر(x);
    حرر(w);
    حرر(b);
    حرر(بیرون);
    حرر(میانگین);
    حرر(rstd);
    حرر(dout);
    حرر(dx);
    حرر(dw);
    حرر(db);
    أرجع 0;
}
