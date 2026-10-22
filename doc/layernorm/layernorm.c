// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <math.h>

فراغ layernorm_forward(عائم* بهر, عائم* منځنۍ, عائم* rstd,
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
            عائم* بهر_bt = بهر + b * T * C + t * C;
            لكل (صحيح i = 0; i < C; i++) {
                عائم n = (s * (x[i] - m)); // normalized output
                عائم o = n * وزن[i] + bias[i]; // scale and shift it
                بهر_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            منځنۍ[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

فراغ layernorm_backward(عائم* dinp, عائم* dweight, عائم* dbias,
                        عائم* dout, عائم* inp, عائم* وزن, عائم* منځنۍ, عائم* rstd,
                        صحيح B, صحيح T, صحيح C) {
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* dout_bt = dout + b * T * C + t * C;
            عائم* inp_bt = inp + b * T * C + t * C;
            عائم* dinp_bt = dinp + b * T * C + t * C;
            عائم منځنۍ_bt = منځنۍ[b * T + t];
            عائم rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            عائم dnorm_منځنۍ = 0.0f;
            عائم dnorm_norm_منځنۍ = 0.0f;
            لكل (صحيح i = 0; i < C; i++) {
                عائم norm_bti = (inp_bt[i] - منځنۍ_bt) * rstd_bt;
                عائم dnorm_i = وزن[i] * dout_bt[i];
                dnorm_منځنۍ += dnorm_i;
                dnorm_norm_منځنۍ += dnorm_i * norm_bti;
            }
            dnorm_منځنۍ = dnorm_منځنۍ / C;
            dnorm_norm_منځنۍ = dnorm_norm_منځنۍ / C;

            // now iterate again and accumulate all the gradients
            لكل (صحيح i = 0; i < C; i++) {
                عائم norm_bti = (inp_bt[i] - منځنۍ_bt) * rstd_bt;
                عائم dnorm_i = وزن[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                عائم dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_منځنۍ; // term 2
                dval -= norm_bti * dnorm_norm_منځنۍ; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
صحيح check_tensor(عائم *a, عائم *b, صحيح n, حرف* label) {
    صحيح ok = 1;
    چاپ_کړه("%s\n", label);
    لكل (صحيح i = 0; i < n; i++) {
        إذا (fabs(a[i] - b[i]) <= 1e-5) {
            چاپ_کړه("OK ");
        } وإلا {
            چاپ_کړه("NOT OK ");
            ok = 0;
        }
        چاپ_کړه("%f %f\n", a[i], b[i]);
    }
    أرجع ok;
}

صحيح اصلي() {

    صحيح B = 2; // batch
    صحيح T = 3; // time / sequence length
    صحيح C = 4; // number of channels

    عائم* x = (عائم*) حافظه_ورکړه(B * T * C * حجم(عائم));
    عائم* w = (عائم*) حافظه_ورکړه(C * حجم(عائم));
    عائم* b = (عائم*) حافظه_ورکړه(C * حجم(عائم));
    عائم* بهر = (عائم*) حافظه_ورکړه(B * T * C * حجم(عائم));
    عائم* منځنۍ = (عائم*) حافظه_ورکړه(B * T * حجم(عائم));
    عائم* rstd = (عائم*) حافظه_ورکړه(B * T * حجم(عائم));
    عائم* dout = (عائم*) حافظه_ورکړه(B * T * C * حجم(عائم));
    عائم* dx = (عائم*) حافظه_ورکړه(B * T * C * حجم(عائم));
    عائم* dw = (عائم*) حافظه_ورکړه(C * حجم(عائم));
    عائم* db = (عائم*) حافظه_ورکړه(C * حجم(عائم));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    إذا (file == عدم) {
        چاپ_کړه("Error opening file\n");
        أرجع 1;
    }
    fread(x, حجم(عائم), B * T * C, file);
    fread(w, حجم(عائم), C, file);
    fread(b, حجم(عائم), C, file);
    fread(بهر, حجم(عائم), B * T * C, file);
    fread(منځنۍ, حجم(عائم), B * T, file);
    fread(rstd, حجم(عائم), B * T, file);
    fread(dout, حجم(عائم), B * T * C, file);
    fread(dx, حجم(عائم), B * T * C, file);
    fread(dw, حجم(عائم), C, file);
    fread(db, حجم(عائم), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    عائم* c_بهر = (عائم*) حافظه_ورکړه(B * T * C * حجم(عائم));
    عائم* c_منځنۍ = (عائم*) حافظه_ورکړه(B * T * حجم(عائم));
    عائم* c_rstd = (عائم*) حافظه_ورکړه(B * T * حجم(عائم));
    layernorm_forward(c_بهر, c_منځنۍ, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(بهر, c_بهر, B*T*C, "out");
    check_tensor(منځنۍ, c_منځنۍ, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    عائم* c_dx = (عائم*) حافظه_صفر_ورکړه(B * T * C, حجم(عائم));
    عائم* c_dw = (عائم*) حافظه_صفر_ورکړه(B * T, حجم(عائم));
    عائم* c_db = (عائم*) حافظه_صفر_ورکړه(B * T, حجم(عائم));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_منځنۍ, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    حرر(x);
    حرر(w);
    حرر(b);
    حرر(بهر);
    حرر(منځنۍ);
    حرر(rstd);
    حرر(dout);
    حرر(dx);
    حرر(dw);
    حرر(db);
    أرجع 0;
}
