// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>

ਖਾਲੀ layernorm_forward(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ, ਦਸ਼ਮਲਵ* ਔਸਤ, ਦਸ਼ਮਲਵ* rstd,
                       ਦਸ਼ਮਲਵ* inp, ਦਸ਼ਮਲਵ* ਭਾਰ, ਦਸ਼ਮਲਵ* bias,
                       ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C) {
    ਦਸ਼ਮਲਵ eps = 1e-5f;
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ਦਸ਼ਮਲਵ* x = inp + b * T * C + t * C;
            // calculate the mean
            ਦਸ਼ਮਲਵ m = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ਦਸ਼ਮਲਵ v = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            ਦਸ਼ਮਲਵ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ_bt = ਬਾਹਰਲਾ + b * T * C + t * C;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ n = (s * (x[i] - m)); // normalized output
                ਦਸ਼ਮਲਵ o = n * ਭਾਰ[i] + bias[i]; // scale and shift it
                ਬਾਹਰਲਾ_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            ਔਸਤ[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ਖਾਲੀ layernorm_backward(ਦਸ਼ਮਲਵ* dinp, ਦਸ਼ਮਲਵ* dweight, ਦਸ਼ਮਲਵ* dbias,
                        ਦਸ਼ਮਲਵ* dout, ਦਸ਼ਮਲਵ* inp, ਦਸ਼ਮਲਵ* ਭਾਰ, ਦਸ਼ਮਲਵ* ਔਸਤ, ਦਸ਼ਮਲਵ* rstd,
                        ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C) {
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਦਸ਼ਮਲਵ* dout_bt = dout + b * T * C + t * C;
            ਦਸ਼ਮਲਵ* inp_bt = inp + b * T * C + t * C;
            ਦਸ਼ਮਲਵ* dinp_bt = dinp + b * T * C + t * C;
            ਦਸ਼ਮਲਵ ਔਸਤ_bt = ਔਸਤ[b * T + t];
            ਦਸ਼ਮਲਵ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ਦਸ਼ਮਲਵ dnorm_ਔਸਤ = 0.0f;
            ਦਸ਼ਮਲਵ dnorm_norm_ਔਸਤ = 0.0f;
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ norm_bti = (inp_bt[i] - ਔਸਤ_bt) * rstd_bt;
                ਦਸ਼ਮਲਵ dnorm_i = ਭਾਰ[i] * dout_bt[i];
                dnorm_ਔਸਤ += dnorm_i;
                dnorm_norm_ਔਸਤ += dnorm_i * norm_bti;
            }
            dnorm_ਔਸਤ = dnorm_ਔਸਤ / C;
            dnorm_norm_ਔਸਤ = dnorm_norm_ਔਸਤ / C;

            // now iterate again and accumulate all the gradients
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ norm_bti = (inp_bt[i] - ਔਸਤ_bt) * rstd_bt;
                ਦਸ਼ਮਲਵ dnorm_i = ਭਾਰ[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ਦਸ਼ਮਲਵ dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_ਔਸਤ; // term 2
                dval -= norm_bti * dnorm_norm_ਔਸਤ; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
ਪੂਰਨ check_tensor(ਦਸ਼ਮਲਵ *a, ਦਸ਼ਮਲਵ *b, ਪੂਰਨ n, ਅੱਖਰ* label) {
    ਪੂਰਨ ok = 1;
    ਲਿਖੋ("%s\n", label);
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ਜੇ (fabs(a[i] - b[i]) <= 1e-5) {
            ਲਿਖੋ("OK ");
        } ਨਹੀਂਤਾਂ {
            ਲਿਖੋ("NOT OK ");
            ok = 0;
        }
        ਲਿਖੋ("%f %f\n", a[i], b[i]);
    }
    ਵਾਪਸ ok;
}

ਪੂਰਨ ਮੁੱਖ() {

    ਪੂਰਨ B = 2; // batch
    ਪੂਰਨ T = 3; // time / sequence length
    ਪੂਰਨ C = 4; // number of channels

    ਦਸ਼ਮਲਵ* x = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* w = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* b = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* ਔਸਤ = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* rstd = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* dout = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* dx = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* dw = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* db = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    ਜੇ (file == ਨਿਰਾ) {
        ਲਿਖੋ("Error opening file\n");
        ਵਾਪਸ 1;
    }
    fread(x, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B * T * C, file);
    fread(w, ਆਕਾਰ(ਦਸ਼ਮਲਵ), C, file);
    fread(b, ਆਕਾਰ(ਦਸ਼ਮਲਵ), C, file);
    fread(ਬਾਹਰਲਾ, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B * T * C, file);
    fread(ਔਸਤ, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B * T, file);
    fread(rstd, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B * T, file);
    fread(dout, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B * T * C, file);
    fread(dx, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B * T * C, file);
    fread(dw, ਆਕਾਰ(ਦਸ਼ਮਲਵ), C, file);
    fread(db, ਆਕਾਰ(ਦਸ਼ਮਲਵ), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    ਦਸ਼ਮਲਵ* c_ਬਾਹਰਲਾ = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * C * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* c_ਔਸਤ = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* c_rstd = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    layernorm_forward(c_ਬਾਹਰਲਾ, c_ਔਸਤ, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(ਬਾਹਰਲਾ, c_ਬਾਹਰਲਾ, B*T*C, "out");
    check_tensor(ਔਸਤ, c_ਔਸਤ, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    ਦਸ਼ਮਲਵ* c_dx = (ਦਸ਼ਮਲਵ*) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * C, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* c_dw = (ਦਸ਼ਮਲਵ*) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* c_db = (ਦਸ਼ਮਲਵ*) ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_ਔਸਤ, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    ਮੁਕਤ(x);
    ਮੁਕਤ(w);
    ਮੁਕਤ(b);
    ਮੁਕਤ(ਬਾਹਰਲਾ);
    ਮੁਕਤ(ਔਸਤ);
    ਮੁਕਤ(rstd);
    ਮੁਕਤ(dout);
    ਮੁਕਤ(dx);
    ਮੁਕਤ(dw);
    ਮੁਕਤ(db);
    ਵਾਪਸ 0;
}
