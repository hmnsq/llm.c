// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <math.h>

خالی layernorm_forward(اعشاری* برآمد, اعشاری* اوسط, اعشاری* rstd,
                       اعشاری* inp, اعشاری* وزن, اعشاری* bias,
                       صحیح B, صحیح T, صحیح C) {
    اعشاری eps = 1e-5f;
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            اعشاری* x = inp + b * T * C + t * C;
            // calculate the mean
            اعشاری m = 0.0f;
            برائے (صحیح i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            اعشاری v = 0.0f;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            اعشاری s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            اعشاری* برآمد_bt = برآمد + b * T * C + t * C;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری n = (s * (x[i] - m)); // normalized output
                اعشاری o = n * وزن[i] + bias[i]; // scale and shift it
                برآمد_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            اوسط[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

خالی layernorm_backward(اعشاری* dinp, اعشاری* dweight, اعشاری* dbias,
                        اعشاری* dout, اعشاری* inp, اعشاری* وزن, اعشاری* اوسط, اعشاری* rstd,
                        صحیح B, صحیح T, صحیح C) {
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            اعشاری* dout_bt = dout + b * T * C + t * C;
            اعشاری* inp_bt = inp + b * T * C + t * C;
            اعشاری* dinp_bt = dinp + b * T * C + t * C;
            اعشاری اوسط_bt = اوسط[b * T + t];
            اعشاری rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            اعشاری dnorm_اوسط = 0.0f;
            اعشاری dnorm_norm_اوسط = 0.0f;
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - اوسط_bt) * rstd_bt;
                اعشاری dnorm_i = وزن[i] * dout_bt[i];
                dnorm_اوسط += dnorm_i;
                dnorm_norm_اوسط += dnorm_i * norm_bti;
            }
            dnorm_اوسط = dnorm_اوسط / C;
            dnorm_norm_اوسط = dnorm_norm_اوسط / C;

            // now iterate again and accumulate all the gradients
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری norm_bti = (inp_bt[i] - اوسط_bt) * rstd_bt;
                اعشاری dnorm_i = وزن[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                اعشاری dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_اوسط; // term 2
                dval -= norm_bti * dnorm_norm_اوسط; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
صحیح check_tensor(اعشاری *a, اعشاری *b, صحیح n, حرف* label) {
    صحیح ok = 1;
    لکھو("%s\n", label);
    برائے (صحیح i = 0; i < n; i++) {
        اگر (fabs(a[i] - b[i]) <= 1e-5) {
            لکھو("OK ");
        } ورنہ {
            لکھو("NOT OK ");
            ok = 0;
        }
        لکھو("%f %f\n", a[i], b[i]);
    }
    واپس ok;
}

صحیح مین() {

    صحیح B = 2; // batch
    صحیح T = 3; // time / sequence length
    صحیح C = 4; // number of channels

    اعشاری* x = (اعشاری*) حافظہ_دو(B * T * C * حجم(اعشاری));
    اعشاری* w = (اعشاری*) حافظہ_دو(C * حجم(اعشاری));
    اعشاری* b = (اعشاری*) حافظہ_دو(C * حجم(اعشاری));
    اعشاری* برآمد = (اعشاری*) حافظہ_دو(B * T * C * حجم(اعشاری));
    اعشاری* اوسط = (اعشاری*) حافظہ_دو(B * T * حجم(اعشاری));
    اعشاری* rstd = (اعشاری*) حافظہ_دو(B * T * حجم(اعشاری));
    اعشاری* dout = (اعشاری*) حافظہ_دو(B * T * C * حجم(اعشاری));
    اعشاری* dx = (اعشاری*) حافظہ_دو(B * T * C * حجم(اعشاری));
    اعشاری* dw = (اعشاری*) حافظہ_دو(C * حجم(اعشاری));
    اعشاری* db = (اعشاری*) حافظہ_دو(C * حجم(اعشاری));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    اگر (file == تہی) {
        لکھو("Error opening file\n");
        واپس 1;
    }
    fread(x, حجم(اعشاری), B * T * C, file);
    fread(w, حجم(اعشاری), C, file);
    fread(b, حجم(اعشاری), C, file);
    fread(برآمد, حجم(اعشاری), B * T * C, file);
    fread(اوسط, حجم(اعشاری), B * T, file);
    fread(rstd, حجم(اعشاری), B * T, file);
    fread(dout, حجم(اعشاری), B * T * C, file);
    fread(dx, حجم(اعشاری), B * T * C, file);
    fread(dw, حجم(اعشاری), C, file);
    fread(db, حجم(اعشاری), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    اعشاری* c_برآمد = (اعشاری*) حافظہ_دو(B * T * C * حجم(اعشاری));
    اعشاری* c_اوسط = (اعشاری*) حافظہ_دو(B * T * حجم(اعشاری));
    اعشاری* c_rstd = (اعشاری*) حافظہ_دو(B * T * حجم(اعشاری));
    layernorm_forward(c_برآمد, c_اوسط, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(برآمد, c_برآمد, B*T*C, "out");
    check_tensor(اوسط, c_اوسط, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    اعشاری* c_dx = (اعشاری*) صفر_حافظہ_دو(B * T * C, حجم(اعشاری));
    اعشاری* c_dw = (اعشاری*) صفر_حافظہ_دو(B * T, حجم(اعشاری));
    اعشاری* c_db = (اعشاری*) صفر_حافظہ_دو(B * T, حجم(اعشاری));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_اوسط, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    آزاد(x);
    آزاد(w);
    آزاد(b);
    آزاد(برآمد);
    آزاد(اوسط);
    آزاد(rstd);
    آزاد(dout);
    آزاد(dx);
    آزاد(dw);
    آزاد(db);
    واپس 0;
}
