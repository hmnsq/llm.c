// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <math.h>

ಶೂನ್ಯ layernorm_forward(ದಶಮಾಂಶ* ಹೊರಹೋಗು, ದಶಮಾಂಶ* ಸರಾಸರಿ, ದಶಮಾಂಶ* rstd,
                       ದಶಮಾಂಶ* inp, ದಶಮಾಂಶ* ತೂಕ, ದಶಮಾಂಶ* bias,
                       ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C) {
    ದಶಮಾಂಶ eps = 1e-5f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            ದಶಮಾಂಶ* x = inp + b * T * C + t * C;
            // calculate the mean
            ದಶಮಾಂಶ m = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            ದಶಮಾಂಶ v = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            ದಶಮಾಂಶ s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            ದಶಮಾಂಶ* ಹೊರಹೋಗು_bt = ಹೊರಹೋಗು + b * T * C + t * C;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ n = (s * (x[i] - m)); // normalized output
                ದಶಮಾಂಶ o = n * ತೂಕ[i] + bias[i]; // scale and shift it
                ಹೊರಹೋಗು_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            ಸರಾಸರಿ[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

ಶೂನ್ಯ layernorm_backward(ದಶಮಾಂಶ* dinp, ದಶಮಾಂಶ* dweight, ದಶಮಾಂಶ* dbias,
                        ದಶಮಾಂಶ* dout, ದಶಮಾಂಶ* inp, ದಶಮಾಂಶ* ತೂಕ, ದಶಮಾಂಶ* ಸರಾಸರಿ, ದಶಮಾಂಶ* rstd,
                        ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ದಶಮಾಂಶ* dout_bt = dout + b * T * C + t * C;
            ದಶಮಾಂಶ* inp_bt = inp + b * T * C + t * C;
            ದಶಮಾಂಶ* dinp_bt = dinp + b * T * C + t * C;
            ದಶಮಾಂಶ ಸರಾಸರಿ_bt = ಸರಾಸರಿ[b * T + t];
            ದಶಮಾಂಶ rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            ದಶಮಾಂಶ dnorm_ಸರಾಸರಿ = 0.0f;
            ದಶಮಾಂಶ dnorm_norm_ಸರಾಸರಿ = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ norm_bti = (inp_bt[i] - ಸರಾಸರಿ_bt) * rstd_bt;
                ದಶಮಾಂಶ dnorm_i = ತೂಕ[i] * dout_bt[i];
                dnorm_ಸರಾಸರಿ += dnorm_i;
                dnorm_norm_ಸರಾಸರಿ += dnorm_i * norm_bti;
            }
            dnorm_ಸರಾಸರಿ = dnorm_ಸರಾಸರಿ / C;
            dnorm_norm_ಸರಾಸರಿ = dnorm_norm_ಸರಾಸರಿ / C;

            // now iterate again and accumulate all the gradients
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ norm_bti = (inp_bt[i] - ಸರಾಸರಿ_bt) * rstd_bt;
                ದಶಮಾಂಶ dnorm_i = ತೂಕ[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                ದಶಮಾಂಶ dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_ಸರಾಸರಿ; // term 2
                dval -= norm_bti * dnorm_norm_ಸರಾಸರಿ; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
ಪೂರ್ಣಾಂಕ check_tensor(ದಶಮಾಂಶ *a, ದಶಮಾಂಶ *b, ಪೂರ್ಣಾಂಕ n, ಅಕ್ಷರ* label) {
    ಪೂರ್ಣಾಂಕ ok = 1;
    ಬರೆ("%s\n", label);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ಆದರೆ (fabs(a[i] - b[i]) <= 1e-5) {
            ಬರೆ("OK ");
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಬರೆ("NOT OK ");
            ok = 0;
        }
        ಬರೆ("%f %f\n", a[i], b[i]);
    }
    ಹಿಂತಿರುಗಿ ok;
}

ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ() {

    ಪೂರ್ಣಾಂಕ B = 2; // batch
    ಪೂರ್ಣಾಂಕ T = 3; // time / sequence length
    ಪೂರ್ಣಾಂಕ C = 4; // number of channels

    ದಶಮಾಂಶ* x = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* w = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* b = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* ಹೊರಹೋಗು = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* ಸರಾಸರಿ = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* rstd = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* dout = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* dx = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* dw = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* db = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(C * ಗಾತ್ರ(ದಶಮಾಂಶ));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    ಆದರೆ (file == ಖಾಲಿ) {
        ಬರೆ("Error opening file\n");
        ಹಿಂತಿರುಗಿ 1;
    }
    fread(x, ಗಾತ್ರ(ದಶಮಾಂಶ), B * T * C, file);
    fread(w, ಗಾತ್ರ(ದಶಮಾಂಶ), C, file);
    fread(b, ಗಾತ್ರ(ದಶಮಾಂಶ), C, file);
    fread(ಹೊರಹೋಗು, ಗಾತ್ರ(ದಶಮಾಂಶ), B * T * C, file);
    fread(ಸರಾಸರಿ, ಗಾತ್ರ(ದಶಮಾಂಶ), B * T, file);
    fread(rstd, ಗಾತ್ರ(ದಶಮಾಂಶ), B * T, file);
    fread(dout, ಗಾತ್ರ(ದಶಮಾಂಶ), B * T * C, file);
    fread(dx, ಗಾತ್ರ(ದಶಮಾಂಶ), B * T * C, file);
    fread(dw, ಗಾತ್ರ(ದಶಮಾಂಶ), C, file);
    fread(db, ಗಾತ್ರ(ದಶಮಾಂಶ), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    ದಶಮಾಂಶ* c_ಹೊರಹೋಗು = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * C * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* c_ಸರಾಸರಿ = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* c_rstd = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * ಗಾತ್ರ(ದಶಮಾಂಶ));
    layernorm_forward(c_ಹೊರಹೋಗು, c_ಸರಾಸರಿ, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(ಹೊರಹೋಗು, c_ಹೊರಹೋಗು, B*T*C, "out");
    check_tensor(ಸರಾಸರಿ, c_ಸರಾಸರಿ, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    ದಶಮಾಂಶ* c_dx = (ದಶಮಾಂಶ*) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(B * T * C, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* c_dw = (ದಶಮಾಂಶ*) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(B * T, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* c_db = (ದಶಮಾಂಶ*) ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(B * T, ಗಾತ್ರ(ದಶಮಾಂಶ));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_ಸರಾಸರಿ, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    ಬಿಡುಗಡೆ(x);
    ಬಿಡುಗಡೆ(w);
    ಬಿಡುಗಡೆ(b);
    ಬಿಡುಗಡೆ(ಹೊರಹೋಗು);
    ಬಿಡುಗಡೆ(ಸರಾಸರಿ);
    ಬಿಡುಗಡೆ(rstd);
    ಬಿಡುಗಡೆ(dout);
    ಬಿಡುಗಡೆ(dx);
    ಬಿಡುಗಡೆ(dw);
    ಬಿಡುಗಡೆ(db);
    ಹಿಂತಿರುಗಿ 0;
}
