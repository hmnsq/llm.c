// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <math.h>

வெறுமை layernorm_forward(தசம* out, தசம* mean, தசம* rstd,
                       தசம* inp, தசம* weight, தசம* bias,
                       முழு B, முழு T, முழு C) {
    தசம eps = 1e-5f;
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            தசம* x = inp + b * T * C + t * C;
            // calculate the mean
            தசம m = 0.0f;
            க்கு (முழு i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            தசம v = 0.0f;
            க்கு (முழு i = 0; i < C; i++) {
                தசம xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            தசம s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            தசம* out_bt = out + b * T * C + t * C;
            க்கு (முழு i = 0; i < C; i++) {
                தசம n = (s * (x[i] - m)); // normalized output
                தசம o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

வெறுமை layernorm_backward(தசம* dinp, தசம* dweight, தசம* dbias,
                        தசம* dout, தசம* inp, தசம* weight, தசம* mean, தசம* rstd,
                        முழு B, முழு T, முழு C) {
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            தசம* dout_bt = dout + b * T * C + t * C;
            தசம* inp_bt = inp + b * T * C + t * C;
            தசம* dinp_bt = dinp + b * T * C + t * C;
            தசம mean_bt = mean[b * T + t];
            தசம rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            தசம dnorm_mean = 0.0f;
            தசம dnorm_norm_mean = 0.0f;
            க்கு (முழு i = 0; i < C; i++) {
                தசம norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                தசம dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            க்கு (முழு i = 0; i < C; i++) {
                தசம norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                தசம dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                தசம dval = 0.0f;
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
முழு check_tensor(தசம *a, தசம *b, முழு n, எழுத்து* label) {
    முழு ok = 1;
    printf("%s\n", label);
    க்கு (முழு i = 0; i < n; i++) {
        எனில் (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } இல்லையெனில் {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    திரும்பு ok;
}

முழு முதன்மை() {

    முழு B = 2; // batch
    முழு T = 3; // time / sequence length
    முழு C = 4; // number of channels

    தசம* x = (தசம*) malloc(B * T * C * அளவு(தசம));
    தசம* w = (தசம*) malloc(C * அளவு(தசம));
    தசம* b = (தசம*) malloc(C * அளவு(தசம));
    தசம* out = (தசம*) malloc(B * T * C * அளவு(தசம));
    தசம* mean = (தசம*) malloc(B * T * அளவு(தசம));
    தசம* rstd = (தசம*) malloc(B * T * அளவு(தசம));
    தசம* dout = (தசம*) malloc(B * T * C * அளவு(தசம));
    தசம* dx = (தசம*) malloc(B * T * C * அளவு(தசம));
    தசம* dw = (தசம*) malloc(C * அளவு(தசம));
    தசம* db = (தசம*) malloc(C * அளவு(தசம));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    எனில் (file == வெற்று) {
        printf("Error opening file\n");
        திரும்பு 1;
    }
    fread(x, அளவு(தசம), B * T * C, file);
    fread(w, அளவு(தசம), C, file);
    fread(b, அளவு(தசம), C, file);
    fread(out, அளவு(தசம), B * T * C, file);
    fread(mean, அளவு(தசம), B * T, file);
    fread(rstd, அளவு(தசம), B * T, file);
    fread(dout, அளவு(தசம), B * T * C, file);
    fread(dx, அளவு(தசம), B * T * C, file);
    fread(dw, அளவு(தசம), C, file);
    fread(db, அளவு(தசம), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    தசம* c_out = (தசம*) malloc(B * T * C * அளவு(தசம));
    தசம* c_mean = (தசம*) malloc(B * T * அளவு(தசம));
    தசம* c_rstd = (தசம*) malloc(B * T * அளவு(தசம));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    தசம* c_dx = (தசம*) calloc(B * T * C, அளவு(தசம));
    தசம* c_dw = (தசம*) calloc(B * T, அளவு(தசம));
    தசம* c_db = (தசம*) calloc(B * T, அளவு(தசம));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    விடுவி(x);
    விடுவி(w);
    விடுவி(b);
    விடுவி(out);
    விடுவி(mean);
    விடுவி(rstd);
    விடுவி(dout);
    விடுவி(dx);
    விடுவி(dw);
    விடுவி(db);
    திரும்பு 0;
}
