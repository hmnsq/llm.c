// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       puur_naa_nka B, puur_naa_nka T, puur_naa_nka C) {
    float eps = 1e-5f;
    for (puur_naa_nka b = 0; b < B; b++) {
        for (puur_naa_nka t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (puur_naa_nka i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (puur_naa_nka i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (puur_naa_nka i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        puur_naa_nka B, puur_naa_nka T, puur_naa_nka C) {
    for (puur_naa_nka b = 0; b < B; b++) {
        for (puur_naa_nka t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (puur_naa_nka i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (puur_naa_nka i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
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
puur_naa_nka check_tensor(float *a, float *b, puur_naa_nka n, var_na* label) {
    puur_naa_nka ok = 1;
    printf("%s\n", label);
    for (puur_naa_nka i = 0; i < n; i++) {
        ᤀᤡ᤺ (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } ᤀᤡ᤺ᤆᤡ {
            printf("NOT OK ");
            ok = 0;
        }
        printf("%f %f\n", a[i], b[i]);
    }
    ᤀᤡᤴ ok;
}

puur_naa_nka main() {

    puur_naa_nka B = 2; // batch
    puur_naa_nka T = 3; // time / sequence length
    puur_naa_nka C = 4; // number of channels

    float* x = (float*) malloc(B * T * C * parimaa_na(float));
    float* w = (float*) malloc(C * parimaa_na(float));
    float* b = (float*) malloc(C * parimaa_na(float));
    float* out = (float*) malloc(B * T * C * parimaa_na(float));
    float* mean = (float*) malloc(B * T * parimaa_na(float));
    float* rstd = (float*) malloc(B * T * parimaa_na(float));
    float* dout = (float*) malloc(B * T * C * parimaa_na(float));
    float* dx = (float*) malloc(B * T * C * parimaa_na(float));
    float* dw = (float*) malloc(C * parimaa_na(float));
    float* db = (float*) malloc(C * parimaa_na(float));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    ᤀᤡ᤺ (file == NULL) {
        printf("Error opening file\n");
        ᤀᤡᤴ 1;
    }
    fread(x, parimaa_na(float), B * T * C, file);
    fread(w, parimaa_na(float), C, file);
    fread(b, parimaa_na(float), C, file);
    fread(out, parimaa_na(float), B * T * C, file);
    fread(mean, parimaa_na(float), B * T, file);
    fread(rstd, parimaa_na(float), B * T, file);
    fread(dout, parimaa_na(float), B * T * C, file);
    fread(dx, parimaa_na(float), B * T * C, file);
    fread(dw, parimaa_na(float), C, file);
    fread(db, parimaa_na(float), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    float* c_out = (float*) malloc(B * T * C * parimaa_na(float));
    float* c_mean = (float*) malloc(B * T * parimaa_na(float));
    float* c_rstd = (float*) malloc(B * T * parimaa_na(float));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    float* c_dx = (float*) calloc(B * T * C, parimaa_na(float));
    float* c_dw = (float*) calloc(B * T, parimaa_na(float));
    float* c_db = (float*) calloc(B * T, parimaa_na(float));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    free(x);
    free(w);
    free(b);
    free(out);
    free(mean);
    free(rstd);
    free(dout);
    free(dx);
    free(dw);
    free(db);
    ᤀᤡᤴ 0;
}
