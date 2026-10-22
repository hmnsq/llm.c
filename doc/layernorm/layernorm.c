// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <math.h>

శూన్య layernorm_forward(దశాంశం* బహిర్గతం, దశాంశం* సగటు, దశాంశం* rstd,
                       దశాంశం* inp, దశాంశం* బరువు, దశాంశం* bias,
                       పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C) {
    దశాంశం eps = 1e-5f;
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            దశాంశం* x = inp + b * T * C + t * C;
            // calculate the mean
            దశాంశం m = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            దశాంశం v = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            దశాంశం s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            దశాంశం* బహిర్గతం_bt = బహిర్గతం + b * T * C + t * C;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం n = (s * (x[i] - m)); // normalized output
                దశాంశం o = n * బరువు[i] + bias[i]; // scale and shift it
                బహిర్గతం_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            సగటు[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

శూన్య layernorm_backward(దశాంశం* dinp, దశాంశం* dweight, దశాంశం* dbias,
                        దశాంశం* dout, దశాంశం* inp, దశాంశం* బరువు, దశాంశం* సగటు, దశాంశం* rstd,
                        పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C) {
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            దశాంశం* dout_bt = dout + b * T * C + t * C;
            దశాంశం* inp_bt = inp + b * T * C + t * C;
            దశాంశం* dinp_bt = dinp + b * T * C + t * C;
            దశాంశం సగటు_bt = సగటు[b * T + t];
            దశాంశం rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            దశాంశం dnorm_సగటు = 0.0f;
            దశాంశం dnorm_norm_సగటు = 0.0f;
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం norm_bti = (inp_bt[i] - సగటు_bt) * rstd_bt;
                దశాంశం dnorm_i = బరువు[i] * dout_bt[i];
                dnorm_సగటు += dnorm_i;
                dnorm_norm_సగటు += dnorm_i * norm_bti;
            }
            dnorm_సగటు = dnorm_సగటు / C;
            dnorm_norm_సగటు = dnorm_norm_సగటు / C;

            // now iterate again and accumulate all the gradients
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం norm_bti = (inp_bt[i] - సగటు_bt) * rstd_bt;
                దశాంశం dnorm_i = బరువు[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                దశాంశం dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_సగటు; // term 2
                dval -= norm_bti * dnorm_norm_సగటు; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// poor man's tensor checker
పూర్ణాంకం check_tensor(దశాంశం *a, దశాంశం *b, పూర్ణాంకం n, అక్షరం* label) {
    పూర్ణాంకం ok = 1;
    రాయి("%s\n", label);
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        ఒకవేళ (fabs(a[i] - b[i]) <= 1e-5) {
            రాయి("OK ");
        } లేకపోతే {
            రాయి("NOT OK ");
            ok = 0;
        }
        రాయి("%f %f\n", a[i], b[i]);
    }
    తిరిగి ok;
}

పూర్ణాంకం ప్రధాన() {

    పూర్ణాంకం B = 2; // batch
    పూర్ణాంకం T = 3; // time / sequence length
    పూర్ణాంకం C = 4; // number of channels

    దశాంశం* x = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * C * పరిమాణం(దశాంశం));
    దశాంశం* w = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(C * పరిమాణం(దశాంశం));
    దశాంశం* b = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(C * పరిమాణం(దశాంశం));
    దశాంశం* బహిర్గతం = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * C * పరిమాణం(దశాంశం));
    దశాంశం* సగటు = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * పరిమాణం(దశాంశం));
    దశాంశం* rstd = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * పరిమాణం(దశాంశం));
    దశాంశం* dout = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * C * పరిమాణం(దశాంశం));
    దశాంశం* dx = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * C * పరిమాణం(దశాంశం));
    దశాంశం* dw = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(C * పరిమాణం(దశాంశం));
    దశాంశం* db = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(C * పరిమాణం(దశాంశం));

    // read reference information from Python
    FILE *file = fopen("ln.bin", "rb");
    ఒకవేళ (file == ఖాళీ) {
        రాయి("Error opening file\n");
        తిరిగి 1;
    }
    fread(x, పరిమాణం(దశాంశం), B * T * C, file);
    fread(w, పరిమాణం(దశాంశం), C, file);
    fread(b, పరిమాణం(దశాంశం), C, file);
    fread(బహిర్గతం, పరిమాణం(దశాంశం), B * T * C, file);
    fread(సగటు, పరిమాణం(దశాంశం), B * T, file);
    fread(rstd, పరిమాణం(దశాంశం), B * T, file);
    fread(dout, పరిమాణం(దశాంశం), B * T * C, file);
    fread(dx, పరిమాణం(దశాంశం), B * T * C, file);
    fread(dw, పరిమాణం(దశాంశం), C, file);
    fread(db, పరిమాణం(దశాంశం), C, file);
    fclose(file);

    // now let's calculate everything ourselves

    // forward pass
    దశాంశం* c_బహిర్గతం = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * C * పరిమాణం(దశాంశం));
    దశాంశం* c_సగటు = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * పరిమాణం(దశాంశం));
    దశాంశం* c_rstd = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * పరిమాణం(దశాంశం));
    layernorm_forward(c_బహిర్గతం, c_సగటు, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(బహిర్గతం, c_బహిర్గతం, B*T*C, "out");
    check_tensor(సగటు, c_సగటు, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    దశాంశం* c_dx = (దశాంశం*) సున్నా_జ్ఞాపకం_ఇవ్వు(B * T * C, పరిమాణం(దశాంశం));
    దశాంశం* c_dw = (దశాంశం*) సున్నా_జ్ఞాపకం_ఇవ్వు(B * T, పరిమాణం(దశాంశం));
    దశాంశం* c_db = (దశాంశం*) సున్నా_జ్ఞాపకం_ఇవ్వు(B * T, పరిమాణం(దశాంశం));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_సగటు, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    విడుదల(x);
    విడుదల(w);
    విడుదల(b);
    విడుదల(బహిర్గతం);
    విడుదల(సగటు);
    విడుదల(rstd);
    విడుదల(dout);
    విడుదల(dx);
    విడుదల(dw);
    విడుదల(db);
    తిరిగి 0;
}
