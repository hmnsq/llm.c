// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <math.h>

शून्य layernorm_forward(भिन्न* out, भिन्न* mean, भिन्न* rstd,
                       भिन्न* inp, भिन्न* weight, भिन्न* bias,
                       पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C) {
    भिन्न eps = 1e-5f;
    प्रति (पूर्णाङ्क b = 0; b < B; b++) {
        प्रति (पूर्णाङ्क t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            भिन्न* x = inp + b * T * C + t * C;
            // calculate the mean
            भिन्न m = 0.0f;
            प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            भिन्न v = 0.0f;
            प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                भिन्न xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            भिन्न s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            भिन्न* out_bt = out + b * T * C + t * C;
            प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                भिन्न n = (s * (x[i] - m)); // normalized output
                भिन्न o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

शून्य layernorm_backward(भिन्न* dinp, भिन्न* dweight, भिन्न* dbias,
                        भिन्न* dout, भिन्न* inp, भिन्न* weight, भिन्न* mean, भिन्न* rstd,
                        पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C) {
    प्रति (पूर्णाङ्क b = 0; b < B; b++) {
        प्रति (पूर्णाङ्क t = 0; t < T; t++) {
            भिन्न* dout_bt = dout + b * T * C + t * C;
            भिन्न* inp_bt = inp + b * T * C + t * C;
            भिन्न* dinp_bt = dinp + b * T * C + t * C;
            भिन्न mean_bt = mean[b * T + t];
            भिन्न rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            भिन्न dnorm_mean = 0.0f;
            भिन्न dnorm_norm_mean = 0.0f;
            प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                भिन्न norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                भिन्न dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                भिन्न norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                भिन्न dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                भिन्न dval = 0.0f;
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
पूर्णाङ्क check_tensor(भिन्न *a, भिन्न *b, पूर्णाङ्क n, वर्ण* label) {
    पूर्णाङ्क ok = 1;
    म_लिखो("%s\n", label);
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        यदि (भ_असल(a[i] - b[i]) <= 1e-5) {
            म_लिखो("OK ");
        } अन्यथा {
            म_लिखो("NOT OK ");
            ok = 0;
        }
        म_लिखो("%f %f\n", a[i], b[i]);
    }
    प्रत्यागम ok;
}

पूर्णाङ्क main() {

    पूर्णाङ्क B = 2; // batch
    पूर्णाङ्क T = 3; // time / sequence length
    पूर्णाङ्क C = 4; // number of channels

    भिन्न* x = (भिन्न*) दो_स्मृति(B * T * C * परिमाण(भिन्न));
    भिन्न* w = (भिन्न*) दो_स्मृति(C * परिमाण(भिन्न));
    भिन्न* b = (भिन्न*) दो_स्मृति(C * परिमाण(भिन्न));
    भिन्न* out = (भिन्न*) दो_स्मृति(B * T * C * परिमाण(भिन्न));
    भिन्न* mean = (भिन्न*) दो_स्मृति(B * T * परिमाण(भिन्न));
    भिन्न* rstd = (भिन्न*) दो_स्मृति(B * T * परिमाण(भिन्न));
    भिन्न* dout = (भिन्न*) दो_स्मृति(B * T * C * परिमाण(भिन्न));
    भिन्न* dx = (भिन्न*) दो_स्मृति(B * T * C * परिमाण(भिन्न));
    भिन्न* dw = (भिन्न*) दो_स्मृति(C * परिमाण(भिन्न));
    भिन्न* db = (भिन्न*) दो_स्मृति(C * परिमाण(भिन्न));

    // read reference information from Python
    खाता *file = ख_खोलो("ln.bin", "rb");
    यदि (file == रिक्त) {
        म_लिखो("Error opening file\n");
        प्रत्यागम 1;
    }
    ख_पढ़ो(x, परिमाण(भिन्न), B * T * C, file);
    ख_पढ़ो(w, परिमाण(भिन्न), C, file);
    ख_पढ़ो(b, परिमाण(भिन्न), C, file);
    ख_पढ़ो(out, परिमाण(भिन्न), B * T * C, file);
    ख_पढ़ो(mean, परिमाण(भिन्न), B * T, file);
    ख_पढ़ो(rstd, परिमाण(भिन्न), B * T, file);
    ख_पढ़ो(dout, परिमाण(भिन्न), B * T * C, file);
    ख_पढ़ो(dx, परिमाण(भिन्न), B * T * C, file);
    ख_पढ़ो(dw, परिमाण(भिन्न), C, file);
    ख_पढ़ो(db, परिमाण(भिन्न), C, file);
    ख_बंद(file);

    // now let's calculate everything ourselves

    // forward pass
    भिन्न* c_out = (भिन्न*) दो_स्मृति(B * T * C * परिमाण(भिन्न));
    भिन्न* c_mean = (भिन्न*) दो_स्मृति(B * T * परिमाण(भिन्न));
    भिन्न* c_rstd = (भिन्न*) दो_स्मृति(B * T * परिमाण(भिन्न));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    भिन्न* c_dx = (भिन्न*) सुस्मृति(B * T * C, परिमाण(भिन्न));
    भिन्न* c_dw = (भिन्न*) सुस्मृति(B * T, परिमाण(भिन्न));
    भिन्न* c_db = (भिन्न*) सुस्मृति(B * T, परिमाण(भिन्न));
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // check correctness of backward pass
    check_tensor(c_dx, dx, B*T*C, "dx");
    check_tensor(c_dw, dw, C, "dw");
    check_tensor(c_db, db, C, "db");

    मुक्त(x);
    मुक्त(w);
    मुक्त(b);
    मुक्त(out);
    मुक्त(mean);
    मुक्त(rstd);
    मुक्त(dout);
    मुक्त(dx);
    मुक्त(dw);
    मुक्त(db);
    प्रत्यागम 0;
}
