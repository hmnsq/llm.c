// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <math.h>

व्योम layernorm_forward(भग्न* out, भग्न* mean, भग्न* rstd,
                       भग्न* inp, भग्न* weight, भग्न* bias,
                       पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    भग्न eps = 1e-5f;
    क्रम (पूर्णांक b = 0; b < B; b++) {
        क्रम (पूर्णांक t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            भग्न* x = inp + b * T * C + t * C;
            // calculate the mean
            भग्न m = 0.0f;
            क्रम (पूर्णांक i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            भग्न v = 0.0f;
            क्रम (पूर्णांक i = 0; i < C; i++) {
                भग्न xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            भग्न s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            भग्न* out_bt = out + b * T * C + t * C;
            क्रम (पूर्णांक i = 0; i < C; i++) {
                भग्न n = (s * (x[i] - m)); // normalized output
                भग्न o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

व्योम layernorm_backward(भग्न* dinp, भग्न* dweight, भग्न* dbias,
                        भग्न* dout, भग्न* inp, भग्न* weight, भग्न* mean, भग्न* rstd,
                        पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    क्रम (पूर्णांक b = 0; b < B; b++) {
        क्रम (पूर्णांक t = 0; t < T; t++) {
            भग्न* dout_bt = dout + b * T * C + t * C;
            भग्न* inp_bt = inp + b * T * C + t * C;
            भग्न* dinp_bt = dinp + b * T * C + t * C;
            भग्न mean_bt = mean[b * T + t];
            भग्न rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            भग्न dnorm_mean = 0.0f;
            भग्न dnorm_norm_mean = 0.0f;
            क्रम (पूर्णांक i = 0; i < C; i++) {
                भग्न norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                भग्न dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            क्रम (पूर्णांक i = 0; i < C; i++) {
                भग्न norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                भग्न dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                भग्न dval = 0.0f;
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
पूर्णांक check_tensor(भग्न *a, भग्न *b, पूर्णांक n, अक्षर* label) {
    पूर्णांक ok = 1;
    म_लिखो("%s\n", label);
    क्रम (पूर्णांक i = 0; i < n; i++) {
        अगर (भ_असल(a[i] - b[i]) <= 1e-5) {
            म_लिखो("OK ");
        } अन्यथा {
            म_लिखो("NOT OK ");
            ok = 0;
        }
        म_लिखो("%f %f\n", a[i], b[i]);
    }
    वापस ok;
}

पूर्णांक मुख्य() {

    पूर्णांक B = 2; // batch
    पूर्णांक T = 3; // time / sequence length
    पूर्णांक C = 4; // number of channels

    भग्न* x = (भग्न*) दो_स्मृति(B * T * C * माप(भग्न));
    भग्न* w = (भग्न*) दो_स्मृति(C * माप(भग्न));
    भग्न* b = (भग्न*) दो_स्मृति(C * माप(भग्न));
    भग्न* out = (भग्न*) दो_स्मृति(B * T * C * माप(भग्न));
    भग्न* mean = (भग्न*) दो_स्मृति(B * T * माप(भग्न));
    भग्न* rstd = (भग्न*) दो_स्मृति(B * T * माप(भग्न));
    भग्न* dout = (भग्न*) दो_स्मृति(B * T * C * माप(भग्न));
    भग्न* dx = (भग्न*) दो_स्मृति(B * T * C * माप(भग्न));
    भग्न* dw = (भग्न*) दो_स्मृति(C * माप(भग्न));
    भग्न* db = (भग्न*) दो_स्मृति(C * माप(भग्न));

    // read reference information from Python
    खाता *file = ख_खोलो("ln.bin", "rb");
    अगर (file == शून्य) {
        म_लिखो("Error opening file\n");
        वापस 1;
    }
    ख_पढ़ो(x, माप(भग्न), B * T * C, file);
    ख_पढ़ो(w, माप(भग्न), C, file);
    ख_पढ़ो(b, माप(भग्न), C, file);
    ख_पढ़ो(out, माप(भग्न), B * T * C, file);
    ख_पढ़ो(mean, माप(भग्न), B * T, file);
    ख_पढ़ो(rstd, माप(भग्न), B * T, file);
    ख_पढ़ो(dout, माप(भग्न), B * T * C, file);
    ख_पढ़ो(dx, माप(भग्न), B * T * C, file);
    ख_पढ़ो(dw, माप(भग्न), C, file);
    ख_पढ़ो(db, माप(भग्न), C, file);
    ख_बंद(file);

    // now let's calculate everything ourselves

    // forward pass
    भग्न* c_out = (भग्न*) दो_स्मृति(B * T * C * माप(भग्न));
    भग्न* c_mean = (भग्न*) दो_स्मृति(B * T * माप(भग्न));
    भग्न* c_rstd = (भग्न*) दो_स्मृति(B * T * माप(भग्न));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    भग्न* c_dx = (भग्न*) सुस्मृति(B * T * C, माप(भग्न));
    भग्न* c_dw = (भग्न*) सुस्मृति(B * T, माप(भग्न));
    भग्न* c_db = (भग्न*) सुस्मृति(B * T, माप(भग्न));
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
    वापस 0;
}
