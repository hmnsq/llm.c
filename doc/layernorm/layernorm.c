// must run `python layernorm.py` first to generate the reference data
// then compile for example as `gcc layernorm.c -o layernorm -lm`
// and then run as `./layernorm` to see the output

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <math.h>

शून्य layernorm_forward(दशांश* out, दशांश* mean, दशांश* rstd,
                       दशांश* inp, दशांश* weight, दशांश* bias,
                       पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    दशांश eps = 1e-5f;
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            दशांश* x = inp + b * T * C + t * C;
            // calculate the mean
            दशांश m = 0.0f;
            साठी (पूर्णांक i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            दशांश v = 0.0f;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            दशांश s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            दशांश* out_bt = out + b * T * C + t * C;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश n = (s * (x[i] - m)); // normalized output
                दशांश o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

शून्य layernorm_backward(दशांश* dinp, दशांश* dweight, दशांश* dbias,
                        दशांश* dout, दशांश* inp, दशांश* weight, दशांश* mean, दशांश* rstd,
                        पूर्णांक B, पूर्णांक T, पूर्णांक C) {
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            दशांश* dout_bt = dout + b * T * C + t * C;
            दशांश* inp_bt = inp + b * T * C + t * C;
            दशांश* dinp_bt = dinp + b * T * C + t * C;
            दशांश mean_bt = mean[b * T + t];
            दशांश rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            दशांश dnorm_mean = 0.0f;
            दशांश dnorm_norm_mean = 0.0f;
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                दशांश dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                दशांश dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                दशांश dval = 0.0f;
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
पूर्णांक check_tensor(दशांश *a, दशांश *b, पूर्णांक n, अक्षर* label) {
    पूर्णांक ok = 1;
    म_लिखो("%s\n", label);
    साठी (पूर्णांक i = 0; i < n; i++) {
        जर (भ_असल(a[i] - b[i]) <= 1e-5) {
            म_लिखो("OK ");
        } अन्यथा {
            म_लिखो("NOT OK ");
            ok = 0;
        }
        म_लिखो("%f %f\n", a[i], b[i]);
    }
    परत ok;
}

पूर्णांक main() {

    पूर्णांक B = 2; // batch
    पूर्णांक T = 3; // time / sequence length
    पूर्णांक C = 4; // number of channels

    दशांश* x = (दशांश*) दो_स्मृति(B * T * C * आकार(दशांश));
    दशांश* w = (दशांश*) दो_स्मृति(C * आकार(दशांश));
    दशांश* b = (दशांश*) दो_स्मृति(C * आकार(दशांश));
    दशांश* out = (दशांश*) दो_स्मृति(B * T * C * आकार(दशांश));
    दशांश* mean = (दशांश*) दो_स्मृति(B * T * आकार(दशांश));
    दशांश* rstd = (दशांश*) दो_स्मृति(B * T * आकार(दशांश));
    दशांश* dout = (दशांश*) दो_स्मृति(B * T * C * आकार(दशांश));
    दशांश* dx = (दशांश*) दो_स्मृति(B * T * C * आकार(दशांश));
    दशांश* dw = (दशांश*) दो_स्मृति(C * आकार(दशांश));
    दशांश* db = (दशांश*) दो_स्मृति(C * आकार(दशांश));

    // read reference information from Python
    खाता *file = ख_खोलो("ln.bin", "rb");
    जर (file == रिक्त) {
        म_लिखो("Error opening file\n");
        परत 1;
    }
    ख_पढ़ो(x, आकार(दशांश), B * T * C, file);
    ख_पढ़ो(w, आकार(दशांश), C, file);
    ख_पढ़ो(b, आकार(दशांश), C, file);
    ख_पढ़ो(out, आकार(दशांश), B * T * C, file);
    ख_पढ़ो(mean, आकार(दशांश), B * T, file);
    ख_पढ़ो(rstd, आकार(दशांश), B * T, file);
    ख_पढ़ो(dout, आकार(दशांश), B * T * C, file);
    ख_पढ़ो(dx, आकार(दशांश), B * T * C, file);
    ख_पढ़ो(dw, आकार(दशांश), C, file);
    ख_पढ़ो(db, आकार(दशांश), C, file);
    ख_बंद(file);

    // now let's calculate everything ourselves

    // forward pass
    दशांश* c_out = (दशांश*) दो_स्मृति(B * T * C * आकार(दशांश));
    दशांश* c_mean = (दशांश*) दो_स्मृति(B * T * आकार(दशांश));
    दशांश* c_rstd = (दशांश*) दो_स्मृति(B * T * आकार(दशांश));
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // check correctness of forward pass
    check_tensor(out, c_out, B*T*C, "out");
    check_tensor(mean, c_mean, B*T, "mean");
    check_tensor(rstd, c_rstd, B*T, "rstd");

    // backward pass (note calloc inits grads to zero)
    दशांश* c_dx = (दशांश*) सुस्मृति(B * T * C, आकार(दशांश));
    दशांश* c_dw = (दशांश*) सुस्मृति(B * T, आकार(दशांश));
    दशांश* c_db = (दशांश*) सुस्मृति(B * T, आकार(दशांश));
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
    परत 0;
}
