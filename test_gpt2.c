#تعریف_کرو TESTING
#شامل "train_gpt2.c"

// poor man's tensor checker
صحیح check_tensor(اعشاری *a, اعشاری *b, صحیح n, ثابت حرف* label) {
    صحیح print_upto = 5;
    صحیح ok = 1;
    اعشاری maxdiff = 0.0f;
    اعشاری tol = 2e-2f;
    لکھو("%s\n", label);
    برائے (صحیح i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        اعشاری diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        اگر (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        اگر (i < print_upto) {
            اگر (diff <= tol) {
                اگر (i < print_upto) { لکھو("OK "); }
            } ورنہ {
                اگر (i < print_upto) { لکھو("NOT OK "); }
            }
            لکھو("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    اگر (ok) {
        لکھو("TENSOR OK, maxdiff = %e\n", maxdiff);
    } ورنہ {
        لکھو("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    واپس ok;
}

صحیح مین(صحیح argc, حرف *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    صحیح C = model.config.channels;
    صحیح V = model.config.vocab_سائز;
    صحیح Vp = model.config.padded_vocab_سائز;
    صحیح maxT = model.config.max_seq_len;
    صحیح L = model.config.عدد_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    اگر (state_file == تہی) { لکھو("Error opening state file\n"); واپس 1; }
    صحیح state_سرخی[256];
    freadCheck(state_سرخی, حجم(صحیح), 256, state_file);
    اگر (state_سرخی[0] != 20240327) { لکھو("Bad magic state file\n"); واپس 1; }
    اگر (state_سرخی[1] != 2) {
        لکھو("Bad version in state file\n");
        لکھو("---> HINT: try to re-run `python train_gpt2.py`\n");
        واپس 1;
    }
    صحیح B = state_سرخی[2]; // batch size, e.g. 4
    صحیح T = state_سرخی[3]; // time / sequence length (e.g. 64, up to maxT)
    لکھو("[State]\n");
    لکھو("batch_size: %d\n", B);
    لکھو("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    اعشاری* expected_grads_حافظہ = malloc_and_نقطہ_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    صحیح* x = (صحیح*) حافظہ_دو(B * T * حجم(صحیح));
    صحیح* y = (صحیح*) حافظہ_دو(B * T * حجم(صحیح));
    اعشاری* expected_logits = (اعشاری*) حافظہ_دو(B * T * V * حجم(اعشاری));
    اعشاری* expected_loss = (اعشاری*) حافظہ_دو(1 * حجم(اعشاری));

    // read reference information from Python
    freadCheck(x, حجم(صحیح), B*T, state_file);
    freadCheck(y, حجم(صحیح), B*T, state_file);
    freadCheck(expected_logits, حجم(اعشاری), B*T*V, state_file);
    freadCheck(expected_loss, حجم(اعشاری), 1, state_file);
    freadCheck(expected_grads_حافظہ, حجم(اعشاری), model.عدد_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    صحیح allok = 1;

    // let's do 10 training iterations, following the pytorch code
    اعشاری expected_losses[10] = {
        5.270007133483887f,
        4.059706687927246f,
        3.3751230239868164f,
        2.8007826805114746f,
        2.315382242202759f,
        1.8490285873413086f,
        1.3946564197540283f,
        0.9991465210914612f,
        0.6240804195404053f,
        0.37651097774505615f
    };
    برائے (صحیح قدم = 0; قدم < 10; قدم++) {

        ساخت timespec آغاز, انتہا;
        clock_gettime(CLOCK_MONOTONIC, &آغاز);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &انتہا);
        دوہرا وقت_elapsed_s = (انتہا.tv_sec - آغاز.tv_sec) + (انتہا.tv_nsec - آغاز.tv_nsec) / 1e9;

        اگر (قدم == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            صحیح logits_ok = 1;
            اعشاری* calculated_logits = model.acts.logits;
            اعشاری max_diff = 0.0f;
            برائے (صحیح bt = 0; bt < B*T; bt++) {
                برائے (صحیح v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    صحیح i = bt * Vp + v; // linearized index, using Vp
                    اگر (i < 10) {
                        لکھو("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    اعشاری diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    اگر (diff >= 1e-2f) {
                        لکھو("MISMATCH AT INDEX %d,%d: ", bt, v);
                        لکھو("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        توڑو;
                    }
                }
            }
            اگر(!logits_ok) { لکھو("NOT "); }
            لکھو("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            اگر (fabsf(model.اوسط_loss - *expected_loss) >= 1e-2) {
                لکھو("LOSS MISMATCH: %f %f\n", model.اوسط_loss, *expected_loss);
                allok = 0;
            } ورنہ {
                لکھو("LOSS OK: %f %f\n", model.اوسط_loss, *expected_loss);
            }

            // finally check all the gradients
            صحیح gradoks[16];
            ParameterTensors grads = model.grads;
            gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V*C, "dwte");
            gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT*C, "dwpe");
            gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L*C, "dln1w");
            gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L*C, "dln1b");
            gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L*3*C*C, "dqkvw");
            gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L*3*C, "dqkvb");
            gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L*C*C, "dattprojw");
            gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L*C, "dattprojb");
            gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L*C, "dln2w");
            gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L*C, "dln2b");
            gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L*4*C*C, "dfcw");
            gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L*4*C, "dfcb");
            gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L*C*4*C, "dfcprojw");
            gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L*C, "dfcprojb");
            gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
            gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
            برائے (صحیح i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, قدم+1);

        // compare the losses
        اعشاری expected_loss = expected_losses[قدم];
        اعشاری actual_loss = model.اوسط_loss;
        صحیح قدم_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && قدم_loss_ok;

        // print the timing information at the end
        لکھو("step %d: loss %f (took %f ms) OK = %d\n", قدم, model.اوسط_loss, وقت_elapsed_s * 1000, قدم_loss_ok);
    }

    // final judgement
    لکھو("overall okay: %d\n", allok);

    // free everything
    آزاد(x);
    آزاد(y);
    آزاد(expected_logits);
    آزاد(expected_loss);
    آزاد(expected_grads_حافظہ);
    gpt2_free(&model);
    واپس 0;
}
