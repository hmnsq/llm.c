#تعریف_کرو TESTING
#ضمن "train_gpt2.c"

// poor man's tensor checker
صحيح check_tensor(عائم *a, عائم *b, صحيح n, ثابت حرف* label) {
    صحيح print_upto = 5;
    صحيح ok = 1;
    عائم maxdiff = 0.0f;
    عائم tol = 2e-2f;
    لکھو("%s\n", label);
    لكل (صحيح i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        عائم diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        إذا (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        إذا (i < print_upto) {
            إذا (diff <= tol) {
                إذا (i < print_upto) { لکھو("OK "); }
            } وإلا {
                إذا (i < print_upto) { لکھو("NOT OK "); }
            }
            لکھو("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    إذا (ok) {
        لکھو("TENSOR OK, maxdiff = %e\n", maxdiff);
    } وإلا {
        لکھو("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    أرجع ok;
}

صحيح مکھ(صحيح argc, حرف *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    صحيح C = model.config.channels;
    صحيح V = model.config.vocab_ناپ;
    صحيح Vp = model.config.padded_vocab_ناپ;
    صحيح maxT = model.config.max_seq_len;
    صحيح L = model.config.عدد_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    إذا (state_file == عدم) { لکھو("Error opening state file\n"); أرجع 1; }
    صحيح state_سرخی[256];
    freadCheck(state_سرخی, حجم(صحيح), 256, state_file);
    إذا (state_سرخی[0] != 20240327) { لکھو("Bad magic state file\n"); أرجع 1; }
    إذا (state_سرخی[1] != 2) {
        لکھو("Bad version in state file\n");
        لکھو("---> HINT: try to re-run `python train_gpt2.py`\n");
        أرجع 1;
    }
    صحيح B = state_سرخی[2]; // batch size, e.g. 4
    صحيح T = state_سرخی[3]; // time / sequence length (e.g. 64, up to maxT)
    لکھو("[State]\n");
    لکھو("batch_size: %d\n", B);
    لکھو("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    عائم* expected_grads_حافظہ = malloc_and_نقطہ_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    صحيح* x = (صحيح*) حافظہ_دیو(B * T * حجم(صحيح));
    صحيح* y = (صحيح*) حافظہ_دیو(B * T * حجم(صحيح));
    عائم* expected_logits = (عائم*) حافظہ_دیو(B * T * V * حجم(عائم));
    عائم* expected_loss = (عائم*) حافظہ_دیو(1 * حجم(عائم));

    // read reference information from Python
    freadCheck(x, حجم(صحيح), B*T, state_file);
    freadCheck(y, حجم(صحيح), B*T, state_file);
    freadCheck(expected_logits, حجم(عائم), B*T*V, state_file);
    freadCheck(expected_loss, حجم(عائم), 1, state_file);
    freadCheck(expected_grads_حافظہ, حجم(عائم), model.عدد_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    صحيح allok = 1;

    // let's do 10 training iterations, following the pytorch code
    عائم expected_losses[10] = {
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
    لكل (صحيح قدم = 0; قدم < 10; قدم++) {

        بنية timespec شروع, سرا;
        clock_gettime(CLOCK_MONOTONIC, &شروع);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &سرا);
        مزدوج ویلا_elapsed_s = (سرا.tv_sec - شروع.tv_sec) + (سرا.tv_nsec - شروع.tv_nsec) / 1e9;

        إذا (قدم == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            صحيح logits_ok = 1;
            عائم* calculated_logits = model.acts.logits;
            عائم max_diff = 0.0f;
            لكل (صحيح bt = 0; bt < B*T; bt++) {
                لكل (صحيح v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    صحيح i = bt * Vp + v; // linearized index, using Vp
                    إذا (i < 10) {
                        لکھو("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    عائم diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    إذا (diff >= 1e-2f) {
                        لکھو("MISMATCH AT INDEX %d,%d: ", bt, v);
                        لکھو("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        اكسر;
                    }
                }
            }
            إذا(!logits_ok) { لکھو("NOT "); }
            لکھو("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            إذا (fabsf(model.اوسط_loss - *expected_loss) >= 1e-2) {
                لکھو("LOSS MISMATCH: %f %f\n", model.اوسط_loss, *expected_loss);
                allok = 0;
            } وإلا {
                لکھو("LOSS OK: %f %f\n", model.اوسط_loss, *expected_loss);
            }

            // finally check all the gradients
            صحيح gradoks[16];
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
            لكل (صحيح i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, قدم+1);

        // compare the losses
        عائم expected_loss = expected_losses[قدم];
        عائم actual_loss = model.اوسط_loss;
        صحيح قدم_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && قدم_loss_ok;

        // print the timing information at the end
        لکھو("step %d: loss %f (took %f ms) OK = %d\n", قدم, model.اوسط_loss, ویلا_elapsed_s * 1000, قدم_loss_ok);
    }

    // final judgement
    لکھو("overall okay: %d\n", allok);

    // free everything
    حرر(x);
    حرر(y);
    حرر(expected_logits);
    حرر(expected_loss);
    حرر(expected_grads_حافظہ);
    gpt2_free(&model);
    أرجع 0;
}
