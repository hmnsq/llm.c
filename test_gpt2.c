#define TESTING
#अन्तर्भाव "train_gpt2.c"

// poor man's tensor checker
पूर्णांक check_tensor(दशांश *a, दशांश *b, पूर्णांक n, स्थिरांक अक्षर* label) {
    पूर्णांक print_upto = 5;
    पूर्णांक ok = 1;
    दशांश maxdiff = 0.0f;
    दशांश tol = 2e-2f;
    म_लिखो("%s\n", label);
    साठी (पूर्णांक i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        दशांश diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        जर (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        जर (i < print_upto) {
            जर (diff <= tol) {
                जर (i < print_upto) { म_लिखो("OK "); }
            } अन्यथा {
                जर (i < print_upto) { म_लिखो("NOT OK "); }
            }
            म_लिखो("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    जर (ok) {
        म_लिखो("TENSOR OK, maxdiff = %e\n", maxdiff);
    } अन्यथा {
        म_लिखो("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    परत ok;
}

पूर्णांक main(पूर्णांक argc, अक्षर *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    पूर्णांक C = model.config.channels;
    पूर्णांक V = model.config.vocab_size;
    पूर्णांक Vp = model.config.padded_vocab_size;
    पूर्णांक maxT = model.config.max_seq_len;
    पूर्णांक L = model.config.अंक_layers;

    // load additional information that we will use for debugging and error checking
    खाता *state_file = ख_खोलो("gpt2_124M_debug_state.bin", "rb");
    जर (state_file == रिक्त) { म_लिखो("Error opening state file\n"); परत 1; }
    पूर्णांक state_header[256];
    freadCheck(state_header, आकार(पूर्णांक), 256, state_file);
    जर (state_header[0] != 20240327) { म_लिखो("Bad magic state file\n"); परत 1; }
    जर (state_header[1] != 2) {
        म_लिखो("Bad version in state file\n");
        म_लिखो("---> HINT: try to re-run `python train_gpt2.py`\n");
        परत 1;
    }
    पूर्णांक B = state_header[2]; // batch size, e.g. 4
    पूर्णांक T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    म_लिखो("[State]\n");
    म_लिखो("batch_size: %d\n", B);
    म_लिखो("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    दशांश* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    पूर्णांक* x = (पूर्णांक*) दो_स्मृति(B * T * आकार(पूर्णांक));
    पूर्णांक* y = (पूर्णांक*) दो_स्मृति(B * T * आकार(पूर्णांक));
    दशांश* expected_logits = (दशांश*) दो_स्मृति(B * T * V * आकार(दशांश));
    दशांश* expected_loss = (दशांश*) दो_स्मृति(1 * आकार(दशांश));

    // read reference information from Python
    freadCheck(x, आकार(पूर्णांक), B*T, state_file);
    freadCheck(y, आकार(पूर्णांक), B*T, state_file);
    freadCheck(expected_logits, आकार(दशांश), B*T*V, state_file);
    freadCheck(expected_loss, आकार(दशांश), 1, state_file);
    freadCheck(expected_grads_memory, आकार(दशांश), model.अंक_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    पूर्णांक allok = 1;

    // let's do 10 training iterations, following the pytorch code
    दशांश expected_losses[10] = {
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
    साठी (पूर्णांक step = 0; step < 10; step++) {

        संरचना timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुण time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        जर (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            पूर्णांक logits_ok = 1;
            दशांश* calculated_logits = model.acts.logits;
            दशांश max_diff = 0.0f;
            साठी (पूर्णांक bt = 0; bt < B*T; bt++) {
                साठी (पूर्णांक v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    पूर्णांक i = bt * Vp + v; // linearized index, using Vp
                    जर (i < 10) {
                        म_लिखो("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    दशांश diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    जर (diff >= 1e-2f) {
                        म_लिखो("MISMATCH AT INDEX %d,%d: ", bt, v);
                        म_लिखो("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        थांबा;
                    }
                }
            }
            जर(!logits_ok) { म_लिखो("NOT "); }
            म_लिखो("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            जर (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                म_लिखो("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } अन्यथा {
                म_लिखो("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            पूर्णांक gradoks[16];
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
            साठी (पूर्णांक i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        दशांश expected_loss = expected_losses[step];
        दशांश actual_loss = model.mean_loss;
        पूर्णांक step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        म_लिखो("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // final judgement
    म_लिखो("overall okay: %d\n", allok);

    // free everything
    मुक्त(x);
    मुक्त(y);
    मुक्त(expected_logits);
    मुक्त(expected_loss);
    मुक्त(expected_grads_memory);
    gpt2_free(&model);
    परत 0;
}
