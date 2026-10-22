#define TESTING
#અન્તર્ભાવ "train_gpt2.c"

// poor man's tensor checker
પૂર્ણાંક check_tensor(દશાંશ *a, દશાંશ *b, પૂર્ણાંક n, અચલ અક્ષર* label) {
    પૂર્ણાંક print_upto = 5;
    પૂર્ણાંક ok = 1;
    દશાંશ maxdiff = 0.0f;
    દશાંશ tol = 2e-2f;
    printf("%s\n", label);
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        દશાંશ diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        જો (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        જો (i < print_upto) {
            જો (diff <= tol) {
                જો (i < print_upto) { printf("OK "); }
            } નહીંતર {
                જો (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    જો (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } નહીંતર {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    પાછા ok;
}

પૂર્ણાંક મુખ્ય(પૂર્ણાંક argc, અક્ષર *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    પૂર્ણાંક C = model.config.channels;
    પૂર્ણાંક V = model.config.vocab_size;
    પૂર્ણાંક Vp = model.config.padded_vocab_size;
    પૂર્ણાંક maxT = model.config.max_seq_len;
    પૂર્ણાંક L = model.config.અંક_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    જો (state_file == ખાલી) { printf("Error opening state file\n"); પાછા 1; }
    પૂર્ણાંક state_header[256];
    freadCheck(state_header, કદ(પૂર્ણાંક), 256, state_file);
    જો (state_header[0] != 20240327) { printf("Bad magic state file\n"); પાછા 1; }
    જો (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        પાછા 1;
    }
    પૂર્ણાંક B = state_header[2]; // batch size, e.g. 4
    પૂર્ણાંક T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    દશાંશ* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    પૂર્ણાંક* x = (પૂર્ણાંક*) malloc(B * T * કદ(પૂર્ણાંક));
    પૂર્ણાંક* y = (પૂર્ણાંક*) malloc(B * T * કદ(પૂર્ણાંક));
    દશાંશ* expected_logits = (દશાંશ*) malloc(B * T * V * કદ(દશાંશ));
    દશાંશ* expected_loss = (દશાંશ*) malloc(1 * કદ(દશાંશ));

    // read reference information from Python
    freadCheck(x, કદ(પૂર્ણાંક), B*T, state_file);
    freadCheck(y, કદ(પૂર્ણાંક), B*T, state_file);
    freadCheck(expected_logits, કદ(દશાંશ), B*T*V, state_file);
    freadCheck(expected_loss, કદ(દશાંશ), 1, state_file);
    freadCheck(expected_grads_memory, કદ(દશાંશ), model.અંક_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    પૂર્ણાંક allok = 1;

    // let's do 10 training iterations, following the pytorch code
    દશાંશ expected_losses[10] = {
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
    માટે (પૂર્ણાંક step = 0; step < 10; step++) {

        સંરચના timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        દ્વિગુણ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        જો (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            પૂર્ણાંક logits_ok = 1;
            દશાંશ* calculated_logits = model.acts.logits;
            દશાંશ max_diff = 0.0f;
            માટે (પૂર્ણાંક bt = 0; bt < B*T; bt++) {
                માટે (પૂર્ણાંક v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    પૂર્ણાંક i = bt * Vp + v; // linearized index, using Vp
                    જો (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    દશાંશ diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    જો (diff >= 1e-2f) {
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        તોડો;
                    }
                }
            }
            જો(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            જો (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } નહીંતર {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            પૂર્ણાંક gradoks[16];
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
            માટે (પૂર્ણાંક i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        દશાંશ expected_loss = expected_losses[step];
        દશાંશ actual_loss = model.mean_loss;
        પૂર્ણાંક step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // final judgement
    printf("overall okay: %d\n", allok);

    // free everything
    મુક્ત(x);
    મુક્ત(y);
    મુક્ત(expected_logits);
    મુક્ત(expected_loss);
    મુક્ત(expected_grads_memory);
    gpt2_free(&model);
    પાછા 0;
}
