#define TESTING
#അന്തര്ഭാവ "train_gpt2.c"

// poor man's tensor checker
പൂർണ്ണസംഖ്യ check_tensor(ദശാംശം *a, ദശാംശം *b, പൂർണ്ണസംഖ്യ n, സ്ഥിരം അക്ഷരം* label) {
    പൂർണ്ണസംഖ്യ print_upto = 5;
    പൂർണ്ണസംഖ്യ ok = 1;
    ദശാംശം maxdiff = 0.0f;
    ദശാംശം tol = 2e-2f;
    printf("%s\n", label);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        ദശാംശം diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        എങ്കിൽ (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        എങ്കിൽ (i < print_upto) {
            എങ്കിൽ (diff <= tol) {
                എങ്കിൽ (i < print_upto) { printf("OK "); }
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    എങ്കിൽ (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } അല്ലെങ്കിൽ {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    തിരികെ ok;
}

പൂർണ്ണസംഖ്യ പ്രധാന(പൂർണ്ണസംഖ്യ argc, അക്ഷരം *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    പൂർണ്ണസംഖ്യ C = model.config.channels;
    പൂർണ്ണസംഖ്യ V = model.config.vocab_size;
    പൂർണ്ണസംഖ്യ Vp = model.config.padded_vocab_size;
    പൂർണ്ണസംഖ്യ maxT = model.config.max_seq_len;
    പൂർണ്ണസംഖ്യ L = model.config.അക്കം_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    എങ്കിൽ (state_file == ശൂന്യ) { printf("Error opening state file\n"); തിരികെ 1; }
    പൂർണ്ണസംഖ്യ state_header[256];
    freadCheck(state_header, വലുപ്പം(പൂർണ്ണസംഖ്യ), 256, state_file);
    എങ്കിൽ (state_header[0] != 20240327) { printf("Bad magic state file\n"); തിരികെ 1; }
    എങ്കിൽ (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        തിരികെ 1;
    }
    പൂർണ്ണസംഖ്യ B = state_header[2]; // batch size, e.g. 4
    പൂർണ്ണസംഖ്യ T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    ദശാംശം* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    പൂർണ്ണസംഖ്യ* x = (പൂർണ്ണസംഖ്യ*) malloc(B * T * വലുപ്പം(പൂർണ്ണസംഖ്യ));
    പൂർണ്ണസംഖ്യ* y = (പൂർണ്ണസംഖ്യ*) malloc(B * T * വലുപ്പം(പൂർണ്ണസംഖ്യ));
    ദശാംശം* expected_logits = (ദശാംശം*) malloc(B * T * V * വലുപ്പം(ദശാംശം));
    ദശാംശം* expected_loss = (ദശാംശം*) malloc(1 * വലുപ്പം(ദശാംശം));

    // read reference information from Python
    freadCheck(x, വലുപ്പം(പൂർണ്ണസംഖ്യ), B*T, state_file);
    freadCheck(y, വലുപ്പം(പൂർണ്ണസംഖ്യ), B*T, state_file);
    freadCheck(expected_logits, വലുപ്പം(ദശാംശം), B*T*V, state_file);
    freadCheck(expected_loss, വലുപ്പം(ദശാംശം), 1, state_file);
    freadCheck(expected_grads_memory, വലുപ്പം(ദശാംശം), model.അക്കം_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    പൂർണ്ണസംഖ്യ allok = 1;

    // let's do 10 training iterations, following the pytorch code
    ദശാംശം expected_losses[10] = {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ step = 0; step < 10; step++) {

        ഘടന timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        ഇരട്ട time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        എങ്കിൽ (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            പൂർണ്ണസംഖ്യ logits_ok = 1;
            ദശാംശം* calculated_logits = model.acts.logits;
            ദശാംശം max_diff = 0.0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ bt = 0; bt < B*T; bt++) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    പൂർണ്ണസംഖ്യ i = bt * Vp + v; // linearized index, using Vp
                    എങ്കിൽ (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    ദശാംശം diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    എങ്കിൽ (diff >= 1e-2f) {
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        നിർത്തുക;
                    }
                }
            }
            എങ്കിൽ(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            എങ്കിൽ (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } അല്ലെങ്കിൽ {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            പൂർണ്ണസംഖ്യ gradoks[16];
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        ദശാംശം expected_loss = expected_losses[step];
        ദശാംശം actual_loss = model.mean_loss;
        പൂർണ്ണസംഖ്യ step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // final judgement
    printf("overall okay: %d\n", allok);

    // free everything
    സ്വതന്ത്രം(x);
    സ്വതന്ത്രം(y);
    സ്വതന്ത്രം(expected_logits);
    സ്വതന്ത്രം(expected_loss);
    സ്വതന്ത്രം(expected_grads_memory);
    gpt2_free(&model);
    തിരികെ 0;
}
