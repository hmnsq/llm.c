#define TESTING
#अन्तर्भाव "train_gpt2.c"

// poor man's tensor checker
पूर्णाङ्क check_tensor(भिन्न *a, भिन्न *b, पूर्णाङ्क n, ध्रुव वर्ण* label) {
    पूर्णाङ्क print_upto = 5;
    पूर्णाङ्क ok = 1;
    भिन्न maxdiff = 0.0f;
    भिन्न tol = 2e-2f;
    printf("%s\n", label);
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        भिन्न diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        जइ (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        जइ (i < print_upto) {
            जइ (diff <= tol) {
                जइ (i < print_upto) { printf("OK "); }
            } अण्णहा {
                जइ (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    जइ (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } अण्णहा {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    प्रत्यागम ok;
}

पूर्णाङ्क main(पूर्णाङ्क argc, वर्ण *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    पूर्णाङ्क C = model.config.channels;
    पूर्णाङ्क V = model.config.vocab_size;
    पूर्णाङ्क Vp = model.config.padded_vocab_size;
    पूर्णाङ्क maxT = model.config.max_seq_len;
    पूर्णाङ्क L = model.config.अङ्कः_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    जइ (state_file == रिक्त) { printf("Error opening state file\n"); प्रत्यागम 1; }
    पूर्णाङ्क state_header[256];
    freadCheck(state_header, परिमाण(पूर्णाङ्क), 256, state_file);
    जइ (state_header[0] != 20240327) { printf("Bad magic state file\n"); प्रत्यागम 1; }
    जइ (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        प्रत्यागम 1;
    }
    पूर्णाङ्क B = state_header[2]; // batch size, e.g. 4
    पूर्णाङ्क T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    भिन्न* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    पूर्णाङ्क* x = (पूर्णाङ्क*) malloc(B * T * परिमाण(पूर्णाङ्क));
    पूर्णाङ्क* y = (पूर्णाङ्क*) malloc(B * T * परिमाण(पूर्णाङ्क));
    भिन्न* expected_logits = (भिन्न*) malloc(B * T * V * परिमाण(भिन्न));
    भिन्न* expected_loss = (भिन्न*) malloc(1 * परिमाण(भिन्न));

    // read reference information from Python
    freadCheck(x, परिमाण(पूर्णाङ्क), B*T, state_file);
    freadCheck(y, परिमाण(पूर्णाङ्क), B*T, state_file);
    freadCheck(expected_logits, परिमाण(भिन्न), B*T*V, state_file);
    freadCheck(expected_loss, परिमाण(भिन्न), 1, state_file);
    freadCheck(expected_grads_memory, परिमाण(भिन्न), model.अङ्कः_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    पूर्णाङ्क allok = 1;

    // let's do 10 training iterations, following the pytorch code
    भिन्न expected_losses[10] = {
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
    प्रति (पूर्णाङ्क step = 0; step < 10; step++) {

        संरचना timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुण time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        जइ (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            पूर्णाङ्क logits_ok = 1;
            भिन्न* calculated_logits = model.acts.logits;
            भिन्न max_diff = 0.0f;
            प्रति (पूर्णाङ्क bt = 0; bt < B*T; bt++) {
                प्रति (पूर्णाङ्क v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    पूर्णाङ्क i = bt * Vp + v; // linearized index, using Vp
                    जइ (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    भिन्न diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    जइ (diff >= 1e-2f) {
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        भङ्ग;
                    }
                }
            }
            जइ(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            जइ (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } अण्णहा {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            पूर्णाङ्क gradoks[16];
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
            प्रति (पूर्णाङ्क i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        भिन्न expected_loss = expected_losses[step];
        भिन्न actual_loss = model.mean_loss;
        पूर्णाङ्क step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // final judgement
    printf("overall okay: %d\n", allok);

    // free everything
    मुक्त(x);
    मुक्त(y);
    मुक्त(expected_logits);
    मुक्त(expected_loss);
    मुक्त(expected_grads_memory);
    gpt2_free(&model);
    प्रत्यागम 0;
}
