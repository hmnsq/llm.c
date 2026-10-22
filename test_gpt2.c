#define TESTING
#ಅನ್ತರ್ಭಾವ "train_gpt2.c"

// poor man's tensor checker
ಪೂರ್ಣಾಙ್ಕ check_tensor(ಭಿನ್ನ *a, ಭಿನ್ನ *b, ಪೂರ್ಣಾಙ್ಕ n, ಧ್ರುವ ವರ್ಣ* label) {
    ಪೂರ್ಣಾಙ್ಕ print_upto = 5;
    ಪೂರ್ಣಾಙ್ಕ ok = 1;
    ಭಿನ್ನ maxdiff = 0.0f;
    ಭಿನ್ನ tol = 2e-2f;
    printf("%s\n", label);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        ಭಿನ್ನ diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        ಯದಿ (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        ಯದಿ (i < print_upto) {
            ಯದಿ (diff <= tol) {
                ಯದಿ (i < print_upto) { printf("OK "); }
            } ಅನ್ಯಥಾ {
                ಯದಿ (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    ಯದಿ (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } ಅನ್ಯಥಾ {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    ಪ್ರತ್ಯಾಗಮ ok;
}

ಪೂರ್ಣಾಙ್ಕ main(ಪೂರ್ಣಾಙ್ಕ argc, ವರ್ಣ *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    ಪೂರ್ಣಾಙ್ಕ C = model.config.channels;
    ಪೂರ್ಣಾಙ್ಕ V = model.config.vocab_size;
    ಪೂರ್ಣಾಙ್ಕ Vp = model.config.padded_vocab_size;
    ಪೂರ್ಣಾಙ್ಕ maxT = model.config.max_seq_len;
    ಪೂರ್ಣಾಙ್ಕ L = model.config.ಅಂಕೆ_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    ಯದಿ (state_file == ರಿಕ್ತ) { printf("Error opening state file\n"); ಪ್ರತ್ಯಾಗಮ 1; }
    ಪೂರ್ಣಾಙ್ಕ state_header[256];
    freadCheck(state_header, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), 256, state_file);
    ಯದಿ (state_header[0] != 20240327) { printf("Bad magic state file\n"); ಪ್ರತ್ಯಾಗಮ 1; }
    ಯದಿ (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        ಪ್ರತ್ಯಾಗಮ 1;
    }
    ಪೂರ್ಣಾಙ್ಕ B = state_header[2]; // batch size, e.g. 4
    ಪೂರ್ಣಾಙ್ಕ T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    ಭಿನ್ನ* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    ಪೂರ್ಣಾಙ್ಕ* x = (ಪೂರ್ಣಾಙ್ಕ*) malloc(B * T * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪೂರ್ಣಾಙ್ಕ* y = (ಪೂರ್ಣಾಙ್ಕ*) malloc(B * T * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಭಿನ್ನ* expected_logits = (ಭಿನ್ನ*) malloc(B * T * V * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಭಿನ್ನ* expected_loss = (ಭಿನ್ನ*) malloc(1 * ಪರಿಮಾಣ(ಭಿನ್ನ));

    // read reference information from Python
    freadCheck(x, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), B*T, state_file);
    freadCheck(y, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), B*T, state_file);
    freadCheck(expected_logits, ಪರಿಮಾಣ(ಭಿನ್ನ), B*T*V, state_file);
    freadCheck(expected_loss, ಪರಿಮಾಣ(ಭಿನ್ನ), 1, state_file);
    freadCheck(expected_grads_memory, ಪರಿಮಾಣ(ಭಿನ್ನ), model.ಅಂಕೆ_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    ಪೂರ್ಣಾಙ್ಕ allok = 1;

    // let's do 10 training iterations, following the pytorch code
    ಭಿನ್ನ expected_losses[10] = {
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
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ step = 0; step < 10; step++) {

        ಸಂರಚನಾ timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        ದ್ವಿಗುಣ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        ಯದಿ (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            ಪೂರ್ಣಾಙ್ಕ logits_ok = 1;
            ಭಿನ್ನ* calculated_logits = model.acts.logits;
            ಭಿನ್ನ max_diff = 0.0f;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ bt = 0; bt < B*T; bt++) {
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    ಪೂರ್ಣಾಙ್ಕ i = bt * Vp + v; // linearized index, using Vp
                    ಯದಿ (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    ಭಿನ್ನ diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    ಯದಿ (diff >= 1e-2f) {
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        ಭಙ್ಗ;
                    }
                }
            }
            ಯದಿ(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            ಯದಿ (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } ಅನ್ಯಥಾ {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            ಪೂರ್ಣಾಙ್ಕ gradoks[16];
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
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        ಭಿನ್ನ expected_loss = expected_losses[step];
        ಭಿನ್ನ actual_loss = model.mean_loss;
        ಪೂರ್ಣಾಙ್ಕ step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // final judgement
    printf("overall okay: %d\n", allok);

    // free everything
    ಮುಕ್ತ(x);
    ಮುಕ್ತ(y);
    ಮುಕ್ತ(expected_logits);
    ಮುಕ್ತ(expected_loss);
    ಮುಕ್ತ(expected_grads_memory);
    gpt2_free(&model);
    ಪ್ರತ್ಯಾಗಮ 0;
}
