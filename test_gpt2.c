#ವ್ಯಾಖ್ಯಾನಿಸು TESTING
#ಅನ್ತರ್ಭಾವ "train_gpt2.c"

// poor man's tensor checker
ಪೂರ್ಣಾಂಕ check_tensor(ದಶಮಾಂಶ *a, ದಶಮಾಂಶ *b, ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* label) {
    ಪೂರ್ಣಾಂಕ print_upto = 5;
    ಪೂರ್ಣಾಂಕ ok = 1;
    ದಶಮಾಂಶ maxdiff = 0.0f;
    ದಶಮಾಂಶ tol = 2e-2f;
    ಬರೆ("%s\n", label);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        ದಶಮಾಂಶ diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        ಆದರೆ (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        ಆದರೆ (i < print_upto) {
            ಆದರೆ (diff <= tol) {
                ಆದರೆ (i < print_upto) { ಬರೆ("OK "); }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (i < print_upto) { ಬರೆ("NOT OK "); }
            }
            ಬರೆ("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    ಆದರೆ (ok) {
        ಬರೆ("TENSOR OK, maxdiff = %e\n", maxdiff);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಬರೆ("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    ಹಿಂತಿರುಗಿ ok;
}

ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ(ಪೂರ್ಣಾಂಕ argc, ಅಕ್ಷರ *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    ಪೂರ್ಣಾಂಕ C = model.config.channels;
    ಪೂರ್ಣಾಂಕ V = model.config.vocab_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ Vp = model.config.padded_vocab_ಗಾತ್ರ;
    ಪೂರ್ಣಾಂಕ maxT = model.config.max_seq_len;
    ಪೂರ್ಣಾಂಕ L = model.config.ಅಂಕೆ_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    ಆದರೆ (state_file == ಖಾಲಿ) { ಬರೆ("Error opening state file\n"); ಹಿಂತಿರುಗಿ 1; }
    ಪೂರ್ಣಾಂಕ state_ಶೀರ್ಷಿಕೆ[256];
    freadCheck(state_ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), 256, state_file);
    ಆದರೆ (state_ಶೀರ್ಷಿಕೆ[0] != 20240327) { ಬರೆ("Bad magic state file\n"); ಹಿಂತಿರುಗಿ 1; }
    ಆದರೆ (state_ಶೀರ್ಷಿಕೆ[1] != 2) {
        ಬರೆ("Bad version in state file\n");
        ಬರೆ("---> HINT: try to re-run `python train_gpt2.py`\n");
        ಹಿಂತಿರುಗಿ 1;
    }
    ಪೂರ್ಣಾಂಕ B = state_ಶೀರ್ಷಿಕೆ[2]; // batch size, e.g. 4
    ಪೂರ್ಣಾಂಕ T = state_ಶೀರ್ಷಿಕೆ[3]; // time / sequence length (e.g. 64, up to maxT)
    ಬರೆ("[State]\n");
    ಬರೆ("batch_size: %d\n", B);
    ಬರೆ("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    ದಶಮಾಂಶ* expected_grads_ಸ್ಮರಣೆ = malloc_and_ಬಿಂದು_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    ಪೂರ್ಣಾಂಕ* x = (ಪೂರ್ಣಾಂಕ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಪೂರ್ಣಾಂಕ* y = (ಪೂರ್ಣಾಂಕ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ದಶಮಾಂಶ* expected_logits = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(B * T * V * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ದಶಮಾಂಶ* expected_loss = (ದಶಮಾಂಶ*) ಸ್ಮರಣೆ_ಕೊಡು(1 * ಗಾತ್ರ(ದಶಮಾಂಶ));

    // read reference information from Python
    freadCheck(x, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), B*T, state_file);
    freadCheck(y, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), B*T, state_file);
    freadCheck(expected_logits, ಗಾತ್ರ(ದಶಮಾಂಶ), B*T*V, state_file);
    freadCheck(expected_loss, ಗಾತ್ರ(ದಶಮಾಂಶ), 1, state_file);
    freadCheck(expected_grads_ಸ್ಮರಣೆ, ಗಾತ್ರ(ದಶಮಾಂಶ), model.ಅಂಕೆ_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    ಪೂರ್ಣಾಂಕ allok = 1;

    // let's do 10 training iterations, following the pytorch code
    ದಶಮಾಂಶ expected_losses[10] = {
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
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < 10; ಹೆಜ್ಜೆ++) {

        ರಚನೆ timespec ಆರಂಭ, ಕೊನೆ;
        clock_gettime(CLOCK_MONOTONIC, &ಆರಂಭ);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &ಕೊನೆ);
        ದ್ವಿಗುಣ ಸಮಯ_elapsed_s = (ಕೊನೆ.tv_sec - ಆರಂಭ.tv_sec) + (ಕೊನೆ.tv_nsec - ಆರಂಭ.tv_nsec) / 1e9;

        ಆದರೆ (ಹೆಜ್ಜೆ == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            ಪೂರ್ಣಾಂಕ logits_ok = 1;
            ದಶಮಾಂಶ* calculated_logits = model.acts.logits;
            ದಶಮಾಂಶ max_diff = 0.0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ bt = 0; bt < B*T; bt++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    ಪೂರ್ಣಾಂಕ i = bt * Vp + v; // linearized index, using Vp
                    ಆದರೆ (i < 10) {
                        ಬರೆ("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    ದಶಮಾಂಶ diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    ಆದರೆ (diff >= 1e-2f) {
                        ಬರೆ("MISMATCH AT INDEX %d,%d: ", bt, v);
                        ಬರೆ("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
            ಆದರೆ(!logits_ok) { ಬರೆ("NOT "); }
            ಬರೆ("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            ಆದರೆ (fabsf(model.ಸರಾಸರಿ_loss - *expected_loss) >= 1e-2) {
                ಬರೆ("LOSS MISMATCH: %f %f\n", model.ಸರಾಸರಿ_loss, *expected_loss);
                allok = 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಬರೆ("LOSS OK: %f %f\n", model.ಸರಾಸರಿ_loss, *expected_loss);
            }

            // finally check all the gradients
            ಪೂರ್ಣಾಂಕ gradoks[16];
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
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, ಹೆಜ್ಜೆ+1);

        // compare the losses
        ದಶಮಾಂಶ expected_loss = expected_losses[ಹೆಜ್ಜೆ];
        ದಶಮಾಂಶ actual_loss = model.ಸರಾಸರಿ_loss;
        ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && ಹೆಜ್ಜೆ_loss_ok;

        // print the timing information at the end
        ಬರೆ("step %d: loss %f (took %f ms) OK = %d\n", ಹೆಜ್ಜೆ, model.ಸರಾಸರಿ_loss, ಸಮಯ_elapsed_s * 1000, ಹೆಜ್ಜೆ_loss_ok);
    }

    // final judgement
    ಬರೆ("overall okay: %d\n", allok);

    // free everything
    ಬಿಡುಗಡೆ(x);
    ಬಿಡುಗಡೆ(y);
    ಬಿಡುಗಡೆ(expected_logits);
    ಬಿಡುಗಡೆ(expected_loss);
    ಬಿಡುಗಡೆ(expected_grads_ಸ್ಮರಣೆ);
    gpt2_free(&model);
    ಹಿಂತಿರುಗಿ 0;
}
