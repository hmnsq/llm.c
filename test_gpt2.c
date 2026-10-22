#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TESTING
#ਅਨ੍ਤਰ੍ਭਾਵ "train_gpt2.c"

// poor man's tensor checker
ਪੂਰਨ check_tensor(ਦਸ਼ਮਲਵ *a, ਦਸ਼ਮਲਵ *b, ਪੂਰਨ n, ਅਚਲ ਅੱਖਰ* label) {
    ਪੂਰਨ print_upto = 5;
    ਪੂਰਨ ok = 1;
    ਦਸ਼ਮਲਵ maxdiff = 0.0f;
    ਦਸ਼ਮਲਵ tol = 2e-2f;
    ਲਿਖੋ("%s\n", label);
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        ਦਸ਼ਮਲਵ diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        ਜੇ (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        ਜੇ (i < print_upto) {
            ਜੇ (diff <= tol) {
                ਜੇ (i < print_upto) { ਲਿਖੋ("OK "); }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (i < print_upto) { ਲਿਖੋ("NOT OK "); }
            }
            ਲਿਖੋ("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    ਜੇ (ok) {
        ਲਿਖੋ("TENSOR OK, maxdiff = %e\n", maxdiff);
    } ਨਹੀਂਤਾਂ {
        ਲਿਖੋ("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    ਵਾਪਸ ok;
}

ਪੂਰਨ ਮੁੱਖ(ਪੂਰਨ argc, ਅੱਖਰ *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    ਪੂਰਨ C = model.config.channels;
    ਪੂਰਨ V = model.config.vocab_ਮਾਪ;
    ਪੂਰਨ Vp = model.config.padded_vocab_ਮਾਪ;
    ਪੂਰਨ maxT = model.config.max_seq_len;
    ਪੂਰਨ L = model.config.ਅੰਕ_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    ਜੇ (state_file == ਨਿਰਾ) { ਲਿਖੋ("Error opening state file\n"); ਵਾਪਸ 1; }
    ਪੂਰਨ state_ਸਿਰਲੇਖ[256];
    freadCheck(state_ਸਿਰਲੇਖ, ਆਕਾਰ(ਪੂਰਨ), 256, state_file);
    ਜੇ (state_ਸਿਰਲੇਖ[0] != 20240327) { ਲਿਖੋ("Bad magic state file\n"); ਵਾਪਸ 1; }
    ਜੇ (state_ਸਿਰਲੇਖ[1] != 2) {
        ਲਿਖੋ("Bad version in state file\n");
        ਲਿਖੋ("---> HINT: try to re-run `python train_gpt2.py`\n");
        ਵਾਪਸ 1;
    }
    ਪੂਰਨ B = state_ਸਿਰਲੇਖ[2]; // batch size, e.g. 4
    ਪੂਰਨ T = state_ਸਿਰਲੇਖ[3]; // time / sequence length (e.g. 64, up to maxT)
    ਲਿਖੋ("[State]\n");
    ਲਿਖੋ("batch_size: %d\n", B);
    ਲਿਖੋ("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    ਦਸ਼ਮਲਵ* expected_grads_ਸਮ੍ਰਿਤੀ = malloc_and_ਬਿੰਦੂ_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    ਪੂਰਨ* x = (ਪੂਰਨ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * ਆਕਾਰ(ਪੂਰਨ));
    ਪੂਰਨ* y = (ਪੂਰਨ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * ਆਕਾਰ(ਪੂਰਨ));
    ਦਸ਼ਮਲਵ* expected_logits = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T * V * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਦਸ਼ਮਲਵ* expected_loss = (ਦਸ਼ਮਲਵ*) ਸਮ੍ਰਿਤੀ_ਦਿਓ(1 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // read reference information from Python
    freadCheck(x, ਆਕਾਰ(ਪੂਰਨ), B*T, state_file);
    freadCheck(y, ਆਕਾਰ(ਪੂਰਨ), B*T, state_file);
    freadCheck(expected_logits, ਆਕਾਰ(ਦਸ਼ਮਲਵ), B*T*V, state_file);
    freadCheck(expected_loss, ਆਕਾਰ(ਦਸ਼ਮਲਵ), 1, state_file);
    freadCheck(expected_grads_ਸਮ੍ਰਿਤੀ, ਆਕਾਰ(ਦਸ਼ਮਲਵ), model.ਅੰਕ_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    ਪੂਰਨ allok = 1;

    // let's do 10 training iterations, following the pytorch code
    ਦਸ਼ਮਲਵ expected_losses[10] = {
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
    ਲਈ (ਪੂਰਨ ਕਦਮ = 0; ਕਦਮ < 10; ਕਦਮ++) {

        ਬਣਤਰ timespec ਸ਼ੁਰੂ, ਸਿਰਾ;
        clock_gettime(CLOCK_MONOTONIC, &ਸ਼ੁਰੂ);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &ਸਿਰਾ);
        ਦੁੱਗਣਾ ਸਮਾਂ_elapsed_s = (ਸਿਰਾ.tv_sec - ਸ਼ੁਰੂ.tv_sec) + (ਸਿਰਾ.tv_nsec - ਸ਼ੁਰੂ.tv_nsec) / 1e9;

        ਜੇ (ਕਦਮ == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            ਪੂਰਨ logits_ok = 1;
            ਦਸ਼ਮਲਵ* calculated_logits = model.acts.logits;
            ਦਸ਼ਮਲਵ max_diff = 0.0f;
            ਲਈ (ਪੂਰਨ bt = 0; bt < B*T; bt++) {
                ਲਈ (ਪੂਰਨ v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    ਪੂਰਨ i = bt * Vp + v; // linearized index, using Vp
                    ਜੇ (i < 10) {
                        ਲਿਖੋ("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    ਦਸ਼ਮਲਵ diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    ਜੇ (diff >= 1e-2f) {
                        ਲਿਖੋ("MISMATCH AT INDEX %d,%d: ", bt, v);
                        ਲਿਖੋ("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        ਰੋਕੋ;
                    }
                }
            }
            ਜੇ(!logits_ok) { ਲਿਖੋ("NOT "); }
            ਲਿਖੋ("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            ਜੇ (fabsf(model.ਔਸਤ_loss - *expected_loss) >= 1e-2) {
                ਲਿਖੋ("LOSS MISMATCH: %f %f\n", model.ਔਸਤ_loss, *expected_loss);
                allok = 0;
            } ਨਹੀਂਤਾਂ {
                ਲਿਖੋ("LOSS OK: %f %f\n", model.ਔਸਤ_loss, *expected_loss);
            }

            // finally check all the gradients
            ਪੂਰਨ gradoks[16];
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
            ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, ਕਦਮ+1);

        // compare the losses
        ਦਸ਼ਮਲਵ expected_loss = expected_losses[ਕਦਮ];
        ਦਸ਼ਮਲਵ actual_loss = model.ਔਸਤ_loss;
        ਪੂਰਨ ਕਦਮ_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && ਕਦਮ_loss_ok;

        // print the timing information at the end
        ਲਿਖੋ("step %d: loss %f (took %f ms) OK = %d\n", ਕਦਮ, model.ਔਸਤ_loss, ਸਮਾਂ_elapsed_s * 1000, ਕਦਮ_loss_ok);
    }

    // final judgement
    ਲਿਖੋ("overall okay: %d\n", allok);

    // free everything
    ਮੁਕਤ(x);
    ਮੁਕਤ(y);
    ਮੁਕਤ(expected_logits);
    ਮੁਕਤ(expected_loss);
    ਮੁਕਤ(expected_grads_ਸਮ੍ਰਿਤੀ);
    gpt2_free(&model);
    ਵਾਪਸ 0;
}
