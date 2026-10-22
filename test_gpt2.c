#నిర్వచించు TESTING
#అన్తర్భావ "train_gpt2.c"

// poor man's tensor checker
పూర్ణాంకం check_tensor(దశాంశం *a, దశాంశం *b, పూర్ణాంకం n, స్థిరరాశి అక్షరం* label) {
    పూర్ణాంకం print_upto = 5;
    పూర్ణాంకం ok = 1;
    దశాంశం maxdiff = 0.0f;
    దశాంశం tol = 2e-2f;
    రాయి("%s\n", label);
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        దశాంశం diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        ఒకవేళ (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        ఒకవేళ (i < print_upto) {
            ఒకవేళ (diff <= tol) {
                ఒకవేళ (i < print_upto) { రాయి("OK "); }
            } లేకపోతే {
                ఒకవేళ (i < print_upto) { రాయి("NOT OK "); }
            }
            రాయి("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    ఒకవేళ (ok) {
        రాయి("TENSOR OK, maxdiff = %e\n", maxdiff);
    } లేకపోతే {
        రాయి("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    తిరిగి ok;
}

పూర్ణాంకం ప్రధాన(పూర్ణాంకం argc, అక్షరం *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    పూర్ణాంకం C = model.config.channels;
    పూర్ణాంకం V = model.config.vocab_కొలత;
    పూర్ణాంకం Vp = model.config.padded_vocab_కొలత;
    పూర్ణాంకం maxT = model.config.max_seq_len;
    పూర్ణాంకం L = model.config.అంకె_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    ఒకవేళ (state_file == ఖాళీ) { రాయి("Error opening state file\n"); తిరిగి 1; }
    పూర్ణాంకం state_శీర్షిక[256];
    freadCheck(state_శీర్షిక, పరిమాణం(పూర్ణాంకం), 256, state_file);
    ఒకవేళ (state_శీర్షిక[0] != 20240327) { రాయి("Bad magic state file\n"); తిరిగి 1; }
    ఒకవేళ (state_శీర్షిక[1] != 2) {
        రాయి("Bad version in state file\n");
        రాయి("---> HINT: try to re-run `python train_gpt2.py`\n");
        తిరిగి 1;
    }
    పూర్ణాంకం B = state_శీర్షిక[2]; // batch size, e.g. 4
    పూర్ణాంకం T = state_శీర్షిక[3]; // time / sequence length (e.g. 64, up to maxT)
    రాయి("[State]\n");
    రాయి("batch_size: %d\n", B);
    రాయి("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    దశాంశం* expected_grads_జ్ఞాపకం = malloc_and_బిందువు_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    పూర్ణాంకం* x = (పూర్ణాంకం*) జ్ఞాపకం_ఇవ్వు(B * T * పరిమాణం(పూర్ణాంకం));
    పూర్ణాంకం* y = (పూర్ణాంకం*) జ్ఞాపకం_ఇవ్వు(B * T * పరిమాణం(పూర్ణాంకం));
    దశాంశం* expected_logits = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(B * T * V * పరిమాణం(దశాంశం));
    దశాంశం* expected_loss = (దశాంశం*) జ్ఞాపకం_ఇవ్వు(1 * పరిమాణం(దశాంశం));

    // read reference information from Python
    freadCheck(x, పరిమాణం(పూర్ణాంకం), B*T, state_file);
    freadCheck(y, పరిమాణం(పూర్ణాంకం), B*T, state_file);
    freadCheck(expected_logits, పరిమాణం(దశాంశం), B*T*V, state_file);
    freadCheck(expected_loss, పరిమాణం(దశాంశం), 1, state_file);
    freadCheck(expected_grads_జ్ఞాపకం, పరిమాణం(దశాంశం), model.అంకె_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    పూర్ణాంకం allok = 1;

    // let's do 10 training iterations, following the pytorch code
    దశాంశం expected_losses[10] = {
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
    కొరకు (పూర్ణాంకం అడుగు = 0; అడుగు < 10; అడుగు++) {

        నిర్మాణం timespec ప్రారంభం, చివర;
        clock_gettime(CLOCK_MONOTONIC, &ప్రారంభం);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &చివర);
        ద్విగుణ సమయం_elapsed_s = (చివర.tv_sec - ప్రారంభం.tv_sec) + (చివర.tv_nsec - ప్రారంభం.tv_nsec) / 1e9;

        ఒకవేళ (అడుగు == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            పూర్ణాంకం logits_ok = 1;
            దశాంశం* calculated_logits = model.acts.logits;
            దశాంశం max_diff = 0.0f;
            కొరకు (పూర్ణాంకం bt = 0; bt < B*T; bt++) {
                కొరకు (పూర్ణాంకం v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    పూర్ణాంకం i = bt * Vp + v; // linearized index, using Vp
                    ఒకవేళ (i < 10) {
                        రాయి("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    దశాంశం diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    ఒకవేళ (diff >= 1e-2f) {
                        రాయి("MISMATCH AT INDEX %d,%d: ", bt, v);
                        రాయి("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        ఆపు;
                    }
                }
            }
            ఒకవేళ(!logits_ok) { రాయి("NOT "); }
            రాయి("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            ఒకవేళ (fabsf(model.సగటు_loss - *expected_loss) >= 1e-2) {
                రాయి("LOSS MISMATCH: %f %f\n", model.సగటు_loss, *expected_loss);
                allok = 0;
            } లేకపోతే {
                రాయి("LOSS OK: %f %f\n", model.సగటు_loss, *expected_loss);
            }

            // finally check all the gradients
            పూర్ణాంకం gradoks[16];
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
            కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, అడుగు+1);

        // compare the losses
        దశాంశం expected_loss = expected_losses[అడుగు];
        దశాంశం actual_loss = model.సగటు_loss;
        పూర్ణాంకం అడుగు_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && అడుగు_loss_ok;

        // print the timing information at the end
        రాయి("step %d: loss %f (took %f ms) OK = %d\n", అడుగు, model.సగటు_loss, సమయం_elapsed_s * 1000, అడుగు_loss_ok);
    }

    // final judgement
    రాయి("overall okay: %d\n", allok);

    // free everything
    విడుదల(x);
    విడుదల(y);
    విడుదల(expected_logits);
    విడుదల(expected_loss);
    విడుదల(expected_grads_జ్ఞాపకం);
    gpt2_free(&model);
    తిరిగి 0;
}
