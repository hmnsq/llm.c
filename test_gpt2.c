#define TESTING
#অন্তর্ভাৱ "train_gpt2.c"

// poor man's tensor checker
পূর্ণাঙ্ক check_tensor(ভিন্ন *a, ভিন্ন *b, পূর্ণাঙ্ক n, ধ্রুৱ ৱর্ণ* label) {
    পূর্ণাঙ্ক print_upto = 5;
    পূর্ণাঙ্ক ok = 1;
    ভিন্ন maxdiff = 0.0f;
    ভিন্ন tol = 2e-2f;
    printf("%s\n", label);
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        ভিন্ন diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        যদি (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        যদি (i < print_upto) {
            যদি (diff <= tol) {
                যদি (i < print_upto) { printf("OK "); }
            } অন্যথা {
                যদি (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    যদি (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } অন্যথা {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    ঘূৰাই ok;
}

পূর্ণাঙ্ক main(পূর্ণাঙ্ক argc, ৱর্ণ *argv[]) {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    পূর্ণাঙ্ক C = model.config.channels;
    পূর্ণাঙ্ক V = model.config.vocab_size;
    পূর্ণাঙ্ক Vp = model.config.padded_vocab_size;
    পূর্ণাঙ্ক maxT = model.config.max_seq_len;
    পূর্ণাঙ্ক L = model.config.অঙ্ক_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    যদি (state_file == রিক্ত) { printf("Error opening state file\n"); ঘূৰাই 1; }
    পূর্ণাঙ্ক state_header[256];
    freadCheck(state_header, পরিমাণ(পূর্ণাঙ্ক), 256, state_file);
    যদি (state_header[0] != 20240327) { printf("Bad magic state file\n"); ঘূৰাই 1; }
    যদি (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        ঘূৰাই 1;
    }
    পূর্ণাঙ্ক B = state_header[2]; // batch size, e.g. 4
    পূর্ণাঙ্ক T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    ParameterTensors expected_grads;
    ভিন্ন* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // inputs and expected outputs, only used for error checking
    পূর্ণাঙ্ক* x = (পূর্ণাঙ্ক*) malloc(B * T * পরিমাণ(পূর্ণাঙ্ক));
    পূর্ণাঙ্ক* y = (পূর্ণাঙ্ক*) malloc(B * T * পরিমাণ(পূর্ণাঙ্ক));
    ভিন্ন* expected_logits = (ভিন্ন*) malloc(B * T * V * পরিমাণ(ভিন্ন));
    ভিন্ন* expected_loss = (ভিন্ন*) malloc(1 * পরিমাণ(ভিন্ন));

    // read reference information from Python
    freadCheck(x, পরিমাণ(পূর্ণাঙ্ক), B*T, state_file);
    freadCheck(y, পরিমাণ(পূর্ণাঙ্ক), B*T, state_file);
    freadCheck(expected_logits, পরিমাণ(ভিন্ন), B*T*V, state_file);
    freadCheck(expected_loss, পরিমাণ(ভিন্ন), 1, state_file);
    freadCheck(expected_grads_memory, পরিমাণ(ভিন্ন), model.অঙ্ক_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    পূর্ণাঙ্ক allok = 1;

    // let's do 10 training iterations, following the pytorch code
    ভিন্ন expected_losses[10] = {
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
    বাবে (পূর্ণাঙ্ক step = 0; step < 10; step++) {

        সংরচনা timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        দ্ৱিগুণ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        যদি (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            পূর্ণাঙ্ক logits_ok = 1;
            ভিন্ন* calculated_logits = model.acts.logits;
            ভিন্ন max_diff = 0.0f;
            বাবে (পূর্ণাঙ্ক bt = 0; bt < B*T; bt++) {
                বাবে (পূর্ণাঙ্ক v = 0; v < V; v++) { // note we only loop to V (ignoring padding)
                    পূর্ণাঙ্ক i = bt * Vp + v; // linearized index, using Vp
                    যদি (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }
                    ভিন্ন diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    যদি (diff >= 1e-2f) {
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // to break out of both loops
                        ভঙ্গ;
                    }
                }
            }
            যদি(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // compare the achieved loss
            যদি (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } অন্যথা {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // finally check all the gradients
            পূর্ণাঙ্ক gradoks[16];
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
            বাবে (পূর্ণাঙ্ক i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // compare the losses
        ভিন্ন expected_loss = expected_losses[step];
        ভিন্ন actual_loss = model.mean_loss;
        পূর্ণাঙ্ক step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;
        allok = allok && step_loss_ok;

        // print the timing information at the end
        printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // final judgement
    printf("overall okay: %d\n", allok);

    // free everything
    মুক্ত(x);
    মুক্ত(y);
    মুক্ত(expected_logits);
    মুক্ত(expected_loss);
    মুক্ত(expected_grads_memory);
    gpt2_free(&model);
    ঘূৰাই 0;
}
