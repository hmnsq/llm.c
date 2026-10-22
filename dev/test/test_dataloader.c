/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ଅନ୍ତର୍ଭାଵ "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
ଅକ୍ଷର shard_name[SHARD_NAME_LEN];
ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_tokens = 140;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_shards = 4;

ଶୂନ୍ୟ check_range(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *tokens, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା start, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା end, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = end - start;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା token = tokens[i];
        ଯଦି (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

ଶୂନ୍ୟ check_equals(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *tokens, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା expected, ଅଚଳ ଅକ୍ଷର *file, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି) {
    // checks that the tokens[0, ... n] are all equal to expected
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା token = tokens[i];
        ଯଦି (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ଧାଡ଼ି);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

ଶୂନ୍ୟ test_simple(ଶୂନ୍ୟ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B = 4;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା T = 8;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା process_rank = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_processes = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ଅଙ୍କ_processes, should_shuffle);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches_fit = ଅଙ୍କ_tokens / (B * T); // number of batches that fit per shard
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା BT = B * T;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_epochs = 4;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା e = 0; e < ଅଙ୍କ_epochs; e++) { // epoch
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < ଅଙ୍କ_shards; s++) { // shard
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = s * ଅଙ୍କ_tokens;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                checkRange(loader.inputs, start, start + BT);
                checkRange(loader.targets, start + 1, start + BT + 1);
                start += BT;
            }
        }
    }
    dataloader_free(&loader);
    printf("OK\n");
}

ଶୂନ୍ୟ test_multiprocess_simple(ଶୂନ୍ୟ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B = 4;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା T = 8;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_processes = 2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, ଅଙ୍କ_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, ଅଙ୍କ_processes, should_shuffle);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches_fit = ଅଙ୍କ_tokens / (B * T * ଅଙ୍କ_processes); // number of batches that fit per shard
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା BT = B * T;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_epochs = 4;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା e = 0; e < ଅଙ୍କ_epochs; e++) { // epoch
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < ଅଙ୍କ_shards; s++) { // shard
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = s * ଅଙ୍କ_tokens;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader0);
                dataloader_next_batch(&loader1);
                checkRange(loader0.inputs, start, start + BT);
                checkRange(loader1.inputs, start + BT, start + 2*BT);
                checkRange(loader0.targets, start + 1, start + BT + 1);
                checkRange(loader1.targets, start + BT + 1, start + 2*BT + 1);
                start += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    printf("OK\n");
}

ଶୂନ୍ୟ test_shuffled(ଶୂନ୍ୟ) {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    printf("test_shuffled... ");
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B = 4;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା T = 8;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା process_rank = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_processes = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ଅଙ୍କ_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_tokens = ଅଙ୍କ_shards * ଅଙ୍କ_tokens;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା *ଅଙ୍କ_seen_inputs = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)calloc(total_tokens, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା *ଅଙ୍କ_seen_targets = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)calloc(total_tokens, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches_fit = ଅଙ୍କ_tokens / (B * T); // number of batches that fit per shard
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା BT = B * T;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_epochs = 4;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା e = 0; e < ଅଙ୍କ_epochs; e ++) { // epoch
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < ଅଙ୍କ_shards; s++) { // shard
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = s * ଅଙ୍କ_tokens;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < BT; i++) {
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_token = loader.inputs[i];
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା target_token = loader.targets[i];
                    ପ୍ରତିଜ୍ଞା(ନିବେଶ_token >= 0 && ନିବେଶ_token < total_tokens);
                    ପ୍ରତିଜ୍ଞା(target_token >= 0 && target_token < total_tokens);
                    ଅଙ୍କ_seen_inputs[ନିବେଶ_token]++;
                    ଅଙ୍କ_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < ଅଙ୍କ_shards; s++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = s * ଅଙ୍କ_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ଅଙ୍କ_seen_inputs + start, tokens_fit, ଅଙ୍କ_epochs);
        checkEquals(ଅଙ୍କ_seen_inputs + start + tokens_fit, ଅଙ୍କ_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ଅଙ୍କ_seen_targets + start + 1, tokens_fit, ଅଙ୍କ_epochs);
        checkEquals(ଅଙ୍କ_seen_targets + start + 1 + tokens_fit,
            (s == (ଅଙ୍କ_shards - 1)) ? ଅଙ୍କ_tokens - tokens_fit - 1 : ଅଙ୍କ_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    ମୁକ୍ତ(ଅଙ୍କ_seen_inputs);
    ମୁକ୍ତ(ଅଙ୍କ_seen_targets);
    printf("OK\n");
}

ଶୂନ୍ୟ test_multiprocess_shuffled(ଶୂନ୍ୟ) {
    /*
    Tests the DataLoader when using both multiprocess and shuffled:
    - multi-shard
    - multi-process
    - shuffled!
    DataLoaders should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times, over multiple epochs.
    */

    printf("test_multiprocess_shuffled... ");
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B = 4;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା T = 8;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_processes = 2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[ଅଙ୍କ_processes];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, ଅଙ୍କ_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_tokens = ଅଙ୍କ_shards * ଅଙ୍କ_tokens;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା *ଅଙ୍କ_seen_inputs = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)calloc(total_tokens, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା *ଅଙ୍କ_seen_targets = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)calloc(total_tokens, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches_fit = ଅଙ୍କ_tokens / (B * T * ଅଙ୍କ_processes); // number of batches that fit per shard
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା BT = B * T;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_epochs = 4;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା e = 0; e < ଅଙ୍କ_epochs; e ++) { // epoch
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < ଅଙ୍କ_shards; s++) { // shard
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = s * ଅଙ୍କ_tokens;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < batches_fit; b++) { // batch
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 0; n < ଅଙ୍କ_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < BT; i++) {
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_token = loader->inputs[i];
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା target_token = loader->targets[i];
                        ପ୍ରତିଜ୍ଞା(ନିବେଶ_token >= 0 && ନିବେଶ_token < total_tokens);
                        ପ୍ରତିଜ୍ଞା(target_token >= 0 && target_token < total_tokens);
                        ଅଙ୍କ_seen_inputs[ନିବେଶ_token]++;
                        ଅଙ୍କ_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା tokens_fit = batches_fit * (B * T * ଅଙ୍କ_processes); // number of tokens that fit per shard
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 0; s < ଅଙ୍କ_shards; s++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା start = s * ଅଙ୍କ_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ଅଙ୍କ_seen_inputs + start, tokens_fit, ଅଙ୍କ_epochs);
        checkEquals(ଅଙ୍କ_seen_inputs + start + tokens_fit, ଅଙ୍କ_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ଅଙ୍କ_seen_targets + start + 1, tokens_fit, ଅଙ୍କ_epochs);
        checkEquals(ଅଙ୍କ_seen_targets + start + 1 + tokens_fit,
            (s == (ଅଙ୍କ_shards - 1)) ? ଅଙ୍କ_tokens - tokens_fit - 1 : ଅଙ୍କ_tokens - tokens_fit,0);
    }

    // cleanup
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    ମୁକ୍ତ(ଅଙ୍କ_seen_inputs);
    ମୁକ୍ତ(ଅଙ୍କ_seen_targets);
    printf("OK\n");
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୁଖ୍ୟ(ଶୂନ୍ୟ) {

    // generate a few dummy shards of data with incrementing tokens
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା header[HEADER_SIZE];
    uint16_t tokens[ଅଙ୍କ_tokens];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା shard_id = 0; shard_id < ଅଙ୍କ_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା token_offset = shard_id * ଅଙ୍କ_tokens;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = ଅଙ୍କ_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା), HEADER_SIZE, shard_file);
        fwrite(tokens, ଆକାର(uint16_t), ଅଙ୍କ_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା shard_id = 0; shard_id < ଅଙ୍କ_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    ଫେରନ୍ତୁ EXIT_SUCCESS;
}