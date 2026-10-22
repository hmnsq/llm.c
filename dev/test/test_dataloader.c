/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#അന്തര്ഭാവ <unistd.h>
#അന്തര്ഭാവ "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
അക്ഷരം shard_name[SHARD_NAME_LEN];
സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_tokens = 140;
പൂർണ്ണസംഖ്യ അക്കം_shards = 4;

ശൂന്യം check_range(സ്ഥിരം പൂർണ്ണസംഖ്യ *tokens, സ്ഥിരം പൂർണ്ണസംഖ്യ start, സ്ഥിരം പൂർണ്ണസംഖ്യ end, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    പൂർണ്ണസംഖ്യ n = end - start;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        പൂർണ്ണസംഖ്യ token = tokens[i];
        എങ്കിൽ (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", വരി);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

ശൂന്യം check_equals(സ്ഥിരം പൂർണ്ണസംഖ്യ *tokens, സ്ഥിരം പൂർണ്ണസംഖ്യ n, സ്ഥിരം പൂർണ്ണസംഖ്യ expected, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    // checks that the tokens[0, ... n] are all equal to expected
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        പൂർണ്ണസംഖ്യ token = tokens[i];
        എങ്കിൽ (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", വരി);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

ശൂന്യം test_simple(ശൂന്യം) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    പൂർണ്ണസംഖ്യ B = 4;
    പൂർണ്ണസംഖ്യ T = 8;
    പൂർണ്ണസംഖ്യ process_rank = 0;
    പൂർണ്ണസംഖ്യ അക്കം_processes = 1;
    പൂർണ്ണസംഖ്യ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, അക്കം_processes, should_shuffle);

    പൂർണ്ണസംഖ്യ batches_fit = അക്കം_tokens / (B * T); // number of batches that fit per shard
    പൂർണ്ണസംഖ്യ BT = B * T;
    പൂർണ്ണസംഖ്യ അക്കം_epochs = 4;
    വേണ്ടി (പൂർണ്ണസംഖ്യ e = 0; e < അക്കം_epochs; e++) { // epoch
        വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < അക്കം_shards; s++) { // shard
            പൂർണ്ണസംഖ്യ start = s * അക്കം_tokens;
            വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < batches_fit; b++) { // batch
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

ശൂന്യം test_multiprocess_simple(ശൂന്യം) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    പൂർണ്ണസംഖ്യ B = 4;
    പൂർണ്ണസംഖ്യ T = 8;
    പൂർണ്ണസംഖ്യ അക്കം_processes = 2;
    പൂർണ്ണസംഖ്യ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, അക്കം_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, അക്കം_processes, should_shuffle);

    പൂർണ്ണസംഖ്യ batches_fit = അക്കം_tokens / (B * T * അക്കം_processes); // number of batches that fit per shard
    പൂർണ്ണസംഖ്യ BT = B * T;
    പൂർണ്ണസംഖ്യ അക്കം_epochs = 4;
    വേണ്ടി (പൂർണ്ണസംഖ്യ e = 0; e < അക്കം_epochs; e++) { // epoch
        വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < അക്കം_shards; s++) { // shard
            പൂർണ്ണസംഖ്യ start = s * അക്കം_tokens;
            വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < batches_fit; b++) { // batch
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

ശൂന്യം test_shuffled(ശൂന്യം) {
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
    പൂർണ്ണസംഖ്യ B = 4;
    പൂർണ്ണസംഖ്യ T = 8;
    പൂർണ്ണസംഖ്യ process_rank = 0;
    പൂർണ്ണസംഖ്യ അക്കം_processes = 1;
    പൂർണ്ണസംഖ്യ should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, അക്കം_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    പൂർണ്ണസംഖ്യ total_tokens = അക്കം_shards * അക്കം_tokens;
    പൂർണ്ണസംഖ്യ *അക്കം_seen_inputs = (പൂർണ്ണസംഖ്യ *)calloc(total_tokens, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    പൂർണ്ണസംഖ്യ *അക്കം_seen_targets = (പൂർണ്ണസംഖ്യ *)calloc(total_tokens, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    പൂർണ്ണസംഖ്യ batches_fit = അക്കം_tokens / (B * T); // number of batches that fit per shard
    പൂർണ്ണസംഖ്യ BT = B * T;
    പൂർണ്ണസംഖ്യ അക്കം_epochs = 4;
    വേണ്ടി (പൂർണ്ണസംഖ്യ e = 0; e < അക്കം_epochs; e ++) { // epoch
        വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < അക്കം_shards; s++) { // shard
            പൂർണ്ണസംഖ്യ start = s * അക്കം_tokens;
            വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < BT; i++) {
                    പൂർണ്ണസംഖ്യ നിവേശം_token = loader.inputs[i];
                    പൂർണ്ണസംഖ്യ target_token = loader.targets[i];
                    പ്രതിജ്ഞാ(നിവേശം_token >= 0 && നിവേശം_token < total_tokens);
                    പ്രതിജ്ഞാ(target_token >= 0 && target_token < total_tokens);
                    അക്കം_seen_inputs[നിവേശം_token]++;
                    അക്കം_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    പൂർണ്ണസംഖ്യ tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < അക്കം_shards; s++) {
        പൂർണ്ണസംഖ്യ start = s * അക്കം_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(അക്കം_seen_inputs + start, tokens_fit, അക്കം_epochs);
        checkEquals(അക്കം_seen_inputs + start + tokens_fit, അക്കം_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(അക്കം_seen_targets + start + 1, tokens_fit, അക്കം_epochs);
        checkEquals(അക്കം_seen_targets + start + 1 + tokens_fit,
            (s == (അക്കം_shards - 1)) ? അക്കം_tokens - tokens_fit - 1 : അക്കം_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    സ്വതന്ത്രം(അക്കം_seen_inputs);
    സ്വതന്ത്രം(അക്കം_seen_targets);
    printf("OK\n");
}

ശൂന്യം test_multiprocess_shuffled(ശൂന്യം) {
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
    പൂർണ്ണസംഖ്യ B = 4;
    പൂർണ്ണസംഖ്യ T = 8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_processes = 2;
    പൂർണ്ണസംഖ്യ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[അക്കം_processes];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, അക്കം_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    പൂർണ്ണസംഖ്യ total_tokens = അക്കം_shards * അക്കം_tokens;
    പൂർണ്ണസംഖ്യ *അക്കം_seen_inputs = (പൂർണ്ണസംഖ്യ *)calloc(total_tokens, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    പൂർണ്ണസംഖ്യ *അക്കം_seen_targets = (പൂർണ്ണസംഖ്യ *)calloc(total_tokens, വലുപ്പം(പൂർണ്ണസംഖ്യ));
    പൂർണ്ണസംഖ്യ batches_fit = അക്കം_tokens / (B * T * അക്കം_processes); // number of batches that fit per shard
    പൂർണ്ണസംഖ്യ BT = B * T;
    പൂർണ്ണസംഖ്യ അക്കം_epochs = 4;
    വേണ്ടി (പൂർണ്ണസംഖ്യ e = 0; e < അക്കം_epochs; e ++) { // epoch
        വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < അക്കം_shards; s++) { // shard
            പൂർണ്ണസംഖ്യ start = s * അക്കം_tokens;
            വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < batches_fit; b++) { // batch
                വേണ്ടി (പൂർണ്ണസംഖ്യ n = 0; n < അക്കം_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < BT; i++) {
                        പൂർണ്ണസംഖ്യ നിവേശം_token = loader->inputs[i];
                        പൂർണ്ണസംഖ്യ target_token = loader->targets[i];
                        പ്രതിജ്ഞാ(നിവേശം_token >= 0 && നിവേശം_token < total_tokens);
                        പ്രതിജ്ഞാ(target_token >= 0 && target_token < total_tokens);
                        അക്കം_seen_inputs[നിവേശം_token]++;
                        അക്കം_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    പൂർണ്ണസംഖ്യ tokens_fit = batches_fit * (B * T * അക്കം_processes); // number of tokens that fit per shard
    വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < അക്കം_shards; s++) {
        പൂർണ്ണസംഖ്യ start = s * അക്കം_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(അക്കം_seen_inputs + start, tokens_fit, അക്കം_epochs);
        checkEquals(അക്കം_seen_inputs + start + tokens_fit, അക്കം_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(അക്കം_seen_targets + start + 1, tokens_fit, അക്കം_epochs);
        checkEquals(അക്കം_seen_targets + start + 1 + tokens_fit,
            (s == (അക്കം_shards - 1)) ? അക്കം_tokens - tokens_fit - 1 : അക്കം_tokens - tokens_fit,0);
    }

    // cleanup
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    സ്വതന്ത്രം(അക്കം_seen_inputs);
    സ്വതന്ത്രം(അക്കം_seen_targets);
    printf("OK\n");
}

പൂർണ്ണസംഖ്യ പ്രധാന(ശൂന്യം) {

    // generate a few dummy shards of data with incrementing tokens
    പൂർണ്ണസംഖ്യ header[HEADER_SIZE];
    uint16_t tokens[അക്കം_tokens];
    വേണ്ടി (പൂർണ്ണസംഖ്യ shard_id = 0; shard_id < അക്കം_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        പൂർണ്ണസംഖ്യ token_offset = shard_id * അക്കം_tokens;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = അക്കം_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, വലുപ്പം(പൂർണ്ണസംഖ്യ), HEADER_SIZE, shard_file);
        fwrite(tokens, വലുപ്പം(uint16_t), അക്കം_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    വേണ്ടി (പൂർണ്ണസംഖ്യ shard_id = 0; shard_id < അക്കം_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    തിരികെ EXIT_SUCCESS;
}