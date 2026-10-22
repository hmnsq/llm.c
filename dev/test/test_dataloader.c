/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
ವರ್ಣ shard_name[SHARD_NAME_LEN];
ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_tokens = 140;
ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_shards = 4;

ಶೂನ್ಯ check_range(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *tokens, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ start, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ end, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    ಪೂರ್ಣಾಙ್ಕ n = end - start;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        ಪೂರ್ಣಾಙ್ಕ token = tokens[i];
        ಯದಿ (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ಸಾಲು);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

ಶೂನ್ಯ check_equals(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *tokens, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ expected, ಧ್ರುವ ವರ್ಣ *file, ಪೂರ್ಣಾಙ್ಕ ಸಾಲು) {
    // checks that the tokens[0, ... n] are all equal to expected
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        ಪೂರ್ಣಾಙ್ಕ token = tokens[i];
        ಯದಿ (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ಸಾಲು);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

ಶೂನ್ಯ test_simple(ಶೂನ್ಯ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    ಪೂರ್ಣಾಙ್ಕ B = 4;
    ಪೂರ್ಣಾಙ್ಕ T = 8;
    ಪೂರ್ಣಾಙ್ಕ process_rank = 0;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes = 1;
    ಪೂರ್ಣಾಙ್ಕ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ಅಂಕೆ_processes, should_shuffle);

    ಪೂರ್ಣಾಙ್ಕ batches_fit = ಅಂಕೆ_tokens / (B * T); // number of batches that fit per shard
    ಪೂರ್ಣಾಙ್ಕ BT = B * T;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_epochs = 4;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ e = 0; e < ಅಂಕೆ_epochs; e++) { // epoch
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಙ್ಕ start = s * ಅಂಕೆ_tokens;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < batches_fit; b++) { // batch
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

ಶೂನ್ಯ test_multiprocess_simple(ಶೂನ್ಯ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    ಪೂರ್ಣಾಙ್ಕ B = 4;
    ಪೂರ್ಣಾಙ್ಕ T = 8;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes = 2;
    ಪೂರ್ಣಾಙ್ಕ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, ಅಂಕೆ_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, ಅಂಕೆ_processes, should_shuffle);

    ಪೂರ್ಣಾಙ್ಕ batches_fit = ಅಂಕೆ_tokens / (B * T * ಅಂಕೆ_processes); // number of batches that fit per shard
    ಪೂರ್ಣಾಙ್ಕ BT = B * T;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_epochs = 4;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ e = 0; e < ಅಂಕೆ_epochs; e++) { // epoch
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಙ್ಕ start = s * ಅಂಕೆ_tokens;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < batches_fit; b++) { // batch
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

ಶೂನ್ಯ test_shuffled(ಶೂನ್ಯ) {
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
    ಪೂರ್ಣಾಙ್ಕ B = 4;
    ಪೂರ್ಣಾಙ್ಕ T = 8;
    ಪೂರ್ಣಾಙ್ಕ process_rank = 0;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes = 1;
    ಪೂರ್ಣಾಙ್ಕ should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ಅಂಕೆ_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    ಪೂರ್ಣಾಙ್ಕ total_tokens = ಅಂಕೆ_shards * ಅಂಕೆ_tokens;
    ಪೂರ್ಣಾಙ್ಕ *ಅಂಕೆ_seen_inputs = (ಪೂರ್ಣಾಙ್ಕ *)calloc(total_tokens, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪೂರ್ಣಾಙ್ಕ *ಅಂಕೆ_seen_targets = (ಪೂರ್ಣಾಙ್ಕ *)calloc(total_tokens, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪೂರ್ಣಾಙ್ಕ batches_fit = ಅಂಕೆ_tokens / (B * T); // number of batches that fit per shard
    ಪೂರ್ಣಾಙ್ಕ BT = B * T;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_epochs = 4;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ e = 0; e < ಅಂಕೆ_epochs; e ++) { // epoch
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಙ್ಕ start = s * ಅಂಕೆ_tokens;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < BT; i++) {
                    ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_token = loader.inputs[i];
                    ಪೂರ್ಣಾಙ್ಕ target_token = loader.targets[i];
                    ಪ್ರತಿಜ್ಞಾ(ಒಳಹರಿವು_token >= 0 && ಒಳಹರಿವು_token < total_tokens);
                    ಪ್ರತಿಜ್ಞಾ(target_token >= 0 && target_token < total_tokens);
                    ಅಂಕೆ_seen_inputs[ಒಳಹರಿವು_token]++;
                    ಅಂಕೆ_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ಪೂರ್ಣಾಙ್ಕ tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < ಅಂಕೆ_shards; s++) {
        ಪೂರ್ಣಾಙ್ಕ start = s * ಅಂಕೆ_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ಅಂಕೆ_seen_inputs + start, tokens_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_seen_inputs + start + tokens_fit, ಅಂಕೆ_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ಅಂಕೆ_seen_targets + start + 1, tokens_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_seen_targets + start + 1 + tokens_fit,
            (s == (ಅಂಕೆ_shards - 1)) ? ಅಂಕೆ_tokens - tokens_fit - 1 : ಅಂಕೆ_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    ಮುಕ್ತ(ಅಂಕೆ_seen_inputs);
    ಮುಕ್ತ(ಅಂಕೆ_seen_targets);
    printf("OK\n");
}

ಶೂನ್ಯ test_multiprocess_shuffled(ಶೂನ್ಯ) {
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
    ಪೂರ್ಣಾಙ್ಕ B = 4;
    ಪೂರ್ಣಾಙ್ಕ T = 8;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes = 2;
    ಪೂರ್ಣಾಙ್ಕ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[ಅಂಕೆ_processes];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, ಅಂಕೆ_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    ಪೂರ್ಣಾಙ್ಕ total_tokens = ಅಂಕೆ_shards * ಅಂಕೆ_tokens;
    ಪೂರ್ಣಾಙ್ಕ *ಅಂಕೆ_seen_inputs = (ಪೂರ್ಣಾಙ್ಕ *)calloc(total_tokens, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪೂರ್ಣಾಙ್ಕ *ಅಂಕೆ_seen_targets = (ಪೂರ್ಣಾಙ್ಕ *)calloc(total_tokens, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    ಪೂರ್ಣಾಙ್ಕ batches_fit = ಅಂಕೆ_tokens / (B * T * ಅಂಕೆ_processes); // number of batches that fit per shard
    ಪೂರ್ಣಾಙ್ಕ BT = B * T;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_epochs = 4;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ e = 0; e < ಅಂಕೆ_epochs; e ++) { // epoch
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಙ್ಕ start = s * ಅಂಕೆ_tokens;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < batches_fit; b++) { // batch
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ n = 0; n < ಅಂಕೆ_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < BT; i++) {
                        ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_token = loader->inputs[i];
                        ಪೂರ್ಣಾಙ್ಕ target_token = loader->targets[i];
                        ಪ್ರತಿಜ್ಞಾ(ಒಳಹರಿವು_token >= 0 && ಒಳಹರಿವು_token < total_tokens);
                        ಪ್ರತಿಜ್ಞಾ(target_token >= 0 && target_token < total_tokens);
                        ಅಂಕೆ_seen_inputs[ಒಳಹರಿವು_token]++;
                        ಅಂಕೆ_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ಪೂರ್ಣಾಙ್ಕ tokens_fit = batches_fit * (B * T * ಅಂಕೆ_processes); // number of tokens that fit per shard
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 0; s < ಅಂಕೆ_shards; s++) {
        ಪೂರ್ಣಾಙ್ಕ start = s * ಅಂಕೆ_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ಅಂಕೆ_seen_inputs + start, tokens_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_seen_inputs + start + tokens_fit, ಅಂಕೆ_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ಅಂಕೆ_seen_targets + start + 1, tokens_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_seen_targets + start + 1 + tokens_fit,
            (s == (ಅಂಕೆ_shards - 1)) ? ಅಂಕೆ_tokens - tokens_fit - 1 : ಅಂಕೆ_tokens - tokens_fit,0);
    }

    // cleanup
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    ಮುಕ್ತ(ಅಂಕೆ_seen_inputs);
    ಮುಕ್ತ(ಅಂಕೆ_seen_targets);
    printf("OK\n");
}

ಪೂರ್ಣಾಙ್ಕ main(ಶೂನ್ಯ) {

    // generate a few dummy shards of data with incrementing tokens
    ಪೂರ್ಣಾಙ್ಕ header[HEADER_SIZE];
    uint16_t tokens[ಅಂಕೆ_tokens];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ shard_id = 0; shard_id < ಅಂಕೆ_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        ಪೂರ್ಣಾಙ್ಕ token_offset = shard_id * ಅಂಕೆ_tokens;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = ಅಂಕೆ_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), HEADER_SIZE, shard_file);
        fwrite(tokens, ಪರಿಮಾಣ(uint16_t), ಅಂಕೆ_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ shard_id = 0; shard_id < ಅಂಕೆ_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    ಪ್ರತ್ಯಾಗಮ EXIT_SUCCESS;
}