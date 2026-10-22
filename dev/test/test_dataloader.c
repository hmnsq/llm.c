/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ضمن <unistd.h>
#ضمن "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
حرف shard_name[SHARD_NAME_LEN];
ثابت صحيح عدد_tokens = 140;
صحيح عدد_shards = 4;

فراغ check_range(ثابت صحيح *tokens, ثابت صحيح start, ثابت صحيح end, ثابت حرف *file, صحيح سطر) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحيح n = end - start;
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = tokens[i];
        إذا (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

فراغ check_equals(ثابت صحيح *tokens, ثابت صحيح n, ثابت صحيح expected, ثابت حرف *file, صحيح سطر) {
    // checks that the tokens[0, ... n] are all equal to expected
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = tokens[i];
        إذا (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

فراغ test_simple(فراغ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح عدد_processes = 1;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    صحيح batches_fit = عدد_tokens / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح start = s * عدد_tokens;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
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

فراغ test_multiprocess_simple(فراغ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح عدد_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, عدد_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, عدد_processes, should_shuffle);

    صحيح batches_fit = عدد_tokens / (B * T * عدد_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح start = s * عدد_tokens;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
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

فراغ test_shuffled(فراغ) {
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
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح عدد_processes = 1;
    صحيح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح total_tokens = عدد_shards * عدد_tokens;
    صحيح *عدد_seen_inputs = (صحيح *)calloc(total_tokens, حجم(صحيح));
    صحيح *عدد_seen_targets = (صحيح *)calloc(total_tokens, حجم(صحيح));
    صحيح batches_fit = عدد_tokens / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح start = s * عدد_tokens;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                لكل (صحيح i = 0; i < BT; i++) {
                    صحيح داخلہ_token = loader.inputs[i];
                    صحيح target_token = loader.targets[i];
                    أكد(داخلہ_token >= 0 && داخلہ_token < total_tokens);
                    أكد(target_token >= 0 && target_token < total_tokens);
                    عدد_seen_inputs[داخلہ_token]++;
                    عدد_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    لكل (صحيح s = 0; s < عدد_shards; s++) {
        صحيح start = s * عدد_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_seen_inputs + start, tokens_fit, عدد_epochs);
        checkEquals(عدد_seen_inputs + start + tokens_fit, عدد_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_seen_targets + start + 1, tokens_fit, عدد_epochs);
        checkEquals(عدد_seen_targets + start + 1 + tokens_fit,
            (s == (عدد_shards - 1)) ? عدد_tokens - tokens_fit - 1 : عدد_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    حرر(عدد_seen_inputs);
    حرر(عدد_seen_targets);
    printf("OK\n");
}

فراغ test_multiprocess_shuffled(فراغ) {
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
    صحيح B = 4;
    صحيح T = 8;
    ثابت صحيح عدد_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[عدد_processes];
    لكل (صحيح i = 0; i < عدد_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, عدد_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح total_tokens = عدد_shards * عدد_tokens;
    صحيح *عدد_seen_inputs = (صحيح *)calloc(total_tokens, حجم(صحيح));
    صحيح *عدد_seen_targets = (صحيح *)calloc(total_tokens, حجم(صحيح));
    صحيح batches_fit = عدد_tokens / (B * T * عدد_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح start = s * عدد_tokens;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                لكل (صحيح n = 0; n < عدد_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    لكل (صحيح i = 0; i < BT; i++) {
                        صحيح داخلہ_token = loader->inputs[i];
                        صحيح target_token = loader->targets[i];
                        أكد(داخلہ_token >= 0 && داخلہ_token < total_tokens);
                        أكد(target_token >= 0 && target_token < total_tokens);
                        عدد_seen_inputs[داخلہ_token]++;
                        عدد_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح tokens_fit = batches_fit * (B * T * عدد_processes); // number of tokens that fit per shard
    لكل (صحيح s = 0; s < عدد_shards; s++) {
        صحيح start = s * عدد_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_seen_inputs + start, tokens_fit, عدد_epochs);
        checkEquals(عدد_seen_inputs + start + tokens_fit, عدد_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_seen_targets + start + 1, tokens_fit, عدد_epochs);
        checkEquals(عدد_seen_targets + start + 1 + tokens_fit,
            (s == (عدد_shards - 1)) ? عدد_tokens - tokens_fit - 1 : عدد_tokens - tokens_fit,0);
    }

    // cleanup
    لكل (صحيح i = 0; i < عدد_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    حرر(عدد_seen_inputs);
    حرر(عدد_seen_targets);
    printf("OK\n");
}

صحيح main(فراغ) {

    // generate a few dummy shards of data with incrementing tokens
    صحيح header[HEADER_SIZE];
    uint16_t tokens[عدد_tokens];
    لكل (صحيح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحيح token_offset = shard_id * عدد_tokens;
        لكل (صحيح i = 0; i < عدد_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = عدد_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, حجم(صحيح), HEADER_SIZE, shard_file);
        fwrite(tokens, حجم(uint16_t), عدد_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    لكل (صحيح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    أرجع EXIT_SUCCESS;
}