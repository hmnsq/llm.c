/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#شامل <unistd.h>
#شامل "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
نویسه shard_name[SHARD_NAME_LEN];
ثابت صحیح رقم_tokens = 140;
صحیح رقم_shards = 4;

تهی check_range(ثابت صحیح *tokens, ثابت صحیح start, ثابت صحیح end, ثابت نویسه *file, صحیح سطر) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحیح n = end - start;
    برای (صحیح i = 0; i < n; i++) {
        صحیح token = tokens[i];
        اگر (token != start + i) {
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

تهی check_equals(ثابت صحیح *tokens, ثابت صحیح n, ثابت صحیح expected, ثابت نویسه *file, صحیح سطر) {
    // checks that the tokens[0, ... n] are all equal to expected
    برای (صحیح i = 0; i < n; i++) {
        صحیح token = tokens[i];
        اگر (token != expected) {
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

تهی test_simple(تهی) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    صحیح B = 4;
    صحیح T = 8;
    صحیح process_rank = 0;
    صحیح رقم_processes = 1;
    صحیح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, رقم_processes, should_shuffle);

    صحیح batches_fit = رقم_tokens / (B * T); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح رقم_epochs = 4;
    برای (صحیح e = 0; e < رقم_epochs; e++) { // epoch
        برای (صحیح s = 0; s < رقم_shards; s++) { // shard
            صحیح start = s * رقم_tokens;
            برای (صحیح b = 0; b < batches_fit; b++) { // batch
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

تهی test_multiprocess_simple(تهی) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    صحیح B = 4;
    صحیح T = 8;
    صحیح رقم_processes = 2;
    صحیح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, رقم_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, رقم_processes, should_shuffle);

    صحیح batches_fit = رقم_tokens / (B * T * رقم_processes); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح رقم_epochs = 4;
    برای (صحیح e = 0; e < رقم_epochs; e++) { // epoch
        برای (صحیح s = 0; s < رقم_shards; s++) { // shard
            صحیح start = s * رقم_tokens;
            برای (صحیح b = 0; b < batches_fit; b++) { // batch
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

تهی test_shuffled(تهی) {
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
    صحیح B = 4;
    صحیح T = 8;
    صحیح process_rank = 0;
    صحیح رقم_processes = 1;
    صحیح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, رقم_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحیح total_tokens = رقم_shards * رقم_tokens;
    صحیح *رقم_seen_inputs = (صحیح *)calloc(total_tokens, اندازه(صحیح));
    صحیح *رقم_seen_targets = (صحیح *)calloc(total_tokens, اندازه(صحیح));
    صحیح batches_fit = رقم_tokens / (B * T); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح رقم_epochs = 4;
    برای (صحیح e = 0; e < رقم_epochs; e ++) { // epoch
        برای (صحیح s = 0; s < رقم_shards; s++) { // shard
            صحیح start = s * رقم_tokens;
            برای (صحیح b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                برای (صحیح i = 0; i < BT; i++) {
                    صحیح درون_داد_token = loader.inputs[i];
                    صحیح target_token = loader.targets[i];
                    أكد(درون_داد_token >= 0 && درون_داد_token < total_tokens);
                    أكد(target_token >= 0 && target_token < total_tokens);
                    رقم_seen_inputs[درون_داد_token]++;
                    رقم_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحیح tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    برای (صحیح s = 0; s < رقم_shards; s++) {
        صحیح start = s * رقم_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(رقم_seen_inputs + start, tokens_fit, رقم_epochs);
        checkEquals(رقم_seen_inputs + start + tokens_fit, رقم_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(رقم_seen_targets + start + 1, tokens_fit, رقم_epochs);
        checkEquals(رقم_seen_targets + start + 1 + tokens_fit,
            (s == (رقم_shards - 1)) ? رقم_tokens - tokens_fit - 1 : رقم_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    آزاد(رقم_seen_inputs);
    آزاد(رقم_seen_targets);
    printf("OK\n");
}

تهی test_multiprocess_shuffled(تهی) {
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
    صحیح B = 4;
    صحیح T = 8;
    ثابت صحیح رقم_processes = 2;
    صحیح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[رقم_processes];
    برای (صحیح i = 0; i < رقم_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, رقم_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحیح total_tokens = رقم_shards * رقم_tokens;
    صحیح *رقم_seen_inputs = (صحیح *)calloc(total_tokens, اندازه(صحیح));
    صحیح *رقم_seen_targets = (صحیح *)calloc(total_tokens, اندازه(صحیح));
    صحیح batches_fit = رقم_tokens / (B * T * رقم_processes); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح رقم_epochs = 4;
    برای (صحیح e = 0; e < رقم_epochs; e ++) { // epoch
        برای (صحیح s = 0; s < رقم_shards; s++) { // shard
            صحیح start = s * رقم_tokens;
            برای (صحیح b = 0; b < batches_fit; b++) { // batch
                برای (صحیح n = 0; n < رقم_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    برای (صحیح i = 0; i < BT; i++) {
                        صحیح درون_داد_token = loader->inputs[i];
                        صحیح target_token = loader->targets[i];
                        أكد(درون_داد_token >= 0 && درون_داد_token < total_tokens);
                        أكد(target_token >= 0 && target_token < total_tokens);
                        رقم_seen_inputs[درون_داد_token]++;
                        رقم_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحیح tokens_fit = batches_fit * (B * T * رقم_processes); // number of tokens that fit per shard
    برای (صحیح s = 0; s < رقم_shards; s++) {
        صحیح start = s * رقم_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(رقم_seen_inputs + start, tokens_fit, رقم_epochs);
        checkEquals(رقم_seen_inputs + start + tokens_fit, رقم_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(رقم_seen_targets + start + 1, tokens_fit, رقم_epochs);
        checkEquals(رقم_seen_targets + start + 1 + tokens_fit,
            (s == (رقم_shards - 1)) ? رقم_tokens - tokens_fit - 1 : رقم_tokens - tokens_fit,0);
    }

    // cleanup
    برای (صحیح i = 0; i < رقم_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    آزاد(رقم_seen_inputs);
    آزاد(رقم_seen_targets);
    printf("OK\n");
}

صحیح اصلی(تهی) {

    // generate a few dummy shards of data with incrementing tokens
    صحیح header[HEADER_SIZE];
    uint16_t tokens[رقم_tokens];
    برای (صحیح shard_id = 0; shard_id < رقم_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحیح token_offset = shard_id * رقم_tokens;
        برای (صحیح i = 0; i < رقم_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = رقم_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, اندازه(صحیح), HEADER_SIZE, shard_file);
        fwrite(tokens, اندازه(uint16_t), رقم_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    برای (صحیح shard_id = 0; shard_id < رقم_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    بازگشت EXIT_SUCCESS;
}