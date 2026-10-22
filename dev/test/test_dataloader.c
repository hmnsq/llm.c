/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#অন্তর্ভাব <unistd.h>
#অন্তর্ভাব "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
অক্ষর shard_name[SHARD_NAME_LEN];
ধ্রুবক পূর্ণসংখ্যা অঙ্ক_tokens = 140;
পূর্ণসংখ্যা অঙ্ক_shards = 4;

শূন্য check_range(ধ্রুবক পূর্ণসংখ্যা *tokens, ধ্রুবক পূর্ণসংখ্যা start, ধ্রুবক পূর্ণসংখ্যা end, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    পূর্ণসংখ্যা n = end - start;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        পূর্ণসংখ্যা token = tokens[i];
        যদি (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", সারি);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

শূন্য check_equals(ধ্রুবক পূর্ণসংখ্যা *tokens, ধ্রুবক পূর্ণসংখ্যা n, ধ্রুবক পূর্ণসংখ্যা expected, ধ্রুবক অক্ষর *file, পূর্ণসংখ্যা সারি) {
    // checks that the tokens[0, ... n] are all equal to expected
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        পূর্ণসংখ্যা token = tokens[i];
        যদি (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", সারি);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

শূন্য test_simple(শূন্য) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    পূর্ণসংখ্যা B = 4;
    পূর্ণসংখ্যা T = 8;
    পূর্ণসংখ্যা process_rank = 0;
    পূর্ণসংখ্যা অঙ্ক_processes = 1;
    পূর্ণসংখ্যা should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    তথ্য_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, অঙ্ক_processes, should_shuffle);

    পূর্ণসংখ্যা batches_fit = অঙ্ক_tokens / (B * T); // number of batches that fit per shard
    পূর্ণসংখ্যা BT = B * T;
    পূর্ণসংখ্যা অঙ্ক_epochs = 4;
    জন্য (পূর্ণসংখ্যা e = 0; e < অঙ্ক_epochs; e++) { // epoch
        জন্য (পূর্ণসংখ্যা s = 0; s < অঙ্ক_shards; s++) { // shard
            পূর্ণসংখ্যা start = s * অঙ্ক_tokens;
            জন্য (পূর্ণসংখ্যা b = 0; b < batches_fit; b++) { // batch
                dataloader_পরবর্তী_batch(&loader);
                checkRange(loader.inputs, start, start + BT);
                checkRange(loader.targets, start + 1, start + BT + 1);
                start += BT;
            }
        }
    }
    dataloader_free(&loader);
    printf("OK\n");
}

শূন্য test_multiprocess_simple(শূন্য) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    পূর্ণসংখ্যা B = 4;
    পূর্ণসংখ্যা T = 8;
    পূর্ণসংখ্যা অঙ্ক_processes = 2;
    পূর্ণসংখ্যা should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    তথ্য_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, অঙ্ক_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, অঙ্ক_processes, should_shuffle);

    পূর্ণসংখ্যা batches_fit = অঙ্ক_tokens / (B * T * অঙ্ক_processes); // number of batches that fit per shard
    পূর্ণসংখ্যা BT = B * T;
    পূর্ণসংখ্যা অঙ্ক_epochs = 4;
    জন্য (পূর্ণসংখ্যা e = 0; e < অঙ্ক_epochs; e++) { // epoch
        জন্য (পূর্ণসংখ্যা s = 0; s < অঙ্ক_shards; s++) { // shard
            পূর্ণসংখ্যা start = s * অঙ্ক_tokens;
            জন্য (পূর্ণসংখ্যা b = 0; b < batches_fit; b++) { // batch
                dataloader_পরবর্তী_batch(&loader0);
                dataloader_পরবর্তী_batch(&loader1);
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

শূন্য test_shuffled(শূন্য) {
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
    পূর্ণসংখ্যা B = 4;
    পূর্ণসংখ্যা T = 8;
    পূর্ণসংখ্যা process_rank = 0;
    পূর্ণসংখ্যা অঙ্ক_processes = 1;
    পূর্ণসংখ্যা should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    তথ্য_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, অঙ্ক_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    পূর্ণসংখ্যা যোগ_tokens = অঙ্ক_shards * অঙ্ক_tokens;
    পূর্ণসংখ্যা *অঙ্ক_seen_inputs = (পূর্ণসংখ্যা *)calloc(যোগ_tokens, আকার(পূর্ণসংখ্যা));
    পূর্ণসংখ্যা *অঙ্ক_seen_targets = (পূর্ণসংখ্যা *)calloc(যোগ_tokens, আকার(পূর্ণসংখ্যা));
    পূর্ণসংখ্যা batches_fit = অঙ্ক_tokens / (B * T); // number of batches that fit per shard
    পূর্ণসংখ্যা BT = B * T;
    পূর্ণসংখ্যা অঙ্ক_epochs = 4;
    জন্য (পূর্ণসংখ্যা e = 0; e < অঙ্ক_epochs; e ++) { // epoch
        জন্য (পূর্ণসংখ্যা s = 0; s < অঙ্ক_shards; s++) { // shard
            পূর্ণসংখ্যা start = s * অঙ্ক_tokens;
            জন্য (পূর্ণসংখ্যা b = 0; b < batches_fit; b++) { // batch
                dataloader_পরবর্তী_batch(&loader);
                // count up the tokens we see
                জন্য (পূর্ণসংখ্যা i = 0; i < BT; i++) {
                    পূর্ণসংখ্যা প্রবেশ_token = loader.inputs[i];
                    পূর্ণসংখ্যা target_token = loader.targets[i];
                    প্রতিজ্ঞা(প্রবেশ_token >= 0 && প্রবেশ_token < যোগ_tokens);
                    প্রতিজ্ঞা(target_token >= 0 && target_token < যোগ_tokens);
                    অঙ্ক_seen_inputs[প্রবেশ_token]++;
                    অঙ্ক_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    পূর্ণসংখ্যা tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    জন্য (পূর্ণসংখ্যা s = 0; s < অঙ্ক_shards; s++) {
        পূর্ণসংখ্যা start = s * অঙ্ক_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(অঙ্ক_seen_inputs + start, tokens_fit, অঙ্ক_epochs);
        checkEquals(অঙ্ক_seen_inputs + start + tokens_fit, অঙ্ক_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(অঙ্ক_seen_targets + start + 1, tokens_fit, অঙ্ক_epochs);
        checkEquals(অঙ্ক_seen_targets + start + 1 + tokens_fit,
            (s == (অঙ্ক_shards - 1)) ? অঙ্ক_tokens - tokens_fit - 1 : অঙ্ক_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    মুক্ত(অঙ্ক_seen_inputs);
    মুক্ত(অঙ্ক_seen_targets);
    printf("OK\n");
}

শূন্য test_multiprocess_shuffled(শূন্য) {
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
    পূর্ণসংখ্যা B = 4;
    পূর্ণসংখ্যা T = 8;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_processes = 2;
    পূর্ণসংখ্যা should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    তথ্য_Loader loaders[অঙ্ক_processes];
    জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, অঙ্ক_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    পূর্ণসংখ্যা যোগ_tokens = অঙ্ক_shards * অঙ্ক_tokens;
    পূর্ণসংখ্যা *অঙ্ক_seen_inputs = (পূর্ণসংখ্যা *)calloc(যোগ_tokens, আকার(পূর্ণসংখ্যা));
    পূর্ণসংখ্যা *অঙ্ক_seen_targets = (পূর্ণসংখ্যা *)calloc(যোগ_tokens, আকার(পূর্ণসংখ্যা));
    পূর্ণসংখ্যা batches_fit = অঙ্ক_tokens / (B * T * অঙ্ক_processes); // number of batches that fit per shard
    পূর্ণসংখ্যা BT = B * T;
    পূর্ণসংখ্যা অঙ্ক_epochs = 4;
    জন্য (পূর্ণসংখ্যা e = 0; e < অঙ্ক_epochs; e ++) { // epoch
        জন্য (পূর্ণসংখ্যা s = 0; s < অঙ্ক_shards; s++) { // shard
            পূর্ণসংখ্যা start = s * অঙ্ক_tokens;
            জন্য (পূর্ণসংখ্যা b = 0; b < batches_fit; b++) { // batch
                জন্য (পূর্ণসংখ্যা n = 0; n < অঙ্ক_processes; n++) { // dataloader
                    তথ্য_Loader *loader = &loaders[n];
                    dataloader_পরবর্তী_batch(loader);
                    // count up the tokens we see
                    জন্য (পূর্ণসংখ্যা i = 0; i < BT; i++) {
                        পূর্ণসংখ্যা প্রবেশ_token = loader->inputs[i];
                        পূর্ণসংখ্যা target_token = loader->targets[i];
                        প্রতিজ্ঞা(প্রবেশ_token >= 0 && প্রবেশ_token < যোগ_tokens);
                        প্রতিজ্ঞা(target_token >= 0 && target_token < যোগ_tokens);
                        অঙ্ক_seen_inputs[প্রবেশ_token]++;
                        অঙ্ক_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    পূর্ণসংখ্যা tokens_fit = batches_fit * (B * T * অঙ্ক_processes); // number of tokens that fit per shard
    জন্য (পূর্ণসংখ্যা s = 0; s < অঙ্ক_shards; s++) {
        পূর্ণসংখ্যা start = s * অঙ্ক_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(অঙ্ক_seen_inputs + start, tokens_fit, অঙ্ক_epochs);
        checkEquals(অঙ্ক_seen_inputs + start + tokens_fit, অঙ্ক_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(অঙ্ক_seen_targets + start + 1, tokens_fit, অঙ্ক_epochs);
        checkEquals(অঙ্ক_seen_targets + start + 1 + tokens_fit,
            (s == (অঙ্ক_shards - 1)) ? অঙ্ক_tokens - tokens_fit - 1 : অঙ্ক_tokens - tokens_fit,0);
    }

    // cleanup
    জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    মুক্ত(অঙ্ক_seen_inputs);
    মুক্ত(অঙ্ক_seen_targets);
    printf("OK\n");
}

পূর্ণসংখ্যা প্রধান(শূন্য) {

    // generate a few dummy shards of data with incrementing tokens
    পূর্ণসংখ্যা header[HEADER_আকার];
    uint16_t tokens[অঙ্ক_tokens];
    জন্য (পূর্ণসংখ্যা shard_id = 0; shard_id < অঙ্ক_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        পূর্ণসংখ্যা token_offset = shard_id * অঙ্ক_tokens;
        জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = অঙ্ক_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, আকার(পূর্ণসংখ্যা), HEADER_আকার, shard_file);
        fwrite(tokens, আকার(uint16_t), অঙ্ক_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    জন্য (পূর্ণসংখ্যা shard_id = 0; shard_id < অঙ্ক_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    ফেরত EXIT_SUCCESS;
}