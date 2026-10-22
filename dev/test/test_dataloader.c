/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#include <unistd.h>
#include "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
var_na shard_name[SHARD_NAME_LEN];
const ཧྲིལ num_tokens = 140;
ཧྲིལ num_shards = 4;

void check_range(const ཧྲིལ *tokens, const ཧྲིལ start, const ཧྲིལ end, const var_na *file, ཧྲིལ line) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    ཧྲིལ n = end - start;
    དོན་ལུ (ཧྲིལ i = 0; i < n; i++) {
        ཧྲིལ token = tokens[i];
        གལ་སྲིད (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

void check_equals(const ཧྲིལ *tokens, const ཧྲིལ n, const ཧྲིལ expected, const var_na *file, ཧྲིལ line) {
    // checks that the tokens[0, ... n] are all equal to expected
    དོན་ལུ (ཧྲིལ i = 0; i < n; i++) {
        ཧྲིལ token = tokens[i];
        གལ་སྲིད (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

void test_simple(void) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    ཧྲིལ B = 4;
    ཧྲིལ T = 8;
    ཧྲིལ process_rank = 0;
    ཧྲིལ num_processes = 1;
    ཧྲིལ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, num_processes, should_shuffle);

    ཧྲིལ batches_fit = num_tokens / (B * T); // number of batches that fit per shard
    ཧྲིལ BT = B * T;
    ཧྲིལ num_epochs = 4;
    དོན་ལུ (ཧྲིལ e = 0; e < num_epochs; e++) { // epoch
        དོན་ལུ (ཧྲིལ s = 0; s < num_shards; s++) { // shard
            ཧྲིལ start = s * num_tokens;
            དོན་ལུ (ཧྲིལ b = 0; b < batches_fit; b++) { // batch
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

void test_multiprocess_simple(void) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    ཧྲིལ B = 4;
    ཧྲིལ T = 8;
    ཧྲིལ num_processes = 2;
    ཧྲིལ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, num_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, num_processes, should_shuffle);

    ཧྲིལ batches_fit = num_tokens / (B * T * num_processes); // number of batches that fit per shard
    ཧྲིལ BT = B * T;
    ཧྲིལ num_epochs = 4;
    དོན་ལུ (ཧྲིལ e = 0; e < num_epochs; e++) { // epoch
        དོན་ལུ (ཧྲིལ s = 0; s < num_shards; s++) { // shard
            ཧྲིལ start = s * num_tokens;
            དོན་ལུ (ཧྲིལ b = 0; b < batches_fit; b++) { // batch
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

void test_shuffled(void) {
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
    ཧྲིལ B = 4;
    ཧྲིལ T = 8;
    ཧྲིལ process_rank = 0;
    ཧྲིལ num_processes = 1;
    ཧྲིལ should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, num_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    ཧྲིལ total_tokens = num_shards * num_tokens;
    ཧྲིལ *num_seen_inputs = (ཧྲིལ *)calloc(total_tokens, parimaa_na(ཧྲིལ));
    ཧྲིལ *num_seen_targets = (ཧྲིལ *)calloc(total_tokens, parimaa_na(ཧྲིལ));
    ཧྲིལ batches_fit = num_tokens / (B * T); // number of batches that fit per shard
    ཧྲིལ BT = B * T;
    ཧྲིལ num_epochs = 4;
    དོན་ལུ (ཧྲིལ e = 0; e < num_epochs; e ++) { // epoch
        དོན་ལུ (ཧྲིལ s = 0; s < num_shards; s++) { // shard
            ཧྲིལ start = s * num_tokens;
            དོན་ལུ (ཧྲིལ b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                དོན་ལུ (ཧྲིལ i = 0; i < BT; i++) {
                    ཧྲིལ input_token = loader.inputs[i];
                    ཧྲིལ target_token = loader.targets[i];
                    assert(input_token >= 0 && input_token < total_tokens);
                    assert(target_token >= 0 && target_token < total_tokens);
                    num_seen_inputs[input_token]++;
                    num_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ཧྲིལ tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    དོན་ལུ (ཧྲིལ s = 0; s < num_shards; s++) {
        ཧྲིལ start = s * num_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(num_seen_inputs + start, tokens_fit, num_epochs);
        checkEquals(num_seen_inputs + start + tokens_fit, num_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(num_seen_targets + start + 1, tokens_fit, num_epochs);
        checkEquals(num_seen_targets + start + 1 + tokens_fit,
            (s == (num_shards - 1)) ? num_tokens - tokens_fit - 1 : num_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    free(num_seen_inputs);
    free(num_seen_targets);
    printf("OK\n");
}

void test_multiprocess_shuffled(void) {
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
    ཧྲིལ B = 4;
    ཧྲིལ T = 8;
    const ཧྲིལ num_processes = 2;
    ཧྲིལ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[num_processes];
    དོན་ལུ (ཧྲིལ i = 0; i < num_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, num_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    ཧྲིལ total_tokens = num_shards * num_tokens;
    ཧྲིལ *num_seen_inputs = (ཧྲིལ *)calloc(total_tokens, parimaa_na(ཧྲིལ));
    ཧྲིལ *num_seen_targets = (ཧྲིལ *)calloc(total_tokens, parimaa_na(ཧྲིལ));
    ཧྲིལ batches_fit = num_tokens / (B * T * num_processes); // number of batches that fit per shard
    ཧྲིལ BT = B * T;
    ཧྲིལ num_epochs = 4;
    དོན་ལུ (ཧྲིལ e = 0; e < num_epochs; e ++) { // epoch
        དོན་ལུ (ཧྲིལ s = 0; s < num_shards; s++) { // shard
            ཧྲིལ start = s * num_tokens;
            དོན་ལུ (ཧྲིལ b = 0; b < batches_fit; b++) { // batch
                དོན་ལུ (ཧྲིལ n = 0; n < num_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    དོན་ལུ (ཧྲིལ i = 0; i < BT; i++) {
                        ཧྲིལ input_token = loader->inputs[i];
                        ཧྲིལ target_token = loader->targets[i];
                        assert(input_token >= 0 && input_token < total_tokens);
                        assert(target_token >= 0 && target_token < total_tokens);
                        num_seen_inputs[input_token]++;
                        num_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ཧྲིལ tokens_fit = batches_fit * (B * T * num_processes); // number of tokens that fit per shard
    དོན་ལུ (ཧྲིལ s = 0; s < num_shards; s++) {
        ཧྲིལ start = s * num_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(num_seen_inputs + start, tokens_fit, num_epochs);
        checkEquals(num_seen_inputs + start + tokens_fit, num_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(num_seen_targets + start + 1, tokens_fit, num_epochs);
        checkEquals(num_seen_targets + start + 1 + tokens_fit,
            (s == (num_shards - 1)) ? num_tokens - tokens_fit - 1 : num_tokens - tokens_fit,0);
    }

    // cleanup
    དོན་ལུ (ཧྲིལ i = 0; i < num_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    free(num_seen_inputs);
    free(num_seen_targets);
    printf("OK\n");
}

ཧྲིལ main(void) {

    // generate a few dummy shards of data with incrementing tokens
    ཧྲིལ header[HEADER_SIZE];
    uint16_t tokens[num_tokens];
    དོན་ལུ (ཧྲིལ shard_id = 0; shard_id < num_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        ཧྲིལ token_offset = shard_id * num_tokens;
        དོན་ལུ (ཧྲིལ i = 0; i < num_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = num_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, parimaa_na(ཧྲིལ), HEADER_SIZE, shard_file);
        fwrite(tokens, parimaa_na(uint16_t), num_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    དོན་ལུ (ཧྲིལ shard_id = 0; shard_id < num_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    ལོག EXIT_SUCCESS;
}