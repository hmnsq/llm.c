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
char shard_name[SHARD_NAME_LEN];
const int नुमेरो_tokens = 140;
int नुमेरो_shards = 4;

void check_range(const int *tokens, const int start, const int end, const char *file, int लिन्या) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    int n = end - start;
    for (int i = 0; i < n; i++) {
        int token = tokens[i];
        if (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", लिन्या);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

void check_equals(const int *tokens, const int n, const int expected, const char *file, int लिन्या) {
    // checks that the tokens[0, ... n] are all equal to expected
    for (int i = 0; i < n; i++) {
        int token = tokens[i];
        if (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", लिन्या);
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
    int B = 4;
    int T = 8;
    int process_rank = 0;
    int नुमेरो_processes = 1;
    int should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, नुमेरो_processes, should_shuffle);

    int batches_fit = नुमेरो_tokens / (B * T); // number of batches that fit per shard
    int BT = B * T;
    int नुमेरो_epochs = 4;
    for (int e = 0; e < नुमेरो_epochs; e++) { // epoch
        for (int s = 0; s < नुमेरो_shards; s++) { // shard
            int start = s * नुमेरो_tokens;
            for (int b = 0; b < batches_fit; b++) { // batch
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
    int B = 4;
    int T = 8;
    int नुमेरो_processes = 2;
    int should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, नुमेरो_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, नुमेरो_processes, should_shuffle);

    int batches_fit = नुमेरो_tokens / (B * T * नुमेरो_processes); // number of batches that fit per shard
    int BT = B * T;
    int नुमेरो_epochs = 4;
    for (int e = 0; e < नुमेरो_epochs; e++) { // epoch
        for (int s = 0; s < नुमेरो_shards; s++) { // shard
            int start = s * नुमेरो_tokens;
            for (int b = 0; b < batches_fit; b++) { // batch
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
    int B = 4;
    int T = 8;
    int process_rank = 0;
    int नुमेरो_processes = 1;
    int should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, नुमेरो_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    int total_tokens = नुमेरो_shards * नुमेरो_tokens;
    int *नुमेरो_seen_inputs = (int *)calloc(total_tokens, sizeof(int));
    int *नुमेरो_seen_targets = (int *)calloc(total_tokens, sizeof(int));
    int batches_fit = नुमेरो_tokens / (B * T); // number of batches that fit per shard
    int BT = B * T;
    int नुमेरो_epochs = 4;
    for (int e = 0; e < नुमेरो_epochs; e ++) { // epoch
        for (int s = 0; s < नुमेरो_shards; s++) { // shard
            int start = s * नुमेरो_tokens;
            for (int b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                for (int i = 0; i < BT; i++) {
                    int पासोक_token = loader.inputs[i];
                    int target_token = loader.targets[i];
                    assert(पासोक_token >= 0 && पासोक_token < total_tokens);
                    assert(target_token >= 0 && target_token < total_tokens);
                    नुमेरो_seen_inputs[पासोक_token]++;
                    नुमेरो_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    int tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    for (int s = 0; s < नुमेरो_shards; s++) {
        int start = s * नुमेरो_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(नुमेरो_seen_inputs + start, tokens_fit, नुमेरो_epochs);
        checkEquals(नुमेरो_seen_inputs + start + tokens_fit, नुमेरो_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(नुमेरो_seen_targets + start + 1, tokens_fit, नुमेरो_epochs);
        checkEquals(नुमेरो_seen_targets + start + 1 + tokens_fit,
            (s == (नुमेरो_shards - 1)) ? नुमेरो_tokens - tokens_fit - 1 : नुमेरो_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    free(नुमेरो_seen_inputs);
    free(नुमेरो_seen_targets);
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
    int B = 4;
    int T = 8;
    const int नुमेरो_processes = 2;
    int should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[नुमेरो_processes];
    for (int i = 0; i < नुमेरो_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, नुमेरो_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    int total_tokens = नुमेरो_shards * नुमेरो_tokens;
    int *नुमेरो_seen_inputs = (int *)calloc(total_tokens, sizeof(int));
    int *नुमेरो_seen_targets = (int *)calloc(total_tokens, sizeof(int));
    int batches_fit = नुमेरो_tokens / (B * T * नुमेरो_processes); // number of batches that fit per shard
    int BT = B * T;
    int नुमेरो_epochs = 4;
    for (int e = 0; e < नुमेरो_epochs; e ++) { // epoch
        for (int s = 0; s < नुमेरो_shards; s++) { // shard
            int start = s * नुमेरो_tokens;
            for (int b = 0; b < batches_fit; b++) { // batch
                for (int n = 0; n < नुमेरो_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    for (int i = 0; i < BT; i++) {
                        int पासोक_token = loader->inputs[i];
                        int target_token = loader->targets[i];
                        assert(पासोक_token >= 0 && पासोक_token < total_tokens);
                        assert(target_token >= 0 && target_token < total_tokens);
                        नुमेरो_seen_inputs[पासोक_token]++;
                        नुमेरो_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    int tokens_fit = batches_fit * (B * T * नुमेरो_processes); // number of tokens that fit per shard
    for (int s = 0; s < नुमेरो_shards; s++) {
        int start = s * नुमेरो_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(नुमेरो_seen_inputs + start, tokens_fit, नुमेरो_epochs);
        checkEquals(नुमेरो_seen_inputs + start + tokens_fit, नुमेरो_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(नुमेरो_seen_targets + start + 1, tokens_fit, नुमेरो_epochs);
        checkEquals(नुमेरो_seen_targets + start + 1 + tokens_fit,
            (s == (नुमेरो_shards - 1)) ? नुमेरो_tokens - tokens_fit - 1 : नुमेरो_tokens - tokens_fit,0);
    }

    // cleanup
    for (int i = 0; i < नुमेरो_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    free(नुमेरो_seen_inputs);
    free(नुमेरो_seen_targets);
    printf("OK\n");
}

int main(void) {

    // generate a few dummy shards of data with incrementing tokens
    int header[HEADER_SIZE];
    uint16_t tokens[नुमेरो_tokens];
    for (int shard_id = 0; shard_id < नुमेरो_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        int token_offset = shard_id * नुमेरो_tokens;
        for (int i = 0; i < नुमेरो_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = नुमेरो_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, sizeof(int), HEADER_SIZE, shard_file);
        fwrite(tokens, sizeof(uint16_t), नुमेरो_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    for (int shard_id = 0; shard_id < नुमेरो_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    return EXIT_SUCCESS;
}