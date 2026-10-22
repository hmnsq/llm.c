/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#अन्तर्भाव <unistd.h>
#अन्तर्भाव "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
वर्ण shard_name[SHARD_NAME_LEN];
ध्रुव पूर्णाङ्क अङ्कः_tokens = 140;
पूर्णाङ्क अङ्कः_shards = 4;

शून्य check_range(ध्रुव पूर्णाङ्क *tokens, ध्रुव पूर्णाङ्क start, ध्रुव पूर्णाङ्क end, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    पूर्णाङ्क n = end - start;
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        पूर्णाङ्क token = tokens[i];
        जइ (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

शून्य check_equals(ध्रुव पूर्णाङ्क *tokens, ध्रुव पूर्णाङ्क n, ध्रुव पूर्णाङ्क expected, ध्रुव वर्ण *file, पूर्णाङ्क पङ्क्तिः) {
    // checks that the tokens[0, ... n] are all equal to expected
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        पूर्णाङ्क token = tokens[i];
        जइ (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", पङ्क्तिः);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

शून्य test_simple(शून्य) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    पूर्णाङ्क process_rank = 0;
    पूर्णाङ्क अङ्कः_processes = 1;
    पूर्णाङ्क should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, अङ्कः_processes, should_shuffle);

    पूर्णाङ्क batches_fit = अङ्कः_tokens / (B * T); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्कः_epochs = 4;
    प्रति (पूर्णाङ्क e = 0; e < अङ्कः_epochs; e++) { // epoch
        प्रति (पूर्णाङ्क s = 0; s < अङ्कः_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्कः_tokens;
            प्रति (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
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

शून्य test_multiprocess_simple(शून्य) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    पूर्णाङ्क अङ्कः_processes = 2;
    पूर्णाङ्क should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, अङ्कः_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, अङ्कः_processes, should_shuffle);

    पूर्णाङ्क batches_fit = अङ्कः_tokens / (B * T * अङ्कः_processes); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्कः_epochs = 4;
    प्रति (पूर्णाङ्क e = 0; e < अङ्कः_epochs; e++) { // epoch
        प्रति (पूर्णाङ्क s = 0; s < अङ्कः_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्कः_tokens;
            प्रति (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
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

शून्य test_shuffled(शून्य) {
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
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    पूर्णाङ्क process_rank = 0;
    पूर्णाङ्क अङ्कः_processes = 1;
    पूर्णाङ्क should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, अङ्कः_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    पूर्णाङ्क total_tokens = अङ्कः_shards * अङ्कः_tokens;
    पूर्णाङ्क *अङ्कः_seen_inputs = (पूर्णाङ्क *)calloc(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क *अङ्कः_seen_targets = (पूर्णाङ्क *)calloc(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क batches_fit = अङ्कः_tokens / (B * T); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्कः_epochs = 4;
    प्रति (पूर्णाङ्क e = 0; e < अङ्कः_epochs; e ++) { // epoch
        प्रति (पूर्णाङ्क s = 0; s < अङ्कः_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्कः_tokens;
            प्रति (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                प्रति (पूर्णाङ्क i = 0; i < BT; i++) {
                    पूर्णाङ्क निवेशः_token = loader.inputs[i];
                    पूर्णाङ्क target_token = loader.targets[i];
                    प्रतिज्ञा(निवेशः_token >= 0 && निवेशः_token < total_tokens);
                    प्रतिज्ञा(target_token >= 0 && target_token < total_tokens);
                    अङ्कः_seen_inputs[निवेशः_token]++;
                    अङ्कः_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    पूर्णाङ्क tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    प्रति (पूर्णाङ्क s = 0; s < अङ्कः_shards; s++) {
        पूर्णाङ्क start = s * अङ्कः_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(अङ्कः_seen_inputs + start, tokens_fit, अङ्कः_epochs);
        checkEquals(अङ्कः_seen_inputs + start + tokens_fit, अङ्कः_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(अङ्कः_seen_targets + start + 1, tokens_fit, अङ्कः_epochs);
        checkEquals(अङ्कः_seen_targets + start + 1 + tokens_fit,
            (s == (अङ्कः_shards - 1)) ? अङ्कः_tokens - tokens_fit - 1 : अङ्कः_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    मुक्त(अङ्कः_seen_inputs);
    मुक्त(अङ्कः_seen_targets);
    printf("OK\n");
}

शून्य test_multiprocess_shuffled(शून्य) {
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
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    ध्रुव पूर्णाङ्क अङ्कः_processes = 2;
    पूर्णाङ्क should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[अङ्कः_processes];
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, अङ्कः_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    पूर्णाङ्क total_tokens = अङ्कः_shards * अङ्कः_tokens;
    पूर्णाङ्क *अङ्कः_seen_inputs = (पूर्णाङ्क *)calloc(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क *अङ्कः_seen_targets = (पूर्णाङ्क *)calloc(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क batches_fit = अङ्कः_tokens / (B * T * अङ्कः_processes); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्कः_epochs = 4;
    प्रति (पूर्णाङ्क e = 0; e < अङ्कः_epochs; e ++) { // epoch
        प्रति (पूर्णाङ्क s = 0; s < अङ्कः_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्कः_tokens;
            प्रति (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
                प्रति (पूर्णाङ्क n = 0; n < अङ्कः_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    प्रति (पूर्णाङ्क i = 0; i < BT; i++) {
                        पूर्णाङ्क निवेशः_token = loader->inputs[i];
                        पूर्णाङ्क target_token = loader->targets[i];
                        प्रतिज्ञा(निवेशः_token >= 0 && निवेशः_token < total_tokens);
                        प्रतिज्ञा(target_token >= 0 && target_token < total_tokens);
                        अङ्कः_seen_inputs[निवेशः_token]++;
                        अङ्कः_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    पूर्णाङ्क tokens_fit = batches_fit * (B * T * अङ्कः_processes); // number of tokens that fit per shard
    प्रति (पूर्णाङ्क s = 0; s < अङ्कः_shards; s++) {
        पूर्णाङ्क start = s * अङ्कः_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(अङ्कः_seen_inputs + start, tokens_fit, अङ्कः_epochs);
        checkEquals(अङ्कः_seen_inputs + start + tokens_fit, अङ्कः_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(अङ्कः_seen_targets + start + 1, tokens_fit, अङ्कः_epochs);
        checkEquals(अङ्कः_seen_targets + start + 1 + tokens_fit,
            (s == (अङ्कः_shards - 1)) ? अङ्कः_tokens - tokens_fit - 1 : अङ्कः_tokens - tokens_fit,0);
    }

    // cleanup
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    मुक्त(अङ्कः_seen_inputs);
    मुक्त(अङ्कः_seen_targets);
    printf("OK\n");
}

पूर्णाङ्क main(शून्य) {

    // generate a few dummy shards of data with incrementing tokens
    पूर्णाङ्क header[HEADER_SIZE];
    uint16_t tokens[अङ्कः_tokens];
    प्रति (पूर्णाङ्क shard_id = 0; shard_id < अङ्कः_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        पूर्णाङ्क token_offset = shard_id * अङ्कः_tokens;
        प्रति (पूर्णाङ्क i = 0; i < अङ्कः_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = अङ्कः_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, परिमाण(पूर्णाङ्क), HEADER_SIZE, shard_file);
        fwrite(tokens, परिमाण(uint16_t), अङ्कः_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    प्रति (पूर्णाङ्क shard_id = 0; shard_id < अङ्कः_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    प्रत्यागम EXIT_SUCCESS;
}