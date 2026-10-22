/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#அந்தர்பாவ <unistd.h>
#அந்தர்பாவ "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
எழுத்து shard_name[SHARD_NAME_LEN];
மாறிலி முழு இலக்கம்_tokens = 140;
முழு இலக்கம்_shards = 4;

வெறுமை check_range(மாறிலி முழு *tokens, மாறிலி முழு start, மாறிலி முழு end, மாறிலி எழுத்து *file, முழு வரி) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    முழு n = end - start;
    க்கு (முழு i = 0; i < n; i++) {
        முழு token = tokens[i];
        எனில் (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", வரி);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

வெறுமை check_equals(மாறிலி முழு *tokens, மாறிலி முழு n, மாறிலி முழு expected, மாறிலி எழுத்து *file, முழு வரி) {
    // checks that the tokens[0, ... n] are all equal to expected
    க்கு (முழு i = 0; i < n; i++) {
        முழு token = tokens[i];
        எனில் (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", வரி);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

வெறுமை test_simple(வெறுமை) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    முழு B = 4;
    முழு T = 8;
    முழு process_rank = 0;
    முழு இலக்கம்_processes = 1;
    முழு should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, இலக்கம்_processes, should_shuffle);

    முழு batches_fit = இலக்கம்_tokens / (B * T); // number of batches that fit per shard
    முழு BT = B * T;
    முழு இலக்கம்_epochs = 4;
    க்கு (முழு e = 0; e < இலக்கம்_epochs; e++) { // epoch
        க்கு (முழு s = 0; s < இலக்கம்_shards; s++) { // shard
            முழு start = s * இலக்கம்_tokens;
            க்கு (முழு b = 0; b < batches_fit; b++) { // batch
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

வெறுமை test_multiprocess_simple(வெறுமை) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    முழு B = 4;
    முழு T = 8;
    முழு இலக்கம்_processes = 2;
    முழு should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, இலக்கம்_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, இலக்கம்_processes, should_shuffle);

    முழு batches_fit = இலக்கம்_tokens / (B * T * இலக்கம்_processes); // number of batches that fit per shard
    முழு BT = B * T;
    முழு இலக்கம்_epochs = 4;
    க்கு (முழு e = 0; e < இலக்கம்_epochs; e++) { // epoch
        க்கு (முழு s = 0; s < இலக்கம்_shards; s++) { // shard
            முழு start = s * இலக்கம்_tokens;
            க்கு (முழு b = 0; b < batches_fit; b++) { // batch
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

வெறுமை test_shuffled(வெறுமை) {
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
    முழு B = 4;
    முழு T = 8;
    முழு process_rank = 0;
    முழு இலக்கம்_processes = 1;
    முழு should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, இலக்கம்_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    முழு total_tokens = இலக்கம்_shards * இலக்கம்_tokens;
    முழு *இலக்கம்_seen_inputs = (முழு *)calloc(total_tokens, அளவு(முழு));
    முழு *இலக்கம்_seen_targets = (முழு *)calloc(total_tokens, அளவு(முழு));
    முழு batches_fit = இலக்கம்_tokens / (B * T); // number of batches that fit per shard
    முழு BT = B * T;
    முழு இலக்கம்_epochs = 4;
    க்கு (முழு e = 0; e < இலக்கம்_epochs; e ++) { // epoch
        க்கு (முழு s = 0; s < இலக்கம்_shards; s++) { // shard
            முழு start = s * இலக்கம்_tokens;
            க்கு (முழு b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                க்கு (முழு i = 0; i < BT; i++) {
                    முழு உள்ளீடு_token = loader.inputs[i];
                    முழு target_token = loader.targets[i];
                    ப்ரதிஜ்ஞா(உள்ளீடு_token >= 0 && உள்ளீடு_token < total_tokens);
                    ப்ரதிஜ்ஞா(target_token >= 0 && target_token < total_tokens);
                    இலக்கம்_seen_inputs[உள்ளீடு_token]++;
                    இலக்கம்_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    முழு tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    க்கு (முழு s = 0; s < இலக்கம்_shards; s++) {
        முழு start = s * இலக்கம்_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(இலக்கம்_seen_inputs + start, tokens_fit, இலக்கம்_epochs);
        checkEquals(இலக்கம்_seen_inputs + start + tokens_fit, இலக்கம்_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(இலக்கம்_seen_targets + start + 1, tokens_fit, இலக்கம்_epochs);
        checkEquals(இலக்கம்_seen_targets + start + 1 + tokens_fit,
            (s == (இலக்கம்_shards - 1)) ? இலக்கம்_tokens - tokens_fit - 1 : இலக்கம்_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    விடுவி(இலக்கம்_seen_inputs);
    விடுவி(இலக்கம்_seen_targets);
    printf("OK\n");
}

வெறுமை test_multiprocess_shuffled(வெறுமை) {
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
    முழு B = 4;
    முழு T = 8;
    மாறிலி முழு இலக்கம்_processes = 2;
    முழு should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[இலக்கம்_processes];
    க்கு (முழு i = 0; i < இலக்கம்_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, இலக்கம்_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    முழு total_tokens = இலக்கம்_shards * இலக்கம்_tokens;
    முழு *இலக்கம்_seen_inputs = (முழு *)calloc(total_tokens, அளவு(முழு));
    முழு *இலக்கம்_seen_targets = (முழு *)calloc(total_tokens, அளவு(முழு));
    முழு batches_fit = இலக்கம்_tokens / (B * T * இலக்கம்_processes); // number of batches that fit per shard
    முழு BT = B * T;
    முழு இலக்கம்_epochs = 4;
    க்கு (முழு e = 0; e < இலக்கம்_epochs; e ++) { // epoch
        க்கு (முழு s = 0; s < இலக்கம்_shards; s++) { // shard
            முழு start = s * இலக்கம்_tokens;
            க்கு (முழு b = 0; b < batches_fit; b++) { // batch
                க்கு (முழு n = 0; n < இலக்கம்_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    க்கு (முழு i = 0; i < BT; i++) {
                        முழு உள்ளீடு_token = loader->inputs[i];
                        முழு target_token = loader->targets[i];
                        ப்ரதிஜ்ஞா(உள்ளீடு_token >= 0 && உள்ளீடு_token < total_tokens);
                        ப்ரதிஜ்ஞா(target_token >= 0 && target_token < total_tokens);
                        இலக்கம்_seen_inputs[உள்ளீடு_token]++;
                        இலக்கம்_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    முழு tokens_fit = batches_fit * (B * T * இலக்கம்_processes); // number of tokens that fit per shard
    க்கு (முழு s = 0; s < இலக்கம்_shards; s++) {
        முழு start = s * இலக்கம்_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(இலக்கம்_seen_inputs + start, tokens_fit, இலக்கம்_epochs);
        checkEquals(இலக்கம்_seen_inputs + start + tokens_fit, இலக்கம்_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(இலக்கம்_seen_targets + start + 1, tokens_fit, இலக்கம்_epochs);
        checkEquals(இலக்கம்_seen_targets + start + 1 + tokens_fit,
            (s == (இலக்கம்_shards - 1)) ? இலக்கம்_tokens - tokens_fit - 1 : இலக்கம்_tokens - tokens_fit,0);
    }

    // cleanup
    க்கு (முழு i = 0; i < இலக்கம்_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    விடுவி(இலக்கம்_seen_inputs);
    விடுவி(இலக்கம்_seen_targets);
    printf("OK\n");
}

முழு முதன்மை(வெறுமை) {

    // generate a few dummy shards of data with incrementing tokens
    முழு header[HEADER_SIZE];
    uint16_t tokens[இலக்கம்_tokens];
    க்கு (முழு shard_id = 0; shard_id < இலக்கம்_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        முழு token_offset = shard_id * இலக்கம்_tokens;
        க்கு (முழு i = 0; i < இலக்கம்_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = இலக்கம்_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, அளவு(முழு), HEADER_SIZE, shard_file);
        fwrite(tokens, அளவு(uint16_t), இலக்கம்_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    க்கு (முழு shard_id = 0; shard_id < இலக்கம்_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    திரும்பு EXIT_SUCCESS;
}