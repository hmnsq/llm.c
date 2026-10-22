/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#અન્તર્ભાવ <unistd.h>
#અન્તર્ભાવ "../../llmc/dataloader.h"

#define SHARD_NAME_LEN 64
અક્ષર shard_name[SHARD_NAME_LEN];
અચલ પૂર્ણાંક અંક_tokens = 140;
પૂર્ણાંક અંક_shards = 4;

શૂન્ય check_range(અચલ પૂર્ણાંક *tokens, અચલ પૂર્ણાંક start, અચલ પૂર્ણાંક end, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    પૂર્ણાંક n = end - start;
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        પૂર્ણાંક token = tokens[i];
        જો (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", લીટી);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

શૂન્ય check_equals(અચલ પૂર્ણાંક *tokens, અચલ પૂર્ણાંક n, અચલ પૂર્ણાંક expected, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    // checks that the tokens[0, ... n] are all equal to expected
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        પૂર્ણાંક token = tokens[i];
        જો (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", લીટી);
            exit(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

શૂન્ય test_simple(શૂન્ય) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    printf("test_simple... ");
    પૂર્ણાંક B = 4;
    પૂર્ણાંક T = 8;
    પૂર્ણાંક process_rank = 0;
    પૂર્ણાંક અંક_processes = 1;
    પૂર્ણાંક should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, અંક_processes, should_shuffle);

    પૂર્ણાંક batches_fit = અંક_tokens / (B * T); // number of batches that fit per shard
    પૂર્ણાંક BT = B * T;
    પૂર્ણાંક અંક_epochs = 4;
    માટે (પૂર્ણાંક e = 0; e < અંક_epochs; e++) { // epoch
        માટે (પૂર્ણાંક s = 0; s < અંક_shards; s++) { // shard
            પૂર્ણાંક start = s * અંક_tokens;
            માટે (પૂર્ણાંક b = 0; b < batches_fit; b++) { // batch
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

શૂન્ય test_multiprocess_simple(શૂન્ય) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    printf("test_multiprocess_simple... ");
    પૂર્ણાંક B = 4;
    પૂર્ણાંક T = 8;
    પૂર્ણાંક અંક_processes = 2;
    પૂર્ણાંક should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, અંક_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, અંક_processes, should_shuffle);

    પૂર્ણાંક batches_fit = અંક_tokens / (B * T * અંક_processes); // number of batches that fit per shard
    પૂર્ણાંક BT = B * T;
    પૂર્ણાંક અંક_epochs = 4;
    માટે (પૂર્ણાંક e = 0; e < અંક_epochs; e++) { // epoch
        માટે (પૂર્ણાંક s = 0; s < અંક_shards; s++) { // shard
            પૂર્ણાંક start = s * અંક_tokens;
            માટે (પૂર્ણાંક b = 0; b < batches_fit; b++) { // batch
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

શૂન્ય test_shuffled(શૂન્ય) {
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
    પૂર્ણાંક B = 4;
    પૂર્ણાંક T = 8;
    પૂર્ણાંક process_rank = 0;
    પૂર્ણાંક અંક_processes = 1;
    પૂર્ણાંક should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, અંક_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    પૂર્ણાંક total_tokens = અંક_shards * અંક_tokens;
    પૂર્ણાંક *અંક_seen_inputs = (પૂર્ણાંક *)calloc(total_tokens, કદ(પૂર્ણાંક));
    પૂર્ણાંક *અંક_seen_targets = (પૂર્ણાંક *)calloc(total_tokens, કદ(પૂર્ણાંક));
    પૂર્ણાંક batches_fit = અંક_tokens / (B * T); // number of batches that fit per shard
    પૂર્ણાંક BT = B * T;
    પૂર્ણાંક અંક_epochs = 4;
    માટે (પૂર્ણાંક e = 0; e < અંક_epochs; e ++) { // epoch
        માટે (પૂર્ણાંક s = 0; s < અંક_shards; s++) { // shard
            પૂર્ણાંક start = s * અંક_tokens;
            માટે (પૂર્ણાંક b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                માટે (પૂર્ણાંક i = 0; i < BT; i++) {
                    પૂર્ણાંક આગત_token = loader.inputs[i];
                    પૂર્ણાંક target_token = loader.targets[i];
                    પ્રતિજ્ઞા(આગત_token >= 0 && આગત_token < total_tokens);
                    પ્રતિજ્ઞા(target_token >= 0 && target_token < total_tokens);
                    અંક_seen_inputs[આગત_token]++;
                    અંક_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    પૂર્ણાંક tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    માટે (પૂર્ણાંક s = 0; s < અંક_shards; s++) {
        પૂર્ણાંક start = s * અંક_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(અંક_seen_inputs + start, tokens_fit, અંક_epochs);
        checkEquals(અંક_seen_inputs + start + tokens_fit, અંક_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(અંક_seen_targets + start + 1, tokens_fit, અંક_epochs);
        checkEquals(અંક_seen_targets + start + 1 + tokens_fit,
            (s == (અંક_shards - 1)) ? અંક_tokens - tokens_fit - 1 : અંક_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    મુક્ત(અંક_seen_inputs);
    મુક્ત(અંક_seen_targets);
    printf("OK\n");
}

શૂન્ય test_multiprocess_shuffled(શૂન્ય) {
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
    પૂર્ણાંક B = 4;
    પૂર્ણાંક T = 8;
    અચલ પૂર્ણાંક અંક_processes = 2;
    પૂર્ણાંક should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[અંક_processes];
    માટે (પૂર્ણાંક i = 0; i < અંક_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, અંક_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    પૂર્ણાંક total_tokens = અંક_shards * અંક_tokens;
    પૂર્ણાંક *અંક_seen_inputs = (પૂર્ણાંક *)calloc(total_tokens, કદ(પૂર્ણાંક));
    પૂર્ણાંક *અંક_seen_targets = (પૂર્ણાંક *)calloc(total_tokens, કદ(પૂર્ણાંક));
    પૂર્ણાંક batches_fit = અંક_tokens / (B * T * અંક_processes); // number of batches that fit per shard
    પૂર્ણાંક BT = B * T;
    પૂર્ણાંક અંક_epochs = 4;
    માટે (પૂર્ણાંક e = 0; e < અંક_epochs; e ++) { // epoch
        માટે (પૂર્ણાંક s = 0; s < અંક_shards; s++) { // shard
            પૂર્ણાંક start = s * અંક_tokens;
            માટે (પૂર્ણાંક b = 0; b < batches_fit; b++) { // batch
                માટે (પૂર્ણાંક n = 0; n < અંક_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    માટે (પૂર્ણાંક i = 0; i < BT; i++) {
                        પૂર્ણાંક આગત_token = loader->inputs[i];
                        પૂર્ણાંક target_token = loader->targets[i];
                        પ્રતિજ્ઞા(આગત_token >= 0 && આગત_token < total_tokens);
                        પ્રતિજ્ઞા(target_token >= 0 && target_token < total_tokens);
                        અંક_seen_inputs[આગત_token]++;
                        અંક_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    પૂર્ણાંક tokens_fit = batches_fit * (B * T * અંક_processes); // number of tokens that fit per shard
    માટે (પૂર્ણાંક s = 0; s < અંક_shards; s++) {
        પૂર્ણાંક start = s * અંક_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(અંક_seen_inputs + start, tokens_fit, અંક_epochs);
        checkEquals(અંક_seen_inputs + start + tokens_fit, અંક_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(અંક_seen_targets + start + 1, tokens_fit, અંક_epochs);
        checkEquals(અંક_seen_targets + start + 1 + tokens_fit,
            (s == (અંક_shards - 1)) ? અંક_tokens - tokens_fit - 1 : અંક_tokens - tokens_fit,0);
    }

    // cleanup
    માટે (પૂર્ણાંક i = 0; i < અંક_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    મુક્ત(અંક_seen_inputs);
    મુક્ત(અંક_seen_targets);
    printf("OK\n");
}

પૂર્ણાંક મુખ્ય(શૂન્ય) {

    // generate a few dummy shards of data with incrementing tokens
    પૂર્ણાંક header[HEADER_SIZE];
    uint16_t tokens[અંક_tokens];
    માટે (પૂર્ણાંક shard_id = 0; shard_id < અંક_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        પૂર્ણાંક token_offset = shard_id * અંક_tokens;
        માટે (પૂર્ણાંક i = 0; i < અંક_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = અંક_tokens; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, કદ(પૂર્ણાંક), HEADER_SIZE, shard_file);
        fwrite(tokens, કદ(uint16_t), અંક_tokens, shard_file);
        fcloseCheck(shard_file);
        printf("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    માટે (પૂર્ણાંક shard_id = 0; shard_id < અંક_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    પાછા EXIT_SUCCESS;
}