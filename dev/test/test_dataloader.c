/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#समावेश <unistd.h>
#समावेश "../../llmc/dataloader.h"

#घोषणा SHARD_NAME_LEN 64
अक्षर shard_name[SHARD_NAME_LEN];
स्थिर पूर्णांक अंक_tokens = 140;
पूर्णांक अंक_shards = 4;

व्योम check_range(स्थिर पूर्णांक *tokens, स्थिर पूर्णांक start, स्थिर पूर्णांक end, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    पूर्णांक n = end - start;
    क्रम (पूर्णांक i = 0; i < n; i++) {
        पूर्णांक token = tokens[i];
        अगर (token != start + i) {
            ख_लिखो(मानक_त्रुटि, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
            निकास(निकास_त्रुटि);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#घोषणा checkRange(tokens, start, end) check_range(tokens, start, end, __खाता__, __LINE__)

व्योम check_equals(स्थिर पूर्णांक *tokens, स्थिर पूर्णांक n, स्थिर पूर्णांक expected, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    // checks that the tokens[0, ... n] are all equal to expected
    क्रम (पूर्णांक i = 0; i < n; i++) {
        पूर्णांक token = tokens[i];
        अगर (token != expected) {
            ख_लिखो(मानक_त्रुटि, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पंक्ति);
            निकास(निकास_त्रुटि);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#घोषणा checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __खाता__, __LINE__)

व्योम test_simple(व्योम) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    म_लिखो("test_simple... ");
    पूर्णांक B = 4;
    पूर्णांक T = 8;
    पूर्णांक process_rank = 0;
    पूर्णांक अंक_processes = 1;
    पूर्णांक should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    तथ्य_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, अंक_processes, should_shuffle);

    पूर्णांक batches_fit = अंक_tokens / (B * T); // number of batches that fit per shard
    पूर्णांक BT = B * T;
    पूर्णांक अंक_epochs = 4;
    क्रम (पूर्णांक e = 0; e < अंक_epochs; e++) { // epoch
        क्रम (पूर्णांक s = 0; s < अंक_shards; s++) { // shard
            पूर्णांक start = s * अंक_tokens;
            क्रम (पूर्णांक b = 0; b < batches_fit; b++) { // batch
                dataloader_अग्र_batch(&loader);
                checkRange(loader.inputs, start, start + BT);
                checkRange(loader.targets, start + 1, start + BT + 1);
                start += BT;
            }
        }
    }
    dataloader_free(&loader);
    म_लिखो("OK\n");
}

व्योम test_multiprocess_simple(व्योम) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    म_लिखो("test_multiprocess_simple... ");
    पूर्णांक B = 4;
    पूर्णांक T = 8;
    पूर्णांक अंक_processes = 2;
    पूर्णांक should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    तथ्य_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, अंक_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, अंक_processes, should_shuffle);

    पूर्णांक batches_fit = अंक_tokens / (B * T * अंक_processes); // number of batches that fit per shard
    पूर्णांक BT = B * T;
    पूर्णांक अंक_epochs = 4;
    क्रम (पूर्णांक e = 0; e < अंक_epochs; e++) { // epoch
        क्रम (पूर्णांक s = 0; s < अंक_shards; s++) { // shard
            पूर्णांक start = s * अंक_tokens;
            क्रम (पूर्णांक b = 0; b < batches_fit; b++) { // batch
                dataloader_अग्र_batch(&loader0);
                dataloader_अग्र_batch(&loader1);
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
    म_लिखो("OK\n");
}

व्योम test_shuffled(व्योम) {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    म_लिखो("test_shuffled... ");
    पूर्णांक B = 4;
    पूर्णांक T = 8;
    पूर्णांक process_rank = 0;
    पूर्णांक अंक_processes = 1;
    पूर्णांक should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    तथ्य_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, अंक_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    पूर्णांक योग_tokens = अंक_shards * अंक_tokens;
    पूर्णांक *अंक_seen_inputs = (पूर्णांक *)सुस्मृति(योग_tokens, माप(पूर्णांक));
    पूर्णांक *अंक_seen_targets = (पूर्णांक *)सुस्मृति(योग_tokens, माप(पूर्णांक));
    पूर्णांक batches_fit = अंक_tokens / (B * T); // number of batches that fit per shard
    पूर्णांक BT = B * T;
    पूर्णांक अंक_epochs = 4;
    क्रम (पूर्णांक e = 0; e < अंक_epochs; e ++) { // epoch
        क्रम (पूर्णांक s = 0; s < अंक_shards; s++) { // shard
            पूर्णांक start = s * अंक_tokens;
            क्रम (पूर्णांक b = 0; b < batches_fit; b++) { // batch
                dataloader_अग्र_batch(&loader);
                // count up the tokens we see
                क्रम (पूर्णांक i = 0; i < BT; i++) {
                    पूर्णांक निवेश_token = loader.inputs[i];
                    पूर्णांक target_token = loader.targets[i];
                    निश्चित(निवेश_token >= 0 && निवेश_token < योग_tokens);
                    निश्चित(target_token >= 0 && target_token < योग_tokens);
                    अंक_seen_inputs[निवेश_token]++;
                    अंक_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    पूर्णांक tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    क्रम (पूर्णांक s = 0; s < अंक_shards; s++) {
        पूर्णांक start = s * अंक_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(अंक_seen_inputs + start, tokens_fit, अंक_epochs);
        checkEquals(अंक_seen_inputs + start + tokens_fit, अंक_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(अंक_seen_targets + start + 1, tokens_fit, अंक_epochs);
        checkEquals(अंक_seen_targets + start + 1 + tokens_fit,
            (s == (अंक_shards - 1)) ? अंक_tokens - tokens_fit - 1 : अंक_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    मुक्त(अंक_seen_inputs);
    मुक्त(अंक_seen_targets);
    म_लिखो("OK\n");
}

व्योम test_multiprocess_shuffled(व्योम) {
    /*
    Tests the DataLoader when using both multiprocess and shuffled:
    - multi-shard
    - multi-process
    - shuffled!
    DataLoaders should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times, over multiple epochs.
    */

    म_लिखो("test_multiprocess_shuffled... ");
    पूर्णांक B = 4;
    पूर्णांक T = 8;
    स्थिर पूर्णांक अंक_processes = 2;
    पूर्णांक should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    तथ्य_Loader loaders[अंक_processes];
    क्रम (पूर्णांक i = 0; i < अंक_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, अंक_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    पूर्णांक योग_tokens = अंक_shards * अंक_tokens;
    पूर्णांक *अंक_seen_inputs = (पूर्णांक *)सुस्मृति(योग_tokens, माप(पूर्णांक));
    पूर्णांक *अंक_seen_targets = (पूर्णांक *)सुस्मृति(योग_tokens, माप(पूर्णांक));
    पूर्णांक batches_fit = अंक_tokens / (B * T * अंक_processes); // number of batches that fit per shard
    पूर्णांक BT = B * T;
    पूर्णांक अंक_epochs = 4;
    क्रम (पूर्णांक e = 0; e < अंक_epochs; e ++) { // epoch
        क्रम (पूर्णांक s = 0; s < अंक_shards; s++) { // shard
            पूर्णांक start = s * अंक_tokens;
            क्रम (पूर्णांक b = 0; b < batches_fit; b++) { // batch
                क्रम (पूर्णांक n = 0; n < अंक_processes; n++) { // dataloader
                    तथ्य_Loader *loader = &loaders[n];
                    dataloader_अग्र_batch(loader);
                    // count up the tokens we see
                    क्रम (पूर्णांक i = 0; i < BT; i++) {
                        पूर्णांक निवेश_token = loader->inputs[i];
                        पूर्णांक target_token = loader->targets[i];
                        निश्चित(निवेश_token >= 0 && निवेश_token < योग_tokens);
                        निश्चित(target_token >= 0 && target_token < योग_tokens);
                        अंक_seen_inputs[निवेश_token]++;
                        अंक_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    पूर्णांक tokens_fit = batches_fit * (B * T * अंक_processes); // number of tokens that fit per shard
    क्रम (पूर्णांक s = 0; s < अंक_shards; s++) {
        पूर्णांक start = s * अंक_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(अंक_seen_inputs + start, tokens_fit, अंक_epochs);
        checkEquals(अंक_seen_inputs + start + tokens_fit, अंक_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(अंक_seen_targets + start + 1, tokens_fit, अंक_epochs);
        checkEquals(अंक_seen_targets + start + 1 + tokens_fit,
            (s == (अंक_shards - 1)) ? अंक_tokens - tokens_fit - 1 : अंक_tokens - tokens_fit,0);
    }

    // cleanup
    क्रम (पूर्णांक i = 0; i < अंक_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    मुक्त(अंक_seen_inputs);
    मुक्त(अंक_seen_targets);
    म_लिखो("OK\n");
}

पूर्णांक मुख्य(व्योम) {

    // generate a few dummy shards of data with incrementing tokens
    पूर्णांक header[HEADER_आकार];
    uint16_t tokens[अंक_tokens];
    क्रम (पूर्णांक shard_id = 0; shard_id < अंक_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        पूर्णांक token_offset = shard_id * अंक_tokens;
        क्रम (पूर्णांक i = 0; i < अंक_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = अंक_tokens; // number of tokens within
        खाता* shard_file = fopenCheck(shard_name, "wb");
        ख_डालो(header, माप(पूर्णांक), HEADER_आकार, shard_file);
        ख_डालो(tokens, माप(uint16_t), अंक_tokens, shard_file);
        fcloseCheck(shard_file);
        म_लिखो("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    क्रम (पूर्णांक shard_id = 0; shard_id < अंक_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        हटाओ(shard_name);
    }

    वापस निकास_सफल;
}