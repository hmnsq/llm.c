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
अचर पूर्णाङ्क अङ्क_tokens = 140;
पूर्णाङ्क अङ्क_shards = 4;

शून्य check_range(अचर पूर्णाङ्क *tokens, अचर पूर्णाङ्क start, अचर पूर्णाङ्क end, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    पूर्णाङ्क n = end - start;
    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        पूर्णाङ्क token = tokens[i];
        यदि (token != start + i) {
            ख_लिखो(मानक_त्रुटि, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
            निकास(निकास_त्रुटि);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

शून्य check_equals(अचर पूर्णाङ्क *tokens, अचर पूर्णाङ्क n, अचर पूर्णाङ्क expected, अचर वर्ण *file, पूर्णाङ्क पङ्क्ति) {
    // checks that the tokens[0, ... n] are all equal to expected
    लागि (पूर्णाङ्क i = 0; i < n; i++) {
        पूर्णाङ्क token = tokens[i];
        यदि (token != expected) {
            ख_लिखो(मानक_त्रुटि, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            ख_लिखो(मानक_त्रुटि, "Error details:\n");
            ख_लिखो(मानक_त्रुटि, "  File: %s\n", file);
            ख_लिखो(मानक_त्रुटि, "  Line: %d\n", पङ्क्ति);
            निकास(निकास_त्रुटि);
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
    म_लिखो("test_simple... ");
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    पूर्णाङ्क process_rank = 0;
    पूर्णाङ्क अङ्क_processes = 1;
    पूर्णाङ्क should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, अङ्क_processes, should_shuffle);

    पूर्णाङ्क batches_fit = अङ्क_tokens / (B * T); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्क_epochs = 4;
    लागि (पूर्णाङ्क e = 0; e < अङ्क_epochs; e++) { // epoch
        लागि (पूर्णाङ्क s = 0; s < अङ्क_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्क_tokens;
            लागि (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                checkRange(loader.inputs, start, start + BT);
                checkRange(loader.targets, start + 1, start + BT + 1);
                start += BT;
            }
        }
    }
    dataloader_free(&loader);
    म_लिखो("OK\n");
}

शून्य test_multiprocess_simple(शून्य) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    म_लिखो("test_multiprocess_simple... ");
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    पूर्णाङ्क अङ्क_processes = 2;
    पूर्णाङ्क should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, अङ्क_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, अङ्क_processes, should_shuffle);

    पूर्णाङ्क batches_fit = अङ्क_tokens / (B * T * अङ्क_processes); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्क_epochs = 4;
    लागि (पूर्णाङ्क e = 0; e < अङ्क_epochs; e++) { // epoch
        लागि (पूर्णाङ्क s = 0; s < अङ्क_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्क_tokens;
            लागि (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
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
    म_लिखो("OK\n");
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
    म_लिखो("test_shuffled... ");
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    पूर्णाङ्क process_rank = 0;
    पूर्णाङ्क अङ्क_processes = 1;
    पूर्णाङ्क should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, अङ्क_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    पूर्णाङ्क total_tokens = अङ्क_shards * अङ्क_tokens;
    पूर्णाङ्क *अङ्क_seen_inputs = (पूर्णाङ्क *)सुस्मृति(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क *अङ्क_seen_targets = (पूर्णाङ्क *)सुस्मृति(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क batches_fit = अङ्क_tokens / (B * T); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्क_epochs = 4;
    लागि (पूर्णाङ्क e = 0; e < अङ्क_epochs; e ++) { // epoch
        लागि (पूर्णाङ्क s = 0; s < अङ्क_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्क_tokens;
            लागि (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
                dataloader_next_batch(&loader);
                // count up the tokens we see
                लागि (पूर्णाङ्क i = 0; i < BT; i++) {
                    पूर्णाङ्क आगत_token = loader.inputs[i];
                    पूर्णाङ्क target_token = loader.targets[i];
                    प्रतिज्ञा(आगत_token >= 0 && आगत_token < total_tokens);
                    प्रतिज्ञा(target_token >= 0 && target_token < total_tokens);
                    अङ्क_seen_inputs[आगत_token]++;
                    अङ्क_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    पूर्णाङ्क tokens_fit = batches_fit * BT; // number of tokens that fit per shard
    लागि (पूर्णाङ्क s = 0; s < अङ्क_shards; s++) {
        पूर्णाङ्क start = s * अङ्क_tokens;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(अङ्क_seen_inputs + start, tokens_fit, अङ्क_epochs);
        checkEquals(अङ्क_seen_inputs + start + tokens_fit, अङ्क_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(अङ्क_seen_targets + start + 1, tokens_fit, अङ्क_epochs);
        checkEquals(अङ्क_seen_targets + start + 1 + tokens_fit,
            (s == (अङ्क_shards - 1)) ? अङ्क_tokens - tokens_fit - 1 : अङ्क_tokens - tokens_fit,0);
    }

    dataloader_free(&loader);
    मुक्त(अङ्क_seen_inputs);
    मुक्त(अङ्क_seen_targets);
    म_लिखो("OK\n");
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

    म_लिखो("test_multiprocess_shuffled... ");
    पूर्णाङ्क B = 4;
    पूर्णाङ्क T = 8;
    अचर पूर्णाङ्क अङ्क_processes = 2;
    पूर्णाङ्क should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[अङ्क_processes];
    लागि (पूर्णाङ्क i = 0; i < अङ्क_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, अङ्क_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    पूर्णाङ्क total_tokens = अङ्क_shards * अङ्क_tokens;
    पूर्णाङ्क *अङ्क_seen_inputs = (पूर्णाङ्क *)सुस्मृति(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क *अङ्क_seen_targets = (पूर्णाङ्क *)सुस्मृति(total_tokens, परिमाण(पूर्णाङ्क));
    पूर्णाङ्क batches_fit = अङ्क_tokens / (B * T * अङ्क_processes); // number of batches that fit per shard
    पूर्णाङ्क BT = B * T;
    पूर्णाङ्क अङ्क_epochs = 4;
    लागि (पूर्णाङ्क e = 0; e < अङ्क_epochs; e ++) { // epoch
        लागि (पूर्णाङ्क s = 0; s < अङ्क_shards; s++) { // shard
            पूर्णाङ्क start = s * अङ्क_tokens;
            लागि (पूर्णाङ्क b = 0; b < batches_fit; b++) { // batch
                लागि (पूर्णाङ्क n = 0; n < अङ्क_processes; n++) { // dataloader
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);
                    // count up the tokens we see
                    लागि (पूर्णाङ्क i = 0; i < BT; i++) {
                        पूर्णाङ्क आगत_token = loader->inputs[i];
                        पूर्णाङ्क target_token = loader->targets[i];
                        प्रतिज्ञा(आगत_token >= 0 && आगत_token < total_tokens);
                        प्रतिज्ञा(target_token >= 0 && target_token < total_tokens);
                        अङ्क_seen_inputs[आगत_token]++;
                        अङ्क_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    पूर्णाङ्क tokens_fit = batches_fit * (B * T * अङ्क_processes); // number of tokens that fit per shard
    लागि (पूर्णाङ्क s = 0; s < अङ्क_shards; s++) {
        पूर्णाङ्क start = s * अङ्क_tokens; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(अङ्क_seen_inputs + start, tokens_fit, अङ्क_epochs);
        checkEquals(अङ्क_seen_inputs + start + tokens_fit, अङ्क_tokens - tokens_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(अङ्क_seen_targets + start + 1, tokens_fit, अङ्क_epochs);
        checkEquals(अङ्क_seen_targets + start + 1 + tokens_fit,
            (s == (अङ्क_shards - 1)) ? अङ्क_tokens - tokens_fit - 1 : अङ्क_tokens - tokens_fit,0);
    }

    // cleanup
    लागि (पूर्णाङ्क i = 0; i < अङ्क_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    मुक्त(अङ्क_seen_inputs);
    मुक्त(अङ्क_seen_targets);
    म_लिखो("OK\n");
}

पूर्णाङ्क main(शून्य) {

    // generate a few dummy shards of data with incrementing tokens
    पूर्णाङ्क header[HEADER_SIZE];
    uint16_t tokens[अङ्क_tokens];
    लागि (पूर्णाङ्क shard_id = 0; shard_id < अङ्क_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        पूर्णाङ्क token_offset = shard_id * अङ्क_tokens;
        लागि (पूर्णाङ्क i = 0; i < अङ्क_tokens; i++) {
            tokens[i] = token_offset + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        header[0] = 20240520; // magic
        header[1] = 1; // version
        header[2] = अङ्क_tokens; // number of tokens within
        खाता* shard_file = fopenCheck(shard_name, "wb");
        ख_डालो(header, परिमाण(पूर्णाङ्क), HEADER_SIZE, shard_file);
        ख_डालो(tokens, परिमाण(uint16_t), अङ्क_tokens, shard_file);
        fcloseCheck(shard_file);
        म_लिखो("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    लागि (पूर्णाङ्क shard_id = 0; shard_id < अङ्क_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        हटाओ(shard_name);
    }

    फर्काऊ निकास_सफल;
}