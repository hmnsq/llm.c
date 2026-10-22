/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ "../../llmc/dataloader.h"

#ವ್ಯಾಖ್ಯಾನಿಸು SHARD_NAME_LEN 64
ಅಕ್ಷರ shard_name[SHARD_NAME_LEN];
ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಟೋಕನುಗಳು = 140;
ಪೂರ್ಣಾಂಕ ಅಂಕೆ_shards = 4;

ಶೂನ್ಯ check_range(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *ಟೋಕನುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಆರಂಭ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಕೊನೆ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    ಪೂರ್ಣಾಂಕ n = ಕೊನೆ - ಆರಂಭ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ಪೂರ್ಣಾಂಕ token = ಟೋಕನುಗಳು[i];
        ಆದರೆ (token != ಆರಂಭ + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, ಆರಂಭ + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ಸಾಲು);
            ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#ವ್ಯಾಖ್ಯಾನಿಸು checkRange(ಟೋಕನುಗಳು, ಆರಂಭ, ಕೊನೆ) check_range(ಟೋಕನುಗಳು, ಆರಂಭ, ಕೊನೆ, __FILE__, __LINE__)

ಶೂನ್ಯ check_equals(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *ಟೋಕನುಗಳು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ expected, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    // checks that the tokens[0, ... n] are all equal to expected
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        ಪೂರ್ಣಾಂಕ token = ಟೋಕನುಗಳು[i];
        ಆದರೆ (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ಸಾಲು);
            ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#ವ್ಯಾಖ್ಯಾನಿಸು checkEquals(ಟೋಕನುಗಳು, n, expected) check_equals(ಟೋಕನುಗಳು, n, expected, __FILE__, __LINE__)

ಶೂನ್ಯ test_simple(ಶೂನ್ಯ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    ಬರೆ("test_simple... ");
    ಪೂರ್ಣಾಂಕ B = 4;
    ಪೂರ್ಣಾಂಕ T = 8;
    ಪೂರ್ಣಾಂಕ process_rank = 0;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes = 1;
    ಪೂರ್ಣಾಂಕ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ದತ್ತಾಂಶ_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ಅಂಕೆ_processes, should_shuffle);

    ಪೂರ್ಣಾಂಕ batches_fit = ಅಂಕೆ_ಟೋಕನುಗಳು / (B * T); // number of batches that fit per shard
    ಪೂರ್ಣಾಂಕ BT = B * T;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_epochs = 4;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ e = 0; e < ಅಂಕೆ_epochs; e++) { // epoch
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಂಕ ಆರಂಭ = s * ಅಂಕೆ_ಟೋಕನುಗಳು;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < batches_fit; b++) { // batch
                dataloader_ಮುಂದಿನ_batch(&loader);
                checkRange(loader.inputs, ಆರಂಭ, ಆರಂಭ + BT);
                checkRange(loader.targets, ಆರಂಭ + 1, ಆರಂಭ + BT + 1);
                ಆರಂಭ += BT;
            }
        }
    }
    dataloader_free(&loader);
    ಬರೆ("OK\n");
}

ಶೂನ್ಯ test_multiprocess_simple(ಶೂನ್ಯ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    ಬರೆ("test_multiprocess_simple... ");
    ಪೂರ್ಣಾಂಕ B = 4;
    ಪೂರ್ಣಾಂಕ T = 8;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes = 2;
    ಪೂರ್ಣಾಂಕ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ದತ್ತಾಂಶ_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, ಅಂಕೆ_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, ಅಂಕೆ_processes, should_shuffle);

    ಪೂರ್ಣಾಂಕ batches_fit = ಅಂಕೆ_ಟೋಕನುಗಳು / (B * T * ಅಂಕೆ_processes); // number of batches that fit per shard
    ಪೂರ್ಣಾಂಕ BT = B * T;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_epochs = 4;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ e = 0; e < ಅಂಕೆ_epochs; e++) { // epoch
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಂಕ ಆರಂಭ = s * ಅಂಕೆ_ಟೋಕನುಗಳು;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < batches_fit; b++) { // batch
                dataloader_ಮುಂದಿನ_batch(&loader0);
                dataloader_ಮುಂದಿನ_batch(&loader1);
                checkRange(loader0.inputs, ಆರಂಭ, ಆರಂಭ + BT);
                checkRange(loader1.inputs, ಆರಂಭ + BT, ಆರಂಭ + 2*BT);
                checkRange(loader0.targets, ಆರಂಭ + 1, ಆರಂಭ + BT + 1);
                checkRange(loader1.targets, ಆರಂಭ + BT + 1, ಆರಂಭ + 2*BT + 1);
                ಆರಂಭ += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    ಬರೆ("OK\n");
}

ಶೂನ್ಯ test_shuffled(ಶೂನ್ಯ) {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    ಬರೆ("test_shuffled... ");
    ಪೂರ್ಣಾಂಕ B = 4;
    ಪೂರ್ಣಾಂಕ T = 8;
    ಪೂರ್ಣಾಂಕ process_rank = 0;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes = 1;
    ಪೂರ್ಣಾಂಕ should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    ದತ್ತಾಂಶ_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ಅಂಕೆ_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಟೋಕನುಗಳು = ಅಂಕೆ_shards * ಅಂಕೆ_ಟೋಕನುಗಳು;
    ಪೂರ್ಣಾಂಕ *ಅಂಕೆ_ನೋಡಿದ_inputs = (ಪೂರ್ಣಾಂಕ *)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ಒಟ್ಟು_ಟೋಕನುಗಳು, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಪೂರ್ಣಾಂಕ *ಅಂಕೆ_ನೋಡಿದ_targets = (ಪೂರ್ಣಾಂಕ *)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ಒಟ್ಟು_ಟೋಕನುಗಳು, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಪೂರ್ಣಾಂಕ batches_fit = ಅಂಕೆ_ಟೋಕನುಗಳು / (B * T); // number of batches that fit per shard
    ಪೂರ್ಣಾಂಕ BT = B * T;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_epochs = 4;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ e = 0; e < ಅಂಕೆ_epochs; e ++) { // epoch
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಂಕ ಆರಂಭ = s * ಅಂಕೆ_ಟೋಕನುಗಳು;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < batches_fit; b++) { // batch
                dataloader_ಮುಂದಿನ_batch(&loader);
                // count up the tokens we see
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < BT; i++) {
                    ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_token = loader.inputs[i];
                    ಪೂರ್ಣಾಂಕ ಗುರಿ_token = loader.targets[i];
                    ಪ್ರತಿಜ್ಞಾ(ಒಳಹರಿವು_token >= 0 && ಒಳಹರಿವು_token < ಒಟ್ಟು_ಟೋಕನುಗಳು);
                    ಪ್ರತಿಜ್ಞಾ(ಗುರಿ_token >= 0 && ಗುರಿ_token < ಒಟ್ಟು_ಟೋಕನುಗಳು);
                    ಅಂಕೆ_ನೋಡಿದ_inputs[ಒಳಹರಿವು_token]++;
                    ಅಂಕೆ_ನೋಡಿದ_targets[ಗುರಿ_token]++;
                }
                ಆರಂಭ += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ಪೂರ್ಣಾಂಕ ಟೋಕನುಗಳು_fit = batches_fit * BT; // number of tokens that fit per shard
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಅಂಕೆ_shards; s++) {
        ಪೂರ್ಣಾಂಕ ಆರಂಭ = s * ಅಂಕೆ_ಟೋಕನುಗಳು;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_inputs + ಆರಂಭ, ಟೋಕನುಗಳು_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_inputs + ಆರಂಭ + ಟೋಕನುಗಳು_fit, ಅಂಕೆ_ಟೋಕನುಗಳು - ಟೋಕನುಗಳು_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_targets + ಆರಂಭ + 1, ಟೋಕನುಗಳು_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_targets + ಆರಂಭ + 1 + ಟೋಕನುಗಳು_fit,
            (s == (ಅಂಕೆ_shards - 1)) ? ಅಂಕೆ_ಟೋಕನುಗಳು - ಟೋಕನುಗಳು_fit - 1 : ಅಂಕೆ_ಟೋಕನುಗಳು - ಟೋಕನುಗಳು_fit,0);
    }

    dataloader_free(&loader);
    ಬಿಡುಗಡೆ(ಅಂಕೆ_ನೋಡಿದ_inputs);
    ಬಿಡುಗಡೆ(ಅಂಕೆ_ನೋಡಿದ_targets);
    ಬರೆ("OK\n");
}

ಶೂನ್ಯ test_multiprocess_shuffled(ಶೂನ್ಯ) {
    /*
    Tests the DataLoader when using both multiprocess and shuffled:
    - multi-shard
    - multi-process
    - shuffled!
    DataLoaders should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times, over multiple epochs.
    */

    ಬರೆ("test_multiprocess_shuffled... ");
    ಪೂರ್ಣಾಂಕ B = 4;
    ಪೂರ್ಣಾಂಕ T = 8;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes = 2;
    ಪೂರ್ಣಾಂಕ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ದತ್ತಾಂಶ_Loader loaders[ಅಂಕೆ_processes];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, ಅಂಕೆ_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಟೋಕನುಗಳು = ಅಂಕೆ_shards * ಅಂಕೆ_ಟೋಕನುಗಳು;
    ಪೂರ್ಣಾಂಕ *ಅಂಕೆ_ನೋಡಿದ_inputs = (ಪೂರ್ಣಾಂಕ *)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ಒಟ್ಟು_ಟೋಕನುಗಳು, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಪೂರ್ಣಾಂಕ *ಅಂಕೆ_ನೋಡಿದ_targets = (ಪೂರ್ಣಾಂಕ *)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(ಒಟ್ಟು_ಟೋಕನುಗಳು, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    ಪೂರ್ಣಾಂಕ batches_fit = ಅಂಕೆ_ಟೋಕನುಗಳು / (B * T * ಅಂಕೆ_processes); // number of batches that fit per shard
    ಪೂರ್ಣಾಂಕ BT = B * T;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_epochs = 4;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ e = 0; e < ಅಂಕೆ_epochs; e ++) { // epoch
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಅಂಕೆ_shards; s++) { // shard
            ಪೂರ್ಣಾಂಕ ಆರಂಭ = s * ಅಂಕೆ_ಟೋಕನುಗಳು;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < batches_fit; b++) { // batch
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < ಅಂಕೆ_processes; n++) { // dataloader
                    ದತ್ತಾಂಶ_Loader *loader = &loaders[n];
                    dataloader_ಮುಂದಿನ_batch(loader);
                    // count up the tokens we see
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < BT; i++) {
                        ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_token = loader->inputs[i];
                        ಪೂರ್ಣಾಂಕ ಗುರಿ_token = loader->targets[i];
                        ಪ್ರತಿಜ್ಞಾ(ಒಳಹರಿವು_token >= 0 && ಒಳಹರಿವು_token < ಒಟ್ಟು_ಟೋಕನುಗಳು);
                        ಪ್ರತಿಜ್ಞಾ(ಗುರಿ_token >= 0 && ಗುರಿ_token < ಒಟ್ಟು_ಟೋಕನುಗಳು);
                        ಅಂಕೆ_ನೋಡಿದ_inputs[ಒಳಹರಿವು_token]++;
                        ಅಂಕೆ_ನೋಡಿದ_targets[ಗುರಿ_token]++;
                    }
                    ಆರಂಭ += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ಪೂರ್ಣಾಂಕ ಟೋಕನುಗಳು_fit = batches_fit * (B * T * ಅಂಕೆ_processes); // number of tokens that fit per shard
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < ಅಂಕೆ_shards; s++) {
        ಪೂರ್ಣಾಂಕ ಆರಂಭ = s * ಅಂಕೆ_ಟೋಕನುಗಳು; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_inputs + ಆರಂಭ, ಟೋಕನುಗಳು_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_inputs + ಆರಂಭ + ಟೋಕನುಗಳು_fit, ಅಂಕೆ_ಟೋಕನುಗಳು - ಟೋಕನುಗಳು_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_targets + ಆರಂಭ + 1, ಟೋಕನುಗಳು_fit, ಅಂಕೆ_epochs);
        checkEquals(ಅಂಕೆ_ನೋಡಿದ_targets + ಆರಂಭ + 1 + ಟೋಕನುಗಳು_fit,
            (s == (ಅಂಕೆ_shards - 1)) ? ಅಂಕೆ_ಟೋಕನುಗಳು - ಟೋಕನುಗಳು_fit - 1 : ಅಂಕೆ_ಟೋಕನುಗಳು - ಟೋಕನುಗಳು_fit,0);
    }

    // cleanup
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    ಬಿಡುಗಡೆ(ಅಂಕೆ_ನೋಡಿದ_inputs);
    ಬಿಡುಗಡೆ(ಅಂಕೆ_ನೋಡಿದ_targets);
    ಬರೆ("OK\n");
}

ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ(ಶೂನ್ಯ) {

    // generate a few dummy shards of data with incrementing tokens
    ಪೂರ್ಣಾಂಕ ಶೀರ್ಷಿಕೆ[ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ];
    uint16_t ಟೋಕನುಗಳು[ಅಂಕೆ_ಟೋಕನುಗಳು];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ shard_id = 0; shard_id < ಅಂಕೆ_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        ಪೂರ್ಣಾಂಕ token_ಸ್ಥಾನಾಂತರ = shard_id * ಅಂಕೆ_ಟೋಕನುಗಳು;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_ಟೋಕನುಗಳು; i++) {
            ಟೋಕನುಗಳು[i] = token_ಸ್ಥಾನಾಂತರ + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        ಶೀರ್ಷಿಕೆ[0] = 20240520; // magic
        ಶೀರ್ಷಿಕೆ[1] = 1; //ಆವೃತ್ತಿ
        ಶೀರ್ಷಿಕೆ[2] = ಅಂಕೆ_ಟೋಕನುಗಳು; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ, shard_file);
        fwrite(ಟೋಕನುಗಳು, ಗಾತ್ರ(uint16_t), ಅಂಕೆ_ಟೋಕನುಗಳು, shard_file);
        fcloseCheck(shard_file);
        ಬರೆ("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ shard_id = 0; shard_id < ಅಂಕೆ_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    ಹಿಂತಿರುಗಿ EXIT_SUCCESS;
}