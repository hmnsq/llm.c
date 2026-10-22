/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "../../llmc/dataloader.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SHARD_NAME_LEN 64
ਅੱਖਰ shard_name[SHARD_NAME_LEN];
ਅਚਲ ਪੂਰਨ ਅੰਕ_ਟੋਕਨ = 140;
ਪੂਰਨ ਅੰਕ_shards = 4;

ਖਾਲੀ check_range(ਅਚਲ ਪੂਰਨ *ਟੋਕਨ, ਅਚਲ ਪੂਰਨ ਸ਼ੁਰੂ, ਅਚਲ ਪੂਰਨ ਸਿਰਾ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    ਪੂਰਨ n = ਸਿਰਾ - ਸ਼ੁਰੂ;
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ਪੂਰਨ token = ਟੋਕਨ[i];
        ਜੇ (token != ਸ਼ੁਰੂ + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, ਸ਼ੁਰੂ + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ਸਤਰ);
            ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ checkRange(ਟੋਕਨ, ਸ਼ੁਰੂ, ਸਿਰਾ) check_range(ਟੋਕਨ, ਸ਼ੁਰੂ, ਸਿਰਾ, __FILE__, __LINE__)

ਖਾਲੀ check_equals(ਅਚਲ ਪੂਰਨ *ਟੋਕਨ, ਅਚਲ ਪੂਰਨ n, ਅਚਲ ਪੂਰਨ expected, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    // checks that the tokens[0, ... n] are all equal to expected
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        ਪੂਰਨ token = ਟੋਕਨ[i];
        ਜੇ (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", ਸਤਰ);
            ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ checkEquals(ਟੋਕਨ, n, expected) check_equals(ਟੋਕਨ, n, expected, __FILE__, __LINE__)

ਖਾਲੀ test_simple(ਖਾਲੀ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    ਲਿਖੋ("test_simple... ");
    ਪੂਰਨ B = 4;
    ਪੂਰਨ T = 8;
    ਪੂਰਨ process_rank = 0;
    ਪੂਰਨ ਅੰਕ_processes = 1;
    ਪੂਰਨ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ਅੰਕੜੇ_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ਅੰਕ_processes, should_shuffle);

    ਪੂਰਨ batches_fit = ਅੰਕ_ਟੋਕਨ / (B * T); // number of batches that fit per shard
    ਪੂਰਨ BT = B * T;
    ਪੂਰਨ ਅੰਕ_epochs = 4;
    ਲਈ (ਪੂਰਨ e = 0; e < ਅੰਕ_epochs; e++) { // epoch
        ਲਈ (ਪੂਰਨ s = 0; s < ਅੰਕ_shards; s++) { // shard
            ਪੂਰਨ ਸ਼ੁਰੂ = s * ਅੰਕ_ਟੋਕਨ;
            ਲਈ (ਪੂਰਨ b = 0; b < batches_fit; b++) { // batch
                dataloader_ਅਗਲਾ_batch(&loader);
                checkRange(loader.inputs, ਸ਼ੁਰੂ, ਸ਼ੁਰੂ + BT);
                checkRange(loader.targets, ਸ਼ੁਰੂ + 1, ਸ਼ੁਰੂ + BT + 1);
                ਸ਼ੁਰੂ += BT;
            }
        }
    }
    dataloader_free(&loader);
    ਲਿਖੋ("OK\n");
}

ਖਾਲੀ test_multiprocess_simple(ਖਾਲੀ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    ਲਿਖੋ("test_multiprocess_simple... ");
    ਪੂਰਨ B = 4;
    ਪੂਰਨ T = 8;
    ਪੂਰਨ ਅੰਕ_processes = 2;
    ਪੂਰਨ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ਅੰਕੜੇ_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, ਅੰਕ_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, ਅੰਕ_processes, should_shuffle);

    ਪੂਰਨ batches_fit = ਅੰਕ_ਟੋਕਨ / (B * T * ਅੰਕ_processes); // number of batches that fit per shard
    ਪੂਰਨ BT = B * T;
    ਪੂਰਨ ਅੰਕ_epochs = 4;
    ਲਈ (ਪੂਰਨ e = 0; e < ਅੰਕ_epochs; e++) { // epoch
        ਲਈ (ਪੂਰਨ s = 0; s < ਅੰਕ_shards; s++) { // shard
            ਪੂਰਨ ਸ਼ੁਰੂ = s * ਅੰਕ_ਟੋਕਨ;
            ਲਈ (ਪੂਰਨ b = 0; b < batches_fit; b++) { // batch
                dataloader_ਅਗਲਾ_batch(&loader0);
                dataloader_ਅਗਲਾ_batch(&loader1);
                checkRange(loader0.inputs, ਸ਼ੁਰੂ, ਸ਼ੁਰੂ + BT);
                checkRange(loader1.inputs, ਸ਼ੁਰੂ + BT, ਸ਼ੁਰੂ + 2*BT);
                checkRange(loader0.targets, ਸ਼ੁਰੂ + 1, ਸ਼ੁਰੂ + BT + 1);
                checkRange(loader1.targets, ਸ਼ੁਰੂ + BT + 1, ਸ਼ੁਰੂ + 2*BT + 1);
                ਸ਼ੁਰੂ += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    ਲਿਖੋ("OK\n");
}

ਖਾਲੀ test_shuffled(ਖਾਲੀ) {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    ਲਿਖੋ("test_shuffled... ");
    ਪੂਰਨ B = 4;
    ਪੂਰਨ T = 8;
    ਪੂਰਨ process_rank = 0;
    ਪੂਰਨ ਅੰਕ_processes = 1;
    ਪੂਰਨ should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    ਅੰਕੜੇ_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, ਅੰਕ_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    ਪੂਰਨ ਕੁੱਲ_ਟੋਕਨ = ਅੰਕ_shards * ਅੰਕ_ਟੋਕਨ;
    ਪੂਰਨ *ਅੰਕ_ਵੇਖਿਆ_inputs = (ਪੂਰਨ *)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਕੁੱਲ_ਟੋਕਨ, ਆਕਾਰ(ਪੂਰਨ));
    ਪੂਰਨ *ਅੰਕ_ਵੇਖਿਆ_targets = (ਪੂਰਨ *)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਕੁੱਲ_ਟੋਕਨ, ਆਕਾਰ(ਪੂਰਨ));
    ਪੂਰਨ batches_fit = ਅੰਕ_ਟੋਕਨ / (B * T); // number of batches that fit per shard
    ਪੂਰਨ BT = B * T;
    ਪੂਰਨ ਅੰਕ_epochs = 4;
    ਲਈ (ਪੂਰਨ e = 0; e < ਅੰਕ_epochs; e ++) { // epoch
        ਲਈ (ਪੂਰਨ s = 0; s < ਅੰਕ_shards; s++) { // shard
            ਪੂਰਨ ਸ਼ੁਰੂ = s * ਅੰਕ_ਟੋਕਨ;
            ਲਈ (ਪੂਰਨ b = 0; b < batches_fit; b++) { // batch
                dataloader_ਅਗਲਾ_batch(&loader);
                // count up the tokens we see
                ਲਈ (ਪੂਰਨ i = 0; i < BT; i++) {
                    ਪੂਰਨ ਨਿਵੇਸ਼_token = loader.inputs[i];
                    ਪੂਰਨ ਨਿਸ਼ਾਨਾ_token = loader.targets[i];
                    ਪ੍ਰਤਿਜ੍ਞਾ(ਨਿਵੇਸ਼_token >= 0 && ਨਿਵੇਸ਼_token < ਕੁੱਲ_ਟੋਕਨ);
                    ਪ੍ਰਤਿਜ੍ਞਾ(ਨਿਸ਼ਾਨਾ_token >= 0 && ਨਿਸ਼ਾਨਾ_token < ਕੁੱਲ_ਟੋਕਨ);
                    ਅੰਕ_ਵੇਖਿਆ_inputs[ਨਿਵੇਸ਼_token]++;
                    ਅੰਕ_ਵੇਖਿਆ_targets[ਨਿਸ਼ਾਨਾ_token]++;
                }
                ਸ਼ੁਰੂ += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ਪੂਰਨ ਟੋਕਨ_fit = batches_fit * BT; // number of tokens that fit per shard
    ਲਈ (ਪੂਰਨ s = 0; s < ਅੰਕ_shards; s++) {
        ਪੂਰਨ ਸ਼ੁਰੂ = s * ਅੰਕ_ਟੋਕਨ;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_inputs + ਸ਼ੁਰੂ, ਟੋਕਨ_fit, ਅੰਕ_epochs);
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_inputs + ਸ਼ੁਰੂ + ਟੋਕਨ_fit, ਅੰਕ_ਟੋਕਨ - ਟੋਕਨ_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_targets + ਸ਼ੁਰੂ + 1, ਟੋਕਨ_fit, ਅੰਕ_epochs);
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_targets + ਸ਼ੁਰੂ + 1 + ਟੋਕਨ_fit,
            (s == (ਅੰਕ_shards - 1)) ? ਅੰਕ_ਟੋਕਨ - ਟੋਕਨ_fit - 1 : ਅੰਕ_ਟੋਕਨ - ਟੋਕਨ_fit,0);
    }

    dataloader_free(&loader);
    ਮੁਕਤ(ਅੰਕ_ਵੇਖਿਆ_inputs);
    ਮੁਕਤ(ਅੰਕ_ਵੇਖਿਆ_targets);
    ਲਿਖੋ("OK\n");
}

ਖਾਲੀ test_multiprocess_shuffled(ਖਾਲੀ) {
    /*
    Tests the DataLoader when using both multiprocess and shuffled:
    - multi-shard
    - multi-process
    - shuffled!
    DataLoaders should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times, over multiple epochs.
    */

    ਲਿਖੋ("test_multiprocess_shuffled... ");
    ਪੂਰਨ B = 4;
    ਪੂਰਨ T = 8;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_processes = 2;
    ਪੂਰਨ should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ਅੰਕੜੇ_Loader loaders[ਅੰਕ_processes];
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, ਅੰਕ_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    ਪੂਰਨ ਕੁੱਲ_ਟੋਕਨ = ਅੰਕ_shards * ਅੰਕ_ਟੋਕਨ;
    ਪੂਰਨ *ਅੰਕ_ਵੇਖਿਆ_inputs = (ਪੂਰਨ *)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਕੁੱਲ_ਟੋਕਨ, ਆਕਾਰ(ਪੂਰਨ));
    ਪੂਰਨ *ਅੰਕ_ਵੇਖਿਆ_targets = (ਪੂਰਨ *)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਕੁੱਲ_ਟੋਕਨ, ਆਕਾਰ(ਪੂਰਨ));
    ਪੂਰਨ batches_fit = ਅੰਕ_ਟੋਕਨ / (B * T * ਅੰਕ_processes); // number of batches that fit per shard
    ਪੂਰਨ BT = B * T;
    ਪੂਰਨ ਅੰਕ_epochs = 4;
    ਲਈ (ਪੂਰਨ e = 0; e < ਅੰਕ_epochs; e ++) { // epoch
        ਲਈ (ਪੂਰਨ s = 0; s < ਅੰਕ_shards; s++) { // shard
            ਪੂਰਨ ਸ਼ੁਰੂ = s * ਅੰਕ_ਟੋਕਨ;
            ਲਈ (ਪੂਰਨ b = 0; b < batches_fit; b++) { // batch
                ਲਈ (ਪੂਰਨ n = 0; n < ਅੰਕ_processes; n++) { // dataloader
                    ਅੰਕੜੇ_Loader *loader = &loaders[n];
                    dataloader_ਅਗਲਾ_batch(loader);
                    // count up the tokens we see
                    ਲਈ (ਪੂਰਨ i = 0; i < BT; i++) {
                        ਪੂਰਨ ਨਿਵੇਸ਼_token = loader->inputs[i];
                        ਪੂਰਨ ਨਿਸ਼ਾਨਾ_token = loader->targets[i];
                        ਪ੍ਰਤਿਜ੍ਞਾ(ਨਿਵੇਸ਼_token >= 0 && ਨਿਵੇਸ਼_token < ਕੁੱਲ_ਟੋਕਨ);
                        ਪ੍ਰਤਿਜ੍ਞਾ(ਨਿਸ਼ਾਨਾ_token >= 0 && ਨਿਸ਼ਾਨਾ_token < ਕੁੱਲ_ਟੋਕਨ);
                        ਅੰਕ_ਵੇਖਿਆ_inputs[ਨਿਵੇਸ਼_token]++;
                        ਅੰਕ_ਵੇਖਿਆ_targets[ਨਿਸ਼ਾਨਾ_token]++;
                    }
                    ਸ਼ੁਰੂ += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    ਪੂਰਨ ਟੋਕਨ_fit = batches_fit * (B * T * ਅੰਕ_processes); // number of tokens that fit per shard
    ਲਈ (ਪੂਰਨ s = 0; s < ਅੰਕ_shards; s++) {
        ਪੂਰਨ ਸ਼ੁਰੂ = s * ਅੰਕ_ਟੋਕਨ; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_inputs + ਸ਼ੁਰੂ, ਟੋਕਨ_fit, ਅੰਕ_epochs);
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_inputs + ਸ਼ੁਰੂ + ਟੋਕਨ_fit, ਅੰਕ_ਟੋਕਨ - ਟੋਕਨ_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_targets + ਸ਼ੁਰੂ + 1, ਟੋਕਨ_fit, ਅੰਕ_epochs);
        checkEquals(ਅੰਕ_ਵੇਖਿਆ_targets + ਸ਼ੁਰੂ + 1 + ਟੋਕਨ_fit,
            (s == (ਅੰਕ_shards - 1)) ? ਅੰਕ_ਟੋਕਨ - ਟੋਕਨ_fit - 1 : ਅੰਕ_ਟੋਕਨ - ਟੋਕਨ_fit,0);
    }

    // cleanup
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    ਮੁਕਤ(ਅੰਕ_ਵੇਖਿਆ_inputs);
    ਮੁਕਤ(ਅੰਕ_ਵੇਖਿਆ_targets);
    ਲਿਖੋ("OK\n");
}

ਪੂਰਨ ਮੁੱਖ(ਖਾਲੀ) {

    // generate a few dummy shards of data with incrementing tokens
    ਪੂਰਨ ਸਿਰਲੇਖ[ਸਿਰਲੇਖ_ਮਾਪ];
    uint16_t ਟੋਕਨ[ਅੰਕ_ਟੋਕਨ];
    ਲਈ (ਪੂਰਨ shard_id = 0; shard_id < ਅੰਕ_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        ਪੂਰਨ token_ਵਿੱਥ = shard_id * ਅੰਕ_ਟੋਕਨ;
        ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_ਟੋਕਨ; i++) {
            ਟੋਕਨ[i] = token_ਵਿੱਥ + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        ਸਿਰਲੇਖ[0] = 20240520; // magic
        ਸਿਰਲੇਖ[1] = 1; //ਸੰਸਕਰਣ
        ਸਿਰਲੇਖ[2] = ਅੰਕ_ਟੋਕਨ; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(ਸਿਰਲੇਖ, ਆਕਾਰ(ਪੂਰਨ), ਸਿਰਲੇਖ_ਮਾਪ, shard_file);
        fwrite(ਟੋਕਨ, ਆਕਾਰ(uint16_t), ਅੰਕ_ਟੋਕਨ, shard_file);
        fcloseCheck(shard_file);
        ਲਿਖੋ("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    ਲਈ (ਪੂਰਨ shard_id = 0; shard_id < ਅੰਕ_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    ਵਾਪਸ EXIT_SUCCESS;
}