/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ضمن <unistd.h>
#ضمن "../../llmc/dataloader.h"

#وصف_ڪريو SHARD_NAME_LEN 64
حرف shard_name[SHARD_NAME_LEN];
ثابت صحيح انگ_ٽوڪن = 140;
صحيح انگ_shards = 4;

فراغ check_range(ثابت صحيح *ٽوڪن, ثابت صحيح شروعات, ثابت صحيح پڇاڙي, ثابت حرف *file, صحيح سٽ) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحيح n = پڇاڙي - شروعات;
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = ٽوڪن[i];
        إذا (token != شروعات + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, شروعات + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سٽ);
            ٻاهر_نڪرو(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#وصف_ڪريو checkRange(ٽوڪن, شروعات, پڇاڙي) check_range(ٽوڪن, شروعات, پڇاڙي, __FILE__, __LINE__)

فراغ check_equals(ثابت صحيح *ٽوڪن, ثابت صحيح n, ثابت صحيح expected, ثابت حرف *file, صحيح سٽ) {
    // checks that the tokens[0, ... n] are all equal to expected
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = ٽوڪن[i];
        إذا (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سٽ);
            ٻاهر_نڪرو(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#وصف_ڪريو checkEquals(ٽوڪن, n, expected) check_equals(ٽوڪن, n, expected, __FILE__, __LINE__)

فراغ test_simple(فراغ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    لکو("test_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح انگ_processes = 1;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    معلومات_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, انگ_processes, should_shuffle);

    صحيح batches_fit = انگ_ٽوڪن / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح انگ_epochs = 4;
    لكل (صحيح e = 0; e < انگ_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < انگ_shards; s++) { // shard
            صحيح شروعات = s * انگ_ٽوڪن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_ايندڙ_batch(&loader);
                checkRange(loader.inputs, شروعات, شروعات + BT);
                checkRange(loader.targets, شروعات + 1, شروعات + BT + 1);
                شروعات += BT;
            }
        }
    }
    dataloader_free(&loader);
    لکو("OK\n");
}

فراغ test_multiprocess_simple(فراغ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    لکو("test_multiprocess_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح انگ_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    معلومات_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, انگ_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, انگ_processes, should_shuffle);

    صحيح batches_fit = انگ_ٽوڪن / (B * T * انگ_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح انگ_epochs = 4;
    لكل (صحيح e = 0; e < انگ_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < انگ_shards; s++) { // shard
            صحيح شروعات = s * انگ_ٽوڪن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_ايندڙ_batch(&loader0);
                dataloader_ايندڙ_batch(&loader1);
                checkRange(loader0.inputs, شروعات, شروعات + BT);
                checkRange(loader1.inputs, شروعات + BT, شروعات + 2*BT);
                checkRange(loader0.targets, شروعات + 1, شروعات + BT + 1);
                checkRange(loader1.targets, شروعات + BT + 1, شروعات + 2*BT + 1);
                شروعات += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    لکو("OK\n");
}

فراغ test_shuffled(فراغ) {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    لکو("test_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح انگ_processes = 1;
    صحيح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    معلومات_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, انگ_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح ڪل_ٽوڪن = انگ_shards * انگ_ٽوڪن;
    صحيح *انگ_ڏٺل_inputs = (صحيح *)صفر_يادگيري_ڏيو(ڪل_ٽوڪن, حجم(صحيح));
    صحيح *انگ_ڏٺل_targets = (صحيح *)صفر_يادگيري_ڏيو(ڪل_ٽوڪن, حجم(صحيح));
    صحيح batches_fit = انگ_ٽوڪن / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح انگ_epochs = 4;
    لكل (صحيح e = 0; e < انگ_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < انگ_shards; s++) { // shard
            صحيح شروعات = s * انگ_ٽوڪن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_ايندڙ_batch(&loader);
                // count up the tokens we see
                لكل (صحيح i = 0; i < BT; i++) {
                    صحيح داخلا_token = loader.inputs[i];
                    صحيح هدف_token = loader.targets[i];
                    أكد(داخلا_token >= 0 && داخلا_token < ڪل_ٽوڪن);
                    أكد(هدف_token >= 0 && هدف_token < ڪل_ٽوڪن);
                    انگ_ڏٺل_inputs[داخلا_token]++;
                    انگ_ڏٺل_targets[هدف_token]++;
                }
                شروعات += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح ٽوڪن_fit = batches_fit * BT; // number of tokens that fit per shard
    لكل (صحيح s = 0; s < انگ_shards; s++) {
        صحيح شروعات = s * انگ_ٽوڪن;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(انگ_ڏٺل_inputs + شروعات, ٽوڪن_fit, انگ_epochs);
        checkEquals(انگ_ڏٺل_inputs + شروعات + ٽوڪن_fit, انگ_ٽوڪن - ٽوڪن_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(انگ_ڏٺل_targets + شروعات + 1, ٽوڪن_fit, انگ_epochs);
        checkEquals(انگ_ڏٺل_targets + شروعات + 1 + ٽوڪن_fit,
            (s == (انگ_shards - 1)) ? انگ_ٽوڪن - ٽوڪن_fit - 1 : انگ_ٽوڪن - ٽوڪن_fit,0);
    }

    dataloader_free(&loader);
    حرر(انگ_ڏٺل_inputs);
    حرر(انگ_ڏٺل_targets);
    لکو("OK\n");
}

فراغ test_multiprocess_shuffled(فراغ) {
    /*
    Tests the DataLoader when using both multiprocess and shuffled:
    - multi-shard
    - multi-process
    - shuffled!
    DataLoaders should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times, over multiple epochs.
    */

    لکو("test_multiprocess_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    ثابت صحيح انگ_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    معلومات_Loader loaders[انگ_processes];
    لكل (صحيح i = 0; i < انگ_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, انگ_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح ڪل_ٽوڪن = انگ_shards * انگ_ٽوڪن;
    صحيح *انگ_ڏٺل_inputs = (صحيح *)صفر_يادگيري_ڏيو(ڪل_ٽوڪن, حجم(صحيح));
    صحيح *انگ_ڏٺل_targets = (صحيح *)صفر_يادگيري_ڏيو(ڪل_ٽوڪن, حجم(صحيح));
    صحيح batches_fit = انگ_ٽوڪن / (B * T * انگ_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح انگ_epochs = 4;
    لكل (صحيح e = 0; e < انگ_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < انگ_shards; s++) { // shard
            صحيح شروعات = s * انگ_ٽوڪن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                لكل (صحيح n = 0; n < انگ_processes; n++) { // dataloader
                    معلومات_Loader *loader = &loaders[n];
                    dataloader_ايندڙ_batch(loader);
                    // count up the tokens we see
                    لكل (صحيح i = 0; i < BT; i++) {
                        صحيح داخلا_token = loader->inputs[i];
                        صحيح هدف_token = loader->targets[i];
                        أكد(داخلا_token >= 0 && داخلا_token < ڪل_ٽوڪن);
                        أكد(هدف_token >= 0 && هدف_token < ڪل_ٽوڪن);
                        انگ_ڏٺل_inputs[داخلا_token]++;
                        انگ_ڏٺل_targets[هدف_token]++;
                    }
                    شروعات += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح ٽوڪن_fit = batches_fit * (B * T * انگ_processes); // number of tokens that fit per shard
    لكل (صحيح s = 0; s < انگ_shards; s++) {
        صحيح شروعات = s * انگ_ٽوڪن; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(انگ_ڏٺل_inputs + شروعات, ٽوڪن_fit, انگ_epochs);
        checkEquals(انگ_ڏٺل_inputs + شروعات + ٽوڪن_fit, انگ_ٽوڪن - ٽوڪن_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(انگ_ڏٺل_targets + شروعات + 1, ٽوڪن_fit, انگ_epochs);
        checkEquals(انگ_ڏٺل_targets + شروعات + 1 + ٽوڪن_fit,
            (s == (انگ_shards - 1)) ? انگ_ٽوڪن - ٽوڪن_fit - 1 : انگ_ٽوڪن - ٽوڪن_fit,0);
    }

    // cleanup
    لكل (صحيح i = 0; i < انگ_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    حرر(انگ_ڏٺل_inputs);
    حرر(انگ_ڏٺل_targets);
    لکو("OK\n");
}

صحيح مکيه(فراغ) {

    // generate a few dummy shards of data with incrementing tokens
    صحيح سرو[سرو_سائيز];
    uint16_t ٽوڪن[انگ_ٽوڪن];
    لكل (صحيح shard_id = 0; shard_id < انگ_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحيح token_وٿي = shard_id * انگ_ٽوڪن;
        لكل (صحيح i = 0; i < انگ_ٽوڪن; i++) {
            ٽوڪن[i] = token_وٿي + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        سرو[0] = 20240520; // magic
        سرو[1] = 1; //نسخو
        سرو[2] = انگ_ٽوڪن; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(سرو, حجم(صحيح), سرو_سائيز, shard_file);
        fwrite(ٽوڪن, حجم(uint16_t), انگ_ٽوڪن, shard_file);
        fcloseCheck(shard_file);
        لکو("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    لكل (صحيح shard_id = 0; shard_id < انگ_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    أرجع EXIT_SUCCESS;
}