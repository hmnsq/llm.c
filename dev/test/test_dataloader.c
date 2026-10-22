/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ضمن <unistd.h>
#ضمن "../../llmc/dataloader.h"

#تعریف_کن SHARD_NAME_LEN 64
حرف shard_name[SHARD_NAME_LEN];
ثابت صحيح رقم_نشانهها = 140;
صحيح رقم_shards = 4;

فراغ check_range(ثابت صحيح *نشانهها, ثابت صحيح شروع, ثابت صحيح انتها, ثابت حرف *file, صحيح سطر) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحيح n = انتها - شروع;
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = نشانهها[i];
        إذا (token != شروع + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, شروع + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            خروج(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#تعریف_کن checkRange(نشانهها, شروع, انتها) check_range(نشانهها, شروع, انتها, __FILE__, __LINE__)

فراغ check_equals(ثابت صحيح *نشانهها, ثابت صحيح n, ثابت صحيح expected, ثابت حرف *file, صحيح سطر) {
    // checks that the tokens[0, ... n] are all equal to expected
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = نشانهها[i];
        إذا (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            خروج(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#تعریف_کن checkEquals(نشانهها, n, expected) check_equals(نشانهها, n, expected, __FILE__, __LINE__)

فراغ test_simple(فراغ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    چاپ_کن("test_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح رقم_processes = 1;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    داده_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, رقم_processes, should_shuffle);

    صحيح batches_fit = رقم_نشانهها / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح رقم_epochs = 4;
    لكل (صحيح e = 0; e < رقم_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < رقم_shards; s++) { // shard
            صحيح شروع = s * رقم_نشانهها;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_بعدی_batch(&loader);
                checkRange(loader.inputs, شروع, شروع + BT);
                checkRange(loader.targets, شروع + 1, شروع + BT + 1);
                شروع += BT;
            }
        }
    }
    dataloader_free(&loader);
    چاپ_کن("OK\n");
}

فراغ test_multiprocess_simple(فراغ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    چاپ_کن("test_multiprocess_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح رقم_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    داده_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, رقم_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, رقم_processes, should_shuffle);

    صحيح batches_fit = رقم_نشانهها / (B * T * رقم_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح رقم_epochs = 4;
    لكل (صحيح e = 0; e < رقم_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < رقم_shards; s++) { // shard
            صحيح شروع = s * رقم_نشانهها;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_بعدی_batch(&loader0);
                dataloader_بعدی_batch(&loader1);
                checkRange(loader0.inputs, شروع, شروع + BT);
                checkRange(loader1.inputs, شروع + BT, شروع + 2*BT);
                checkRange(loader0.targets, شروع + 1, شروع + BT + 1);
                checkRange(loader1.targets, شروع + BT + 1, شروع + 2*BT + 1);
                شروع += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    چاپ_کن("OK\n");
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
    چاپ_کن("test_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح رقم_processes = 1;
    صحيح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    داده_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, رقم_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح مجموع_نشانهها = رقم_shards * رقم_نشانهها;
    صحيح *رقم_دیده_inputs = (صحيح *)حافظه_صفر_بده(مجموع_نشانهها, حجم(صحيح));
    صحيح *رقم_دیده_targets = (صحيح *)حافظه_صفر_بده(مجموع_نشانهها, حجم(صحيح));
    صحيح batches_fit = رقم_نشانهها / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح رقم_epochs = 4;
    لكل (صحيح e = 0; e < رقم_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < رقم_shards; s++) { // shard
            صحيح شروع = s * رقم_نشانهها;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_بعدی_batch(&loader);
                // count up the tokens we see
                لكل (صحيح i = 0; i < BT; i++) {
                    صحيح درون_داد_token = loader.inputs[i];
                    صحيح هدف_token = loader.targets[i];
                    أكد(درون_داد_token >= 0 && درون_داد_token < مجموع_نشانهها);
                    أكد(هدف_token >= 0 && هدف_token < مجموع_نشانهها);
                    رقم_دیده_inputs[درون_داد_token]++;
                    رقم_دیده_targets[هدف_token]++;
                }
                شروع += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح نشانهها_fit = batches_fit * BT; // number of tokens that fit per shard
    لكل (صحيح s = 0; s < رقم_shards; s++) {
        صحيح شروع = s * رقم_نشانهها;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(رقم_دیده_inputs + شروع, نشانهها_fit, رقم_epochs);
        checkEquals(رقم_دیده_inputs + شروع + نشانهها_fit, رقم_نشانهها - نشانهها_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(رقم_دیده_targets + شروع + 1, نشانهها_fit, رقم_epochs);
        checkEquals(رقم_دیده_targets + شروع + 1 + نشانهها_fit,
            (s == (رقم_shards - 1)) ? رقم_نشانهها - نشانهها_fit - 1 : رقم_نشانهها - نشانهها_fit,0);
    }

    dataloader_free(&loader);
    حرر(رقم_دیده_inputs);
    حرر(رقم_دیده_targets);
    چاپ_کن("OK\n");
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

    چاپ_کن("test_multiprocess_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    ثابت صحيح رقم_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    داده_Loader loaders[رقم_processes];
    لكل (صحيح i = 0; i < رقم_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, رقم_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح مجموع_نشانهها = رقم_shards * رقم_نشانهها;
    صحيح *رقم_دیده_inputs = (صحيح *)حافظه_صفر_بده(مجموع_نشانهها, حجم(صحيح));
    صحيح *رقم_دیده_targets = (صحيح *)حافظه_صفر_بده(مجموع_نشانهها, حجم(صحيح));
    صحيح batches_fit = رقم_نشانهها / (B * T * رقم_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح رقم_epochs = 4;
    لكل (صحيح e = 0; e < رقم_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < رقم_shards; s++) { // shard
            صحيح شروع = s * رقم_نشانهها;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                لكل (صحيح n = 0; n < رقم_processes; n++) { // dataloader
                    داده_Loader *loader = &loaders[n];
                    dataloader_بعدی_batch(loader);
                    // count up the tokens we see
                    لكل (صحيح i = 0; i < BT; i++) {
                        صحيح درون_داد_token = loader->inputs[i];
                        صحيح هدف_token = loader->targets[i];
                        أكد(درون_داد_token >= 0 && درون_داد_token < مجموع_نشانهها);
                        أكد(هدف_token >= 0 && هدف_token < مجموع_نشانهها);
                        رقم_دیده_inputs[درون_داد_token]++;
                        رقم_دیده_targets[هدف_token]++;
                    }
                    شروع += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح نشانهها_fit = batches_fit * (B * T * رقم_processes); // number of tokens that fit per shard
    لكل (صحيح s = 0; s < رقم_shards; s++) {
        صحيح شروع = s * رقم_نشانهها; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(رقم_دیده_inputs + شروع, نشانهها_fit, رقم_epochs);
        checkEquals(رقم_دیده_inputs + شروع + نشانهها_fit, رقم_نشانهها - نشانهها_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(رقم_دیده_targets + شروع + 1, نشانهها_fit, رقم_epochs);
        checkEquals(رقم_دیده_targets + شروع + 1 + نشانهها_fit,
            (s == (رقم_shards - 1)) ? رقم_نشانهها - نشانهها_fit - 1 : رقم_نشانهها - نشانهها_fit,0);
    }

    // cleanup
    لكل (صحيح i = 0; i < رقم_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    حرر(رقم_دیده_inputs);
    حرر(رقم_دیده_targets);
    چاپ_کن("OK\n");
}

صحيح اصلی(فراغ) {

    // generate a few dummy shards of data with incrementing tokens
    صحيح سرآیند[سرآیند_اندازه];
    uint16_t نشانهها[رقم_نشانهها];
    لكل (صحيح shard_id = 0; shard_id < رقم_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحيح token_جابه_جایی = shard_id * رقم_نشانهها;
        لكل (صحيح i = 0; i < رقم_نشانهها; i++) {
            نشانهها[i] = token_جابه_جایی + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        سرآیند[0] = 20240520; // magic
        سرآیند[1] = 1; //نسخه
        سرآیند[2] = رقم_نشانهها; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(سرآیند, حجم(صحيح), سرآیند_اندازه, shard_file);
        fwrite(نشانهها, حجم(uint16_t), رقم_نشانهها, shard_file);
        fcloseCheck(shard_file);
        چاپ_کن("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    لكل (صحيح shard_id = 0; shard_id < رقم_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    أرجع EXIT_SUCCESS;
}