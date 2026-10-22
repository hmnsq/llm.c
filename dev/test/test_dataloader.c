/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ضمن <unistd.h>
#ضمن "../../llmc/dataloader.h"

#تعریف_کړه SHARD_NAME_LEN 64
حرف shard_name[SHARD_NAME_LEN];
ثابت صحيح عدد_نښې = 140;
صحيح عدد_shards = 4;

فراغ check_range(ثابت صحيح *نښې, ثابت صحيح پیل, ثابت صحيح پای_ټکی, ثابت حرف *file, صحيح کرښه) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحيح n = پای_ټکی - پیل;
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = نښې[i];
        إذا (token != پیل + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, پیل + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", کرښه);
            وتل(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#تعریف_کړه checkRange(نښې, پیل, پای_ټکی) check_range(نښې, پیل, پای_ټکی, __FILE__, __LINE__)

فراغ check_equals(ثابت صحيح *نښې, ثابت صحيح n, ثابت صحيح expected, ثابت حرف *file, صحيح کرښه) {
    // checks that the tokens[0, ... n] are all equal to expected
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = نښې[i];
        إذا (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", کرښه);
            وتل(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#تعریف_کړه checkEquals(نښې, n, expected) check_equals(نښې, n, expected, __FILE__, __LINE__)

فراغ test_simple(فراغ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    چاپ_کړه("test_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح عدد_processes = 1;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    معلومات_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    صحيح batches_fit = عدد_نښې / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح پیل = s * عدد_نښې;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_بل_batch(&loader);
                checkRange(loader.inputs, پیل, پیل + BT);
                checkRange(loader.targets, پیل + 1, پیل + BT + 1);
                پیل += BT;
            }
        }
    }
    dataloader_free(&loader);
    چاپ_کړه("OK\n");
}

فراغ test_multiprocess_simple(فراغ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    چاپ_کړه("test_multiprocess_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح عدد_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    معلومات_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, عدد_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, عدد_processes, should_shuffle);

    صحيح batches_fit = عدد_نښې / (B * T * عدد_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح پیل = s * عدد_نښې;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_بل_batch(&loader0);
                dataloader_بل_batch(&loader1);
                checkRange(loader0.inputs, پیل, پیل + BT);
                checkRange(loader1.inputs, پیل + BT, پیل + 2*BT);
                checkRange(loader0.targets, پیل + 1, پیل + BT + 1);
                checkRange(loader1.targets, پیل + BT + 1, پیل + 2*BT + 1);
                پیل += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    چاپ_کړه("OK\n");
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
    چاپ_کړه("test_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح عدد_processes = 1;
    صحيح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    معلومات_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح ټول_نښې = عدد_shards * عدد_نښې;
    صحيح *عدد_کتل_شوی_inputs = (صحيح *)حافظه_صفر_ورکړه(ټول_نښې, حجم(صحيح));
    صحيح *عدد_کتل_شوی_targets = (صحيح *)حافظه_صفر_ورکړه(ټول_نښې, حجم(صحيح));
    صحيح batches_fit = عدد_نښې / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح پیل = s * عدد_نښې;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_بل_batch(&loader);
                // count up the tokens we see
                لكل (صحيح i = 0; i < BT; i++) {
                    صحيح ننوت_token = loader.inputs[i];
                    صحيح موخه_token = loader.targets[i];
                    أكد(ننوت_token >= 0 && ننوت_token < ټول_نښې);
                    أكد(موخه_token >= 0 && موخه_token < ټول_نښې);
                    عدد_کتل_شوی_inputs[ننوت_token]++;
                    عدد_کتل_شوی_targets[موخه_token]++;
                }
                پیل += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح نښې_fit = batches_fit * BT; // number of tokens that fit per shard
    لكل (صحيح s = 0; s < عدد_shards; s++) {
        صحيح پیل = s * عدد_نښې;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_کتل_شوی_inputs + پیل, نښې_fit, عدد_epochs);
        checkEquals(عدد_کتل_شوی_inputs + پیل + نښې_fit, عدد_نښې - نښې_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_کتل_شوی_targets + پیل + 1, نښې_fit, عدد_epochs);
        checkEquals(عدد_کتل_شوی_targets + پیل + 1 + نښې_fit,
            (s == (عدد_shards - 1)) ? عدد_نښې - نښې_fit - 1 : عدد_نښې - نښې_fit,0);
    }

    dataloader_free(&loader);
    حرر(عدد_کتل_شوی_inputs);
    حرر(عدد_کتل_شوی_targets);
    چاپ_کړه("OK\n");
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

    چاپ_کړه("test_multiprocess_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    ثابت صحيح عدد_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    معلومات_Loader loaders[عدد_processes];
    لكل (صحيح i = 0; i < عدد_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, عدد_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح ټول_نښې = عدد_shards * عدد_نښې;
    صحيح *عدد_کتل_شوی_inputs = (صحيح *)حافظه_صفر_ورکړه(ټول_نښې, حجم(صحيح));
    صحيح *عدد_کتل_شوی_targets = (صحيح *)حافظه_صفر_ورکړه(ټول_نښې, حجم(صحيح));
    صحيح batches_fit = عدد_نښې / (B * T * عدد_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح پیل = s * عدد_نښې;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                لكل (صحيح n = 0; n < عدد_processes; n++) { // dataloader
                    معلومات_Loader *loader = &loaders[n];
                    dataloader_بل_batch(loader);
                    // count up the tokens we see
                    لكل (صحيح i = 0; i < BT; i++) {
                        صحيح ننوت_token = loader->inputs[i];
                        صحيح موخه_token = loader->targets[i];
                        أكد(ننوت_token >= 0 && ننوت_token < ټول_نښې);
                        أكد(موخه_token >= 0 && موخه_token < ټول_نښې);
                        عدد_کتل_شوی_inputs[ننوت_token]++;
                        عدد_کتل_شوی_targets[موخه_token]++;
                    }
                    پیل += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح نښې_fit = batches_fit * (B * T * عدد_processes); // number of tokens that fit per shard
    لكل (صحيح s = 0; s < عدد_shards; s++) {
        صحيح پیل = s * عدد_نښې; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_کتل_شوی_inputs + پیل, نښې_fit, عدد_epochs);
        checkEquals(عدد_کتل_شوی_inputs + پیل + نښې_fit, عدد_نښې - نښې_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_کتل_شوی_targets + پیل + 1, نښې_fit, عدد_epochs);
        checkEquals(عدد_کتل_شوی_targets + پیل + 1 + نښې_fit,
            (s == (عدد_shards - 1)) ? عدد_نښې - نښې_fit - 1 : عدد_نښې - نښې_fit,0);
    }

    // cleanup
    لكل (صحيح i = 0; i < عدد_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    حرر(عدد_کتل_شوی_inputs);
    حرر(عدد_کتل_شوی_targets);
    چاپ_کړه("OK\n");
}

صحيح اصلي(فراغ) {

    // generate a few dummy shards of data with incrementing tokens
    صحيح سرلیک[سرلیک_اندازه];
    uint16_t نښې[عدد_نښې];
    لكل (صحيح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحيح token_دننه_واټن = shard_id * عدد_نښې;
        لكل (صحيح i = 0; i < عدد_نښې; i++) {
            نښې[i] = token_دننه_واټن + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        سرلیک[0] = 20240520; // magic
        سرلیک[1] = 1; //نسخه
        سرلیک[2] = عدد_نښې; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(سرلیک, حجم(صحيح), سرلیک_اندازه, shard_file);
        fwrite(نښې, حجم(uint16_t), عدد_نښې, shard_file);
        fcloseCheck(shard_file);
        چاپ_کړه("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    لكل (صحيح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    أرجع EXIT_SUCCESS;
}