/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#ضمن <unistd.h>
#ضمن "../../llmc/dataloader.h"

#تعریف_کرو SHARD_NAME_LEN 64
حرف shard_name[SHARD_NAME_LEN];
ثابت صحيح عدد_ٹوکن = 140;
صحيح عدد_shards = 4;

فراغ check_range(ثابت صحيح *ٹوکن, ثابت صحيح شروع, ثابت صحيح سرا, ثابت حرف *file, صحيح سطر) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحيح n = سرا - شروع;
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = ٹوکن[i];
        إذا (token != شروع + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, شروع + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            باہر_نکلو(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#تعریف_کرو checkRange(ٹوکن, شروع, سرا) check_range(ٹوکن, شروع, سرا, __FILE__, __LINE__)

فراغ check_equals(ثابت صحيح *ٹوکن, ثابت صحيح n, ثابت صحيح expected, ثابت حرف *file, صحيح سطر) {
    // checks that the tokens[0, ... n] are all equal to expected
    لكل (صحيح i = 0; i < n; i++) {
        صحيح token = ٹوکن[i];
        إذا (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            باہر_نکلو(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#تعریف_کرو checkEquals(ٹوکن, n, expected) check_equals(ٹوکن, n, expected, __FILE__, __LINE__)

فراغ test_simple(فراغ) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    لکھو("test_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح عدد_processes = 1;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    انکڑے_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    صحيح batches_fit = عدد_ٹوکن / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح شروع = s * عدد_ٹوکن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_اگلا_batch(&loader);
                checkRange(loader.inputs, شروع, شروع + BT);
                checkRange(loader.targets, شروع + 1, شروع + BT + 1);
                شروع += BT;
            }
        }
    }
    dataloader_free(&loader);
    لکھو("OK\n");
}

فراغ test_multiprocess_simple(فراغ) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    لکھو("test_multiprocess_simple... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح عدد_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    انکڑے_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, عدد_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, عدد_processes, should_shuffle);

    صحيح batches_fit = عدد_ٹوکن / (B * T * عدد_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح شروع = s * عدد_ٹوکن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_اگلا_batch(&loader0);
                dataloader_اگلا_batch(&loader1);
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
    لکھو("OK\n");
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
    لکھو("test_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    صحيح process_rank = 0;
    صحيح عدد_processes = 1;
    صحيح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    انکڑے_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح کل_ٹوکن = عدد_shards * عدد_ٹوکن;
    صحيح *عدد_تکیا_inputs = (صحيح *)صفر_حافظہ_دیو(کل_ٹوکن, حجم(صحيح));
    صحيح *عدد_تکیا_targets = (صحيح *)صفر_حافظہ_دیو(کل_ٹوکن, حجم(صحيح));
    صحيح batches_fit = عدد_ٹوکن / (B * T); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح شروع = s * عدد_ٹوکن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                dataloader_اگلا_batch(&loader);
                // count up the tokens we see
                لكل (صحيح i = 0; i < BT; i++) {
                    صحيح داخلہ_token = loader.inputs[i];
                    صحيح نشانہ_token = loader.targets[i];
                    أكد(داخلہ_token >= 0 && داخلہ_token < کل_ٹوکن);
                    أكد(نشانہ_token >= 0 && نشانہ_token < کل_ٹوکن);
                    عدد_تکیا_inputs[داخلہ_token]++;
                    عدد_تکیا_targets[نشانہ_token]++;
                }
                شروع += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح ٹوکن_fit = batches_fit * BT; // number of tokens that fit per shard
    لكل (صحيح s = 0; s < عدد_shards; s++) {
        صحيح شروع = s * عدد_ٹوکن;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_تکیا_inputs + شروع, ٹوکن_fit, عدد_epochs);
        checkEquals(عدد_تکیا_inputs + شروع + ٹوکن_fit, عدد_ٹوکن - ٹوکن_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_تکیا_targets + شروع + 1, ٹوکن_fit, عدد_epochs);
        checkEquals(عدد_تکیا_targets + شروع + 1 + ٹوکن_fit,
            (s == (عدد_shards - 1)) ? عدد_ٹوکن - ٹوکن_fit - 1 : عدد_ٹوکن - ٹوکن_fit,0);
    }

    dataloader_free(&loader);
    حرر(عدد_تکیا_inputs);
    حرر(عدد_تکیا_targets);
    لکھو("OK\n");
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

    لکھو("test_multiprocess_shuffled... ");
    صحيح B = 4;
    صحيح T = 8;
    ثابت صحيح عدد_processes = 2;
    صحيح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    انکڑے_Loader loaders[عدد_processes];
    لكل (صحيح i = 0; i < عدد_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, عدد_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحيح کل_ٹوکن = عدد_shards * عدد_ٹوکن;
    صحيح *عدد_تکیا_inputs = (صحيح *)صفر_حافظہ_دیو(کل_ٹوکن, حجم(صحيح));
    صحيح *عدد_تکیا_targets = (صحيح *)صفر_حافظہ_دیو(کل_ٹوکن, حجم(صحيح));
    صحيح batches_fit = عدد_ٹوکن / (B * T * عدد_processes); // number of batches that fit per shard
    صحيح BT = B * T;
    صحيح عدد_epochs = 4;
    لكل (صحيح e = 0; e < عدد_epochs; e ++) { // epoch
        لكل (صحيح s = 0; s < عدد_shards; s++) { // shard
            صحيح شروع = s * عدد_ٹوکن;
            لكل (صحيح b = 0; b < batches_fit; b++) { // batch
                لكل (صحيح n = 0; n < عدد_processes; n++) { // dataloader
                    انکڑے_Loader *loader = &loaders[n];
                    dataloader_اگلا_batch(loader);
                    // count up the tokens we see
                    لكل (صحيح i = 0; i < BT; i++) {
                        صحيح داخلہ_token = loader->inputs[i];
                        صحيح نشانہ_token = loader->targets[i];
                        أكد(داخلہ_token >= 0 && داخلہ_token < کل_ٹوکن);
                        أكد(نشانہ_token >= 0 && نشانہ_token < کل_ٹوکن);
                        عدد_تکیا_inputs[داخلہ_token]++;
                        عدد_تکیا_targets[نشانہ_token]++;
                    }
                    شروع += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحيح ٹوکن_fit = batches_fit * (B * T * عدد_processes); // number of tokens that fit per shard
    لكل (صحيح s = 0; s < عدد_shards; s++) {
        صحيح شروع = s * عدد_ٹوکن; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_تکیا_inputs + شروع, ٹوکن_fit, عدد_epochs);
        checkEquals(عدد_تکیا_inputs + شروع + ٹوکن_fit, عدد_ٹوکن - ٹوکن_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_تکیا_targets + شروع + 1, ٹوکن_fit, عدد_epochs);
        checkEquals(عدد_تکیا_targets + شروع + 1 + ٹوکن_fit,
            (s == (عدد_shards - 1)) ? عدد_ٹوکن - ٹوکن_fit - 1 : عدد_ٹوکن - ٹوکن_fit,0);
    }

    // cleanup
    لكل (صحيح i = 0; i < عدد_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    حرر(عدد_تکیا_inputs);
    حرر(عدد_تکیا_targets);
    لکھو("OK\n");
}

صحيح مکھ(فراغ) {

    // generate a few dummy shards of data with incrementing tokens
    صحيح سرخی[سرخی_ناپ];
    uint16_t ٹوکن[عدد_ٹوکن];
    لكل (صحيح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحيح token_وتھ = shard_id * عدد_ٹوکن;
        لكل (صحيح i = 0; i < عدد_ٹوکن; i++) {
            ٹوکن[i] = token_وتھ + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        سرخی[0] = 20240520; // magic
        سرخی[1] = 1; //نسخہ
        سرخی[2] = عدد_ٹوکن; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(سرخی, حجم(صحيح), سرخی_ناپ, shard_file);
        fwrite(ٹوکن, حجم(uint16_t), عدد_ٹوکن, shard_file);
        fcloseCheck(shard_file);
        لکھو("Wrote shard %s\n", shard_name);
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