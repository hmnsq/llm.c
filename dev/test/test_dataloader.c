/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#شامل <unistd.h>
#شامل "../../llmc/dataloader.h"

#تعریف_کرو SHARD_NAME_LEN 64
حرف shard_name[SHARD_NAME_LEN];
ثابت صحیح عدد_علامات = 140;
صحیح عدد_shards = 4;

خالی check_range(ثابت صحیح *علامات, ثابت صحیح آغاز, ثابت صحیح انتہا, ثابت حرف *file, صحیح سطر) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    صحیح n = انتہا - آغاز;
    برائے (صحیح i = 0; i < n; i++) {
        صحیح token = علامات[i];
        اگر (token != آغاز + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, آغاز + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            باہر_نکلو(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#تعریف_کرو checkRange(علامات, آغاز, انتہا) check_range(علامات, آغاز, انتہا, __FILE__, __LINE__)

خالی check_equals(ثابت صحیح *علامات, ثابت صحیح n, ثابت صحیح expected, ثابت حرف *file, صحیح سطر) {
    // checks that the tokens[0, ... n] are all equal to expected
    برائے (صحیح i = 0; i < n; i++) {
        صحیح token = علامات[i];
        اگر (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", سطر);
            باہر_نکلو(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#تعریف_کرو checkEquals(علامات, n, expected) check_equals(علامات, n, expected, __FILE__, __LINE__)

خالی test_simple(خالی) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    لکھو("test_simple... ");
    صحیح B = 4;
    صحیح T = 8;
    صحیح process_rank = 0;
    صحیح عدد_processes = 1;
    صحیح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ڈیٹا_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    صحیح batches_fit = عدد_علامات / (B * T); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح عدد_epochs = 4;
    برائے (صحیح e = 0; e < عدد_epochs; e++) { // epoch
        برائے (صحیح s = 0; s < عدد_shards; s++) { // shard
            صحیح آغاز = s * عدد_علامات;
            برائے (صحیح b = 0; b < batches_fit; b++) { // batch
                dataloader_اگلا_batch(&loader);
                checkRange(loader.inputs, آغاز, آغاز + BT);
                checkRange(loader.targets, آغاز + 1, آغاز + BT + 1);
                آغاز += BT;
            }
        }
    }
    dataloader_free(&loader);
    لکھو("OK\n");
}

خالی test_multiprocess_simple(خالی) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    لکھو("test_multiprocess_simple... ");
    صحیح B = 4;
    صحیح T = 8;
    صحیح عدد_processes = 2;
    صحیح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ڈیٹا_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, عدد_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, عدد_processes, should_shuffle);

    صحیح batches_fit = عدد_علامات / (B * T * عدد_processes); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح عدد_epochs = 4;
    برائے (صحیح e = 0; e < عدد_epochs; e++) { // epoch
        برائے (صحیح s = 0; s < عدد_shards; s++) { // shard
            صحیح آغاز = s * عدد_علامات;
            برائے (صحیح b = 0; b < batches_fit; b++) { // batch
                dataloader_اگلا_batch(&loader0);
                dataloader_اگلا_batch(&loader1);
                checkRange(loader0.inputs, آغاز, آغاز + BT);
                checkRange(loader1.inputs, آغاز + BT, آغاز + 2*BT);
                checkRange(loader0.targets, آغاز + 1, آغاز + BT + 1);
                checkRange(loader1.targets, آغاز + BT + 1, آغاز + 2*BT + 1);
                آغاز += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    لکھو("OK\n");
}

خالی test_shuffled(خالی) {
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
    صحیح B = 4;
    صحیح T = 8;
    صحیح process_rank = 0;
    صحیح عدد_processes = 1;
    صحیح should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    ڈیٹا_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, عدد_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    صحیح کل_علامات = عدد_shards * عدد_علامات;
    صحیح *عدد_نظر_آیا_inputs = (صحیح *)صفر_حافظہ_دو(کل_علامات, حجم(صحیح));
    صحیح *عدد_نظر_آیا_targets = (صحیح *)صفر_حافظہ_دو(کل_علامات, حجم(صحیح));
    صحیح batches_fit = عدد_علامات / (B * T); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح عدد_epochs = 4;
    برائے (صحیح e = 0; e < عدد_epochs; e ++) { // epoch
        برائے (صحیح s = 0; s < عدد_shards; s++) { // shard
            صحیح آغاز = s * عدد_علامات;
            برائے (صحیح b = 0; b < batches_fit; b++) { // batch
                dataloader_اگلا_batch(&loader);
                // count up the tokens we see
                برائے (صحیح i = 0; i < BT; i++) {
                    صحیح داخلہ_token = loader.inputs[i];
                    صحیح ہدف_token = loader.targets[i];
                    أكد(داخلہ_token >= 0 && داخلہ_token < کل_علامات);
                    أكد(ہدف_token >= 0 && ہدف_token < کل_علامات);
                    عدد_نظر_آیا_inputs[داخلہ_token]++;
                    عدد_نظر_آیا_targets[ہدف_token]++;
                }
                آغاز += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحیح علامات_fit = batches_fit * BT; // number of tokens that fit per shard
    برائے (صحیح s = 0; s < عدد_shards; s++) {
        صحیح آغاز = s * عدد_علامات;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_نظر_آیا_inputs + آغاز, علامات_fit, عدد_epochs);
        checkEquals(عدد_نظر_آیا_inputs + آغاز + علامات_fit, عدد_علامات - علامات_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_نظر_آیا_targets + آغاز + 1, علامات_fit, عدد_epochs);
        checkEquals(عدد_نظر_آیا_targets + آغاز + 1 + علامات_fit,
            (s == (عدد_shards - 1)) ? عدد_علامات - علامات_fit - 1 : عدد_علامات - علامات_fit,0);
    }

    dataloader_free(&loader);
    آزاد(عدد_نظر_آیا_inputs);
    آزاد(عدد_نظر_آیا_targets);
    لکھو("OK\n");
}

خالی test_multiprocess_shuffled(خالی) {
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
    صحیح B = 4;
    صحیح T = 8;
    ثابت صحیح عدد_processes = 2;
    صحیح should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    ڈیٹا_Loader loaders[عدد_processes];
    برائے (صحیح i = 0; i < عدد_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, عدد_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    صحیح کل_علامات = عدد_shards * عدد_علامات;
    صحیح *عدد_نظر_آیا_inputs = (صحیح *)صفر_حافظہ_دو(کل_علامات, حجم(صحیح));
    صحیح *عدد_نظر_آیا_targets = (صحیح *)صفر_حافظہ_دو(کل_علامات, حجم(صحیح));
    صحیح batches_fit = عدد_علامات / (B * T * عدد_processes); // number of batches that fit per shard
    صحیح BT = B * T;
    صحیح عدد_epochs = 4;
    برائے (صحیح e = 0; e < عدد_epochs; e ++) { // epoch
        برائے (صحیح s = 0; s < عدد_shards; s++) { // shard
            صحیح آغاز = s * عدد_علامات;
            برائے (صحیح b = 0; b < batches_fit; b++) { // batch
                برائے (صحیح n = 0; n < عدد_processes; n++) { // dataloader
                    ڈیٹا_Loader *loader = &loaders[n];
                    dataloader_اگلا_batch(loader);
                    // count up the tokens we see
                    برائے (صحیح i = 0; i < BT; i++) {
                        صحیح داخلہ_token = loader->inputs[i];
                        صحیح ہدف_token = loader->targets[i];
                        أكد(داخلہ_token >= 0 && داخلہ_token < کل_علامات);
                        أكد(ہدف_token >= 0 && ہدف_token < کل_علامات);
                        عدد_نظر_آیا_inputs[داخلہ_token]++;
                        عدد_نظر_آیا_targets[ہدف_token]++;
                    }
                    آغاز += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    صحیح علامات_fit = batches_fit * (B * T * عدد_processes); // number of tokens that fit per shard
    برائے (صحیح s = 0; s < عدد_shards; s++) {
        صحیح آغاز = s * عدد_علامات; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(عدد_نظر_آیا_inputs + آغاز, علامات_fit, عدد_epochs);
        checkEquals(عدد_نظر_آیا_inputs + آغاز + علامات_fit, عدد_علامات - علامات_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(عدد_نظر_آیا_targets + آغاز + 1, علامات_fit, عدد_epochs);
        checkEquals(عدد_نظر_آیا_targets + آغاز + 1 + علامات_fit,
            (s == (عدد_shards - 1)) ? عدد_علامات - علامات_fit - 1 : عدد_علامات - علامات_fit,0);
    }

    // cleanup
    برائے (صحیح i = 0; i < عدد_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    آزاد(عدد_نظر_آیا_inputs);
    آزاد(عدد_نظر_آیا_targets);
    لکھو("OK\n");
}

صحیح مین(خالی) {

    // generate a few dummy shards of data with incrementing tokens
    صحیح سرخی[سرخی_سائز];
    uint16_t علامات[عدد_علامات];
    برائے (صحیح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        صحیح token_انحراف = shard_id * عدد_علامات;
        برائے (صحیح i = 0; i < عدد_علامات; i++) {
            علامات[i] = token_انحراف + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        سرخی[0] = 20240520; // magic
        سرخی[1] = 1; //نسخہ
        سرخی[2] = عدد_علامات; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(سرخی, حجم(صحیح), سرخی_سائز, shard_file);
        fwrite(علامات, حجم(uint16_t), عدد_علامات, shard_file);
        fcloseCheck(shard_file);
        لکھو("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    برائے (صحیح shard_id = 0; shard_id < عدد_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    واپس EXIT_SUCCESS;
}