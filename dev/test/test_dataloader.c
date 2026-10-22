/*
Tests our DataLoader

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader

TODOs:
- test load/save state of DataLoader
*/
#అన్తర్భావ <unistd.h>
#అన్తర్భావ "../../llmc/dataloader.h"

#నిర్వచించు SHARD_NAME_LEN 64
అక్షరం shard_name[SHARD_NAME_LEN];
స్థిరరాశి పూర్ణాంకం అంకె_టోకెన్లు = 140;
పూర్ణాంకం అంకె_shards = 4;

శూన్య check_range(స్థిరరాశి పూర్ణాంకం *టోకెన్లు, స్థిరరాశి పూర్ణాంకం ప్రారంభం, స్థిరరాశి పూర్ణాంకం చివర, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    // checks that the tokens[0, ... end-start] are the range [start, end)
    పూర్ణాంకం n = చివర - ప్రారంభం;
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        పూర్ణాంకం token = టోకెన్లు[i];
        ఒకవేళ (token != ప్రారంభం + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, ప్రారంభం + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", పంక్తి);
            నిష్క్రమించు(EXIT_FAILURE);
        }
    }
    // printf("tokens in range [%d, %d) OK\n", start, end);
}
#నిర్వచించు checkRange(టోకెన్లు, ప్రారంభం, చివర) check_range(టోకెన్లు, ప్రారంభం, చివర, __FILE__, __LINE__)

శూన్య check_equals(స్థిరరాశి పూర్ణాంకం *టోకెన్లు, స్థిరరాశి పూర్ణాంకం n, స్థిరరాశి పూర్ణాంకం expected, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    // checks that the tokens[0, ... n] are all equal to expected
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        పూర్ణాంకం token = టోకెన్లు[i];
        ఒకవేళ (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", పంక్తి);
            నిష్క్రమించు(EXIT_FAILURE);
        }
    }
    // printf("tokens all equal to %d OK\n", expected);
}
#నిర్వచించు checkEquals(టోకెన్లు, n, expected) check_equals(టోకెన్లు, n, expected, __FILE__, __LINE__)

శూన్య test_simple(శూన్య) {
    /*
    Tests the simplest DataLoader functionality:
    - multi-shard
    - single-process
    - not shuffled
    DataLoader should just return all the tokens in order
    */
    రాయి("test_simple... ");
    పూర్ణాంకం B = 4;
    పూర్ణాంకం T = 8;
    పూర్ణాంకం process_rank = 0;
    పూర్ణాంకం అంకె_processes = 1;
    పూర్ణాంకం should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    దత్తాంశం_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, అంకె_processes, should_shuffle);

    పూర్ణాంకం batches_fit = అంకె_టోకెన్లు / (B * T); // number of batches that fit per shard
    పూర్ణాంకం BT = B * T;
    పూర్ణాంకం అంకె_epochs = 4;
    కొరకు (పూర్ణాంకం e = 0; e < అంకె_epochs; e++) { // epoch
        కొరకు (పూర్ణాంకం s = 0; s < అంకె_shards; s++) { // shard
            పూర్ణాంకం ప్రారంభం = s * అంకె_టోకెన్లు;
            కొరకు (పూర్ణాంకం b = 0; b < batches_fit; b++) { // batch
                dataloader_తదుపరి_batch(&loader);
                checkRange(loader.inputs, ప్రారంభం, ప్రారంభం + BT);
                checkRange(loader.targets, ప్రారంభం + 1, ప్రారంభం + BT + 1);
                ప్రారంభం += BT;
            }
        }
    }
    dataloader_free(&loader);
    రాయి("OK\n");
}

శూన్య test_multiprocess_simple(శూన్య) {
    /*
    Same as simple above, but using 2 processes.
    (which we of course use in a serial, single process way here)
    The DataLoaders simply pull chunks of consecutive tokens, so
    we expect them to alternate in the "token space".
    */
    రాయి("test_multiprocess_simple... ");
    పూర్ణాంకం B = 4;
    పూర్ణాంకం T = 8;
    పూర్ణాంకం అంకె_processes = 2;
    పూర్ణాంకం should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    దత్తాంశం_Loader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, అంకె_processes, should_shuffle);
    dataloader_init(&loader1, shard_name, B, T, 1, అంకె_processes, should_shuffle);

    పూర్ణాంకం batches_fit = అంకె_టోకెన్లు / (B * T * అంకె_processes); // number of batches that fit per shard
    పూర్ణాంకం BT = B * T;
    పూర్ణాంకం అంకె_epochs = 4;
    కొరకు (పూర్ణాంకం e = 0; e < అంకె_epochs; e++) { // epoch
        కొరకు (పూర్ణాంకం s = 0; s < అంకె_shards; s++) { // shard
            పూర్ణాంకం ప్రారంభం = s * అంకె_టోకెన్లు;
            కొరకు (పూర్ణాంకం b = 0; b < batches_fit; b++) { // batch
                dataloader_తదుపరి_batch(&loader0);
                dataloader_తదుపరి_batch(&loader1);
                checkRange(loader0.inputs, ప్రారంభం, ప్రారంభం + BT);
                checkRange(loader1.inputs, ప్రారంభం + BT, ప్రారంభం + 2*BT);
                checkRange(loader0.targets, ప్రారంభం + 1, ప్రారంభం + BT + 1);
                checkRange(loader1.targets, ప్రారంభం + BT + 1, ప్రారంభం + 2*BT + 1);
                ప్రారంభం += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    రాయి("OK\n");
}

శూన్య test_shuffled(శూన్య) {
    /*
    Tests the DataLoader when using shuffled:
    - multi-shard
    - single-process
    - shuffled!
    DataLoader should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times.
    */
    రాయి("test_shuffled... ");
    పూర్ణాంకం B = 4;
    పూర్ణాంకం T = 8;
    పూర్ణాంకం process_rank = 0;
    పూర్ణాంకం అంకె_processes = 1;
    పూర్ణాంకం should_shuffle = 1; // should shuffle bit turn on
    snprintf(shard_name, 64, "shard_????.bin");
    దత్తాంశం_Loader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, అంకె_processes, should_shuffle);

    // get batches from the dataloader and keep stats on what tokens we see
    పూర్ణాంకం మొత్తం_టోకెన్లు = అంకె_shards * అంకె_టోకెన్లు;
    పూర్ణాంకం *అంకె_చూసిన_inputs = (పూర్ణాంకం *)సున్నా_జ్ఞాపకం_ఇవ్వు(మొత్తం_టోకెన్లు, పరిమాణం(పూర్ణాంకం));
    పూర్ణాంకం *అంకె_చూసిన_targets = (పూర్ణాంకం *)సున్నా_జ్ఞాపకం_ఇవ్వు(మొత్తం_టోకెన్లు, పరిమాణం(పూర్ణాంకం));
    పూర్ణాంకం batches_fit = అంకె_టోకెన్లు / (B * T); // number of batches that fit per shard
    పూర్ణాంకం BT = B * T;
    పూర్ణాంకం అంకె_epochs = 4;
    కొరకు (పూర్ణాంకం e = 0; e < అంకె_epochs; e ++) { // epoch
        కొరకు (పూర్ణాంకం s = 0; s < అంకె_shards; s++) { // shard
            పూర్ణాంకం ప్రారంభం = s * అంకె_టోకెన్లు;
            కొరకు (పూర్ణాంకం b = 0; b < batches_fit; b++) { // batch
                dataloader_తదుపరి_batch(&loader);
                // count up the tokens we see
                కొరకు (పూర్ణాంకం i = 0; i < BT; i++) {
                    పూర్ణాంకం ఆదానం_token = loader.inputs[i];
                    పూర్ణాంకం లక్ష్యం_token = loader.targets[i];
                    ప్రతిజ్ఞా(ఆదానం_token >= 0 && ఆదానం_token < మొత్తం_టోకెన్లు);
                    ప్రతిజ్ఞా(లక్ష్యం_token >= 0 && లక్ష్యం_token < మొత్తం_టోకెన్లు);
                    అంకె_చూసిన_inputs[ఆదానం_token]++;
                    అంకె_చూసిన_targets[లక్ష్యం_token]++;
                }
                ప్రారంభం += BT;
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    పూర్ణాంకం టోకెన్లు_fit = batches_fit * BT; // number of tokens that fit per shard
    కొరకు (పూర్ణాంకం s = 0; s < అంకె_shards; s++) {
        పూర్ణాంకం ప్రారంభం = s * అంకె_టోకెన్లు;
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(అంకె_చూసిన_inputs + ప్రారంభం, టోకెన్లు_fit, అంకె_epochs);
        checkEquals(అంకె_చూసిన_inputs + ప్రారంభం + టోకెన్లు_fit, అంకె_టోకెన్లు - టోకెన్లు_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(అంకె_చూసిన_targets + ప్రారంభం + 1, టోకెన్లు_fit, అంకె_epochs);
        checkEquals(అంకె_చూసిన_targets + ప్రారంభం + 1 + టోకెన్లు_fit,
            (s == (అంకె_shards - 1)) ? అంకె_టోకెన్లు - టోకెన్లు_fit - 1 : అంకె_టోకెన్లు - టోకెన్లు_fit,0);
    }

    dataloader_free(&loader);
    విడుదల(అంకె_చూసిన_inputs);
    విడుదల(అంకె_చూసిన_targets);
    రాయి("OK\n");
}

శూన్య test_multiprocess_shuffled(శూన్య) {
    /*
    Tests the DataLoader when using both multiprocess and shuffled:
    - multi-shard
    - multi-process
    - shuffled!
    DataLoaders should return all the tokens, but in randperm order.
    So all we check is that we see all the tokens we expect to see,
    the correct number of times, over multiple epochs.
    */

    రాయి("test_multiprocess_shuffled... ");
    పూర్ణాంకం B = 4;
    పూర్ణాంకం T = 8;
    స్థిరరాశి పూర్ణాంకం అంకె_processes = 2;
    పూర్ణాంకం should_shuffle = 0;
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    దత్తాంశం_Loader loaders[అంకె_processes];
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, అంకె_processes, should_shuffle);
    }

    // get batches from the dataloader and keep stats on what tokens we see
    పూర్ణాంకం మొత్తం_టోకెన్లు = అంకె_shards * అంకె_టోకెన్లు;
    పూర్ణాంకం *అంకె_చూసిన_inputs = (పూర్ణాంకం *)సున్నా_జ్ఞాపకం_ఇవ్వు(మొత్తం_టోకెన్లు, పరిమాణం(పూర్ణాంకం));
    పూర్ణాంకం *అంకె_చూసిన_targets = (పూర్ణాంకం *)సున్నా_జ్ఞాపకం_ఇవ్వు(మొత్తం_టోకెన్లు, పరిమాణం(పూర్ణాంకం));
    పూర్ణాంకం batches_fit = అంకె_టోకెన్లు / (B * T * అంకె_processes); // number of batches that fit per shard
    పూర్ణాంకం BT = B * T;
    పూర్ణాంకం అంకె_epochs = 4;
    కొరకు (పూర్ణాంకం e = 0; e < అంకె_epochs; e ++) { // epoch
        కొరకు (పూర్ణాంకం s = 0; s < అంకె_shards; s++) { // shard
            పూర్ణాంకం ప్రారంభం = s * అంకె_టోకెన్లు;
            కొరకు (పూర్ణాంకం b = 0; b < batches_fit; b++) { // batch
                కొరకు (పూర్ణాంకం n = 0; n < అంకె_processes; n++) { // dataloader
                    దత్తాంశం_Loader *loader = &loaders[n];
                    dataloader_తదుపరి_batch(loader);
                    // count up the tokens we see
                    కొరకు (పూర్ణాంకం i = 0; i < BT; i++) {
                        పూర్ణాంకం ఆదానం_token = loader->inputs[i];
                        పూర్ణాంకం లక్ష్యం_token = loader->targets[i];
                        ప్రతిజ్ఞా(ఆదానం_token >= 0 && ఆదానం_token < మొత్తం_టోకెన్లు);
                        ప్రతిజ్ఞా(లక్ష్యం_token >= 0 && లక్ష్యం_token < మొత్తం_టోకెన్లు);
                        అంకె_చూసిన_inputs[ఆదానం_token]++;
                        అంకె_చూసిన_targets[లక్ష్యం_token]++;
                    }
                    ప్రారంభం += BT;
                }
            }
        }
    }

    // verify that we saw all the tokens the correct number of times
    పూర్ణాంకం టోకెన్లు_fit = batches_fit * (B * T * అంకె_processes); // number of tokens that fit per shard
    కొరకు (పూర్ణాంకం s = 0; s < అంకె_shards; s++) {
        పూర్ణాంకం ప్రారంభం = s * అంకె_టోకెన్లు; // token id that starts this shard
        // verify the inputs counts for this shard:
        // - the first tokens_fit should have been seen num_epochs times
        // - the rest of the tokens in that should should have been seen zero times
        checkEquals(అంకె_చూసిన_inputs + ప్రారంభం, టోకెన్లు_fit, అంకె_epochs);
        checkEquals(అంకె_చూసిన_inputs + ప్రారంభం + టోకెన్లు_fit, అంకె_టోకెన్లు - టోకెన్లు_fit, 0);
        // verify the target counts. same thing but offset by 1
        checkEquals(అంకె_చూసిన_targets + ప్రారంభం + 1, టోకెన్లు_fit, అంకె_epochs);
        checkEquals(అంకె_చూసిన_targets + ప్రారంభం + 1 + టోకెన్లు_fit,
            (s == (అంకె_shards - 1)) ? అంకె_టోకెన్లు - టోకెన్లు_fit - 1 : అంకె_టోకెన్లు - టోకెన్లు_fit,0);
    }

    // cleanup
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    విడుదల(అంకె_చూసిన_inputs);
    విడుదల(అంకె_చూసిన_targets);
    రాయి("OK\n");
}

పూర్ణాంకం ప్రధాన(శూన్య) {

    // generate a few dummy shards of data with incrementing tokens
    పూర్ణాంకం శీర్షిక[శీర్షిక_కొలత];
    uint16_t టోకెన్లు[అంకె_టోకెన్లు];
    కొరకు (పూర్ణాంకం shard_id = 0; shard_id < అంకె_shards; shard_id++) {
        // ensure unique tokens across the shards for ez accounting below
        పూర్ణాంకం token_స్థానభ్రంశం = shard_id * అంకె_టోకెన్లు;
        కొరకు (పూర్ణాంకం i = 0; i < అంకె_టోకెన్లు; i++) {
            టోకెన్లు[i] = token_స్థానభ్రంశం + i;
        }
        // write the shard
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        శీర్షిక[0] = 20240520; // magic
        శీర్షిక[1] = 1; //సంస్కరణ
        శీర్షిక[2] = అంకె_టోకెన్లు; // number of tokens within
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(శీర్షిక, పరిమాణం(పూర్ణాంకం), శీర్షిక_కొలత, shard_file);
        fwrite(టోకెన్లు, పరిమాణం(uint16_t), అంకె_టోకెన్లు, shard_file);
        fcloseCheck(shard_file);
        రాయి("Wrote shard %s\n", shard_name);
    }

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // clean up the shards
    కొరకు (పూర్ణాంకం shard_id = 0; shard_id < అంకె_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    తిరిగి EXIT_SUCCESS;
}