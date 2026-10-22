/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#నిర్వచితం_కాకపోతే DATALOADER_H
#నిర్వచించు DATALOADER_H

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#అన్తర్భావ "utils.h"
#అన్తర్భావ "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#నిర్వచితం_కాకపోతే _WIN32
#అన్తర్భావ <glob.h>
#అయితే_ముగింపు
// ----------------------------------------------------------------------------
// Distributed Data Loader
#నిర్వచించు శీర్షిక_కొలత 256

రకనిర్వచనం నిర్మాణం {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    పూర్ణాంకం process_rank;
    పూర్ణాంకం అంకె_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t అంకె_టోకెన్లు; // total number of tokens
    size_t shard_అంకె_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_ఫలితం; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* టోకెన్లు_file;
    // data buffers
    uint16_t* బఫరు; // we fread data from file into this buffer
    పూర్ణాంకం* inputs;  // input tokens into transformer
    పూర్ణాంకం* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    పూర్ణాంకం should_shuffle;
    పూర్ణాంకం* shard_indices;
    పూర్ణాంకం* intra_shard_indices;
    // sizes in bytes
    size_t మొత్తం_batch_కొలత_బైట్లు;  // total across all processes
    size_t local_batch_స్థానభ్రంశం_బైట్లు;  // inner-sample offset for this process
    size_t శీర్షిక_బైట్లు;  // header size in bytes
    int64_t file_కొలత_బైట్లు;
} దత్తాంశం_Loader;

int64_t dataloader_ఎక్కించు_shard(దత్తాంశం_Loader *loader, పూర్ణాంకం shard_సూచీ) {
    ఒకవేళ (loader->should_shuffle) {
        shard_సూచీ = loader->shard_indices[shard_సూచీ];
    }
    // use the first glob match as the filename for now
    స్థిరరాశి అక్షరం* filename = loader->glob_ఫలితం.gl_pathv[shard_సూచీ];
    // open the input file for reading. also only a single file can be opened at a time
    ఒకవేళ (loader->టోకెన్లు_file != ఖాళీ) {
        fcloseCheck(loader->టోకెన్లు_file);
    }
    loader->టోకెన్లు_file = fopenCheck(filename, "rb");
    // validate the header
    పూర్ణాంకం శీర్షిక[శీర్షిక_కొలత];
    freadCheck(శీర్షిక, పరిమాణం(పూర్ణాంకం), శీర్షిక_కొలత, loader->టోకెన్లు_file);
    ఒకవేళ (శీర్షిక[0] != 20240520) {
        రాయి("Bad magic in the data file\n");
        రాయి("---> HINT: Are you passing in a correct file?\n");
        రాయి("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        నిష్క్రమించు(EXIT_FAILURE);
    }
    ఒకవేళ (శీర్షిక[1] != 1) { రాయి("Bad version in data file\n"); నిష్క్రమించు(EXIT_FAILURE); }
    int64_t ntok = శీర్షిక[2]; // number of tokens in the file
    ప్రతిజ్ఞా(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->టోకెన్లు_file, 0, SEEK_END); // seek to end of file
    loader->file_కొలత_బైట్లు = ftell(loader->టోకెన్లు_file); // read the offset, i.e. file size
    fseekCheck(loader->టోకెన్లు_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_కొలత = శీర్షిక_కొలత * పరిమాణం(పూర్ణాంకం) + ntok * పరిమాణం(uint16_t);
    ఒకవేళ (loader->file_కొలత_బైట్లు != expected_file_కొలత) {
        రాయి("Error: file size is not as expected\n");
        నిష్క్రమించు(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_అంకె_samples = (ntok * పరిమాణం(uint16_t) - పరిమాణం(uint16_t)) / loader->మొత్తం_batch_కొలత_బైట్లు;
    తిరిగి ntok;
}

శూన్య prepare_intra_shard_indices_(దత్తాంశం_Loader *loader) {
    // shuffle the examples inside the shards
    ఒకవేళ (loader->intra_shard_indices != ఖాళీ) {
        // in case shards have different number of samples / sizes
        విడుదల(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (పూర్ణాంకం*)mallocCheck(loader->shard_అంకె_samples * పరిమాణం(పూర్ణాంకం));
    init_తటస్థం_permutation(loader->intra_shard_indices, (పూర్ణాంకం) loader->shard_అంకె_samples);
    random_permutation(loader->intra_shard_indices, (పూర్ణాంకం) loader->shard_అంకె_samples, &loader->shuffle_rng);
}

శూన్య dataloader_reset(దత్తాంశం_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    ఒకవేళ (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (పూర్ణాంకం) loader->glob_ఫలితం.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_ఎక్కించు_shard(loader, (పూర్ణాంకం) loader->current_shard_idx);

    ఒకవేళ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

శూన్య dataloader_advance_(దత్తాంశం_Loader *loader) {
    ఒకవేళ (loader->current_shard_idx == loader->glob_ఫలితం.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        తిరిగి;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_ఫలితం.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_ఎక్కించు_shard(loader, (పూర్ణాంకం) loader->current_shard_idx);

    ఒకవేళ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

శూన్య dataloader_init(దత్తాంశం_Loader *loader,
                     స్థిరరాశి అక్షరం* filename_నమూనా,
                     size_t B,
                     size_t T,
                     పూర్ణాంకం process_rank,
                     పూర్ణాంకం అంకె_processes,
                     పూర్ణాంకం should_shuffle) {
    loader->process_rank = process_rank;
    loader->అంకె_processes = అంకె_processes;
    loader->B = B;
    loader->T = T;
    loader->టోకెన్లు_file = ఖాళీ;
    loader->should_shuffle = should_shuffle;
    loader->శీర్షిక_బైట్లు = శీర్షిక_కొలత * పరిమాణం(పూర్ణాంకం);
    loader->మొత్తం_batch_కొలత_బైట్లు = ((loader->అంకె_processes * (loader->B * loader->T)) * పరిమాణం(uint16_t));
    loader->local_batch_స్థానభ్రంశం_బైట్లు = loader->process_rank * loader->B * loader->T * పరిమాణం(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    పూర్ణాంకం glob_స్థితి = glob(filename_నమూనా, 0, ఖాళీ, &loader->glob_ఫలితం);
    ఒకవేళ (glob_స్థితి != 0) {
        రాయి("Error: failed to glob pattern: %s\n", filename_నమూనా);
        నిష్క్రమించు(EXIT_FAILURE);
    }
    ఒకవేళ (loader->glob_ఫలితం.gl_pathc == 0) {
        రాయి("Error: no files found matching the pattern: %s\n", filename_నమూనా);
        నిష్క్రమించు(EXIT_FAILURE);
    }

    ఒకవేళ (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_విత్తనం(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (పూర్ణాంకం*)mallocCheck(loader->glob_ఫలితం.gl_pathc * పరిమాణం(పూర్ణాంకం));
        init_తటస్థం_permutation(loader->shard_indices, (పూర్ణాంకం) loader->glob_ఫలితం.gl_pathc);
        loader->intra_shard_indices = ఖాళీ;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_మొత్తం = 0;
    కొరకు (పూర్ణాంకం shard_సూచీ = 0; shard_సూచీ < loader->glob_ఫలితం.gl_pathc; shard_సూచీ++) {
        int64_t shard_ntok = dataloader_ఎక్కించు_shard(loader, shard_సూచీ);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        ప్రతిజ్ఞా(shard_ntok >= (int64_t) (అంకె_processes * B * T + 1));
        ntok_మొత్తం += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->బఫరు = (uint16_t*)mallocCheck((B * T + 1) * పరిమాణం(uint16_t));
    loader->inputs = (పూర్ణాంకం*)mallocCheck(B * T * పరిమాణం(పూర్ణాంకం));
    loader->targets = (పూర్ణాంకం*)mallocCheck(B * T * పరిమాణం(పూర్ణాంకం));
    loader->అంకె_టోకెన్లు = ntok_మొత్తం;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

శూన్య dataloader_ఎక్కించు_batch(దత్తాంశం_Loader* loader) {
    ప్రతిజ్ఞా(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != ఖాళీ));
    ప్రతిజ్ఞా(loader->current_sample_idx < loader->shard_అంకె_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_స్థానభ్రంశం_బైట్లు = idx * loader->మొత్తం_batch_కొలత_బైట్లు;
    int64_t current_స్థానభ్రంశం = loader->శీర్షిక_బైట్లు + global_batch_స్థానభ్రంశం_బైట్లు + loader->local_batch_స్థానభ్రంశం_బైట్లు;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->టోకెన్లు_file, (పూర్ణాంకం) current_స్థానభ్రంశం, SEEK_SET);
    freadCheck(loader->బఫరు, పరిమాణం(uint16_t), B*T+1, loader->టోకెన్లు_file);
    // decode the buffer into inputs and targets (cast to int)
    కొరకు (పూర్ణాంకం i = 0; i < B*T; i++) {
        loader->inputs[i] = (పూర్ణాంకం)loader->బఫరు[i];
        loader->targets[i] = (పూర్ణాంకం)loader->బఫరు[i+1];
    }
}

శూన్య dataloader_తదుపరి_batch(దత్తాంశం_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    ఒకవేళ (loader->current_sample_idx >= loader->shard_అంకె_samples) {
        dataloader_advance_(loader);
    }
    dataloader_ఎక్కించు_batch(loader);
    loader->current_sample_idx += 1;
}


శూన్య dataloader_resume(దత్తాంశం_Loader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_ఎక్కించు_shard(loader, (పూర్ణాంకం) loader->current_shard_idx);
}

శూన్య dataloader_free(దత్తాంశం_Loader *loader) {
    విడుదల(loader->బఫరు);
    విడుదల(loader->inputs);
    విడుదల(loader->targets);
    ఒకవేళ (loader->should_shuffle) {
        విడుదల(loader->shard_indices);
        విడుదల(loader->intra_shard_indices);
    }
    fcloseCheck(loader->టోకెన్లు_file);
    globfree(&loader->glob_ఫలితం);
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t:
    - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
#నిర్వచించు ASSUMED_అంకె_COMPLETIONS 4
// helper macro for ceildiv
#నిర్వచించు CEIL_DIV(M, N) (((M) + (N)-1) / (N))

రకనిర్వచనం నిర్మాణం {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    పూర్ణాంకం process_rank;
    పూర్ణాంకం అంకె_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* బఫరు; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    పూర్ణాంకం అంకె_examples; // in total across all processes
    పూర్ణాంకం అంకె_batches; // to process the entire dataset across all processes
    పూర్ణాంకం ప్రారంభం_example_సూచీ; // the assignment of work for this process, start
    పూర్ణాంకం చివర_example_సూచీ; // and end. start is inclusive, end is exclusive
    పూర్ణాంకం current_example_సూచీ; // the next example we would read
    పూర్ణాంకం* inputs;  // input tokens into transformer
    పూర్ణాంకం* targets; // target tokens for the transformer
    అక్షరం* ముసుగు; // mask=1 at all completion token locations
    పూర్ణాంకం* label; // the correct completion labels
    పూర్ణాంకం అంకె_completions; // number of completions for this example
} EvalLoader;

శూన్య evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    పూర్ణాంకం examples_చొప్పున_process = CEIL_DIV(loader->అంకె_examples, loader->అంకె_processes);
    పూర్ణాంకం can_fit_examples = (పూర్ణాంకం) (loader->B / ASSUMED_అంకె_COMPLETIONS);
    ఒకవేళ (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        రాయి("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_అంకె_COMPLETIONS);
        రాయి("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        నిష్క్రమించు(EXIT_FAILURE);
    }
    loader->అంకె_batches = CEIL_DIV(examples_చొప్పున_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->ప్రారంభం_example_సూచీ = examples_చొప్పున_process * loader->process_rank;
    loader->చివర_example_సూచీ = examples_చొప్పున_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    ఒకవేళ (loader->చివర_example_సూచీ > loader->అంకె_examples) {
        loader->చివర_example_సూచీ = loader->అంకె_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t శీర్షిక_బైట్లు = శీర్షిక_కొలత * పరిమాణం(పూర్ణాంకం);
    fseekCheck(loader->eval_file, (పూర్ణాంకం) శీర్షిక_బైట్లు, SEEK_SET);
    కొరకు (పూర్ణాంకం i = 0; i < loader->ప్రారంభం_example_సూచీ; i++) {
        uint16_t example_శీర్షిక[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_శీర్షిక[0], పరిమాణం(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        ప్రతిజ్ఞా(example_శీర్షిక[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        ప్రతిజ్ఞా(example_శీర్షిక[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t మిగిలిన_బైట్లు = example_శీర్షిక[1] - పరిమాణం(uint16_t) * 3;
        ప్రతిజ్ఞా(మిగిలిన_బైట్లు > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (పూర్ణాంకం) మిగిలిన_బైట్లు, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_సూచీ = loader->ప్రారంభం_example_సూచీ;
}

శూన్య evalloader_init(EvalLoader *loader,
                     స్థిరరాశి అక్షరం* filename,
                     size_t B,
                     size_t T,
                     పూర్ణాంకం process_rank,
                     పూర్ణాంకం అంకె_processes) {
    loader->process_rank = process_rank;
    loader->అంకె_processes = అంకె_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    పూర్ణాంకం శీర్షిక[శీర్షిక_కొలత];
    freadCheck(శీర్షిక, పరిమాణం(పూర్ణాంకం), శీర్షిక_కొలత, loader->eval_file);
    ఒకవేళ (శీర్షిక[0] != 20240522) { రాయి("Bad magic in eval file\n"); నిష్క్రమించు(EXIT_FAILURE); }
    ఒకవేళ (శీర్షిక[1] != 1) { రాయి("Bad version in data file\n"); నిష్క్రమించు(EXIT_FAILURE); }
    loader->అంకె_examples = శీర్షిక[2]; // number of examples in the file
    ప్రతిజ్ఞా(loader->అంకె_examples >= అంకె_processes); // avoid headaches for now
    size_t longest_example_బైట్లు = శీర్షిక[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    ప్రతిజ్ఞా(longest_example_బైట్లు > 0 && longest_example_బైట్లు < (1+ASSUMED_అంకె_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    పూర్ణాంకం can_fit_examples = (పూర్ణాంకం) (B / ASSUMED_అంకె_COMPLETIONS);
    loader->బఫరు = (uint16_t*)mallocCheck(longest_example_బైట్లు);
    loader->inputs = (పూర్ణాంకం*)సున్నా_జ్ఞాపకం_ఇవ్వు(B * T, పరిమాణం(పూర్ణాంకం));
    loader->targets = (పూర్ణాంకం*)సున్నా_జ్ఞాపకం_ఇవ్వు(B * T, పరిమాణం(పూర్ణాంకం));
    loader->ముసుగు = (అక్షరం*)mallocCheck(B * T * పరిమాణం(అక్షరం));
    loader->label = (పూర్ణాంకం*)mallocCheck(can_fit_examples * పరిమాణం(పూర్ణాంకం));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

శూన్య evalloader_తదుపరి_example(EvalLoader *loader, పూర్ణాంకం example_batch_సూచీ) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    పూర్ణాంకం batch_dim_స్థానభ్రంశం = example_batch_సూచీ * ASSUMED_అంకె_COMPLETIONS;
    // read the current example header
    uint16_t example_శీర్షిక[3];
    freadCheck(&example_శీర్షిక[0], పరిమాణం(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    ప్రతిజ్ఞా(example_శీర్షిక[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    ప్రతిజ్ఞా(example_శీర్షిక[2] == loader->current_example_సూచీ); // <EXAMPLE_INDEX> should match the loop index
    ప్రతిజ్ఞా(example_శీర్షిక[2] >= loader->ప్రారంభం_example_సూచీ && example_శీర్షిక[2] < loader->చివర_example_సూచీ);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_బైట్లు = example_శీర్షిక[1] - పరిమాణం(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->బఫరు, పరిమాణం(అక్షరం), example_బైట్లు, loader->eval_file);
    // process the example label
    పూర్ణాంకం label = (పూర్ణాంకం)loader->బఫరు[0];
    పూర్ణాంకం can_fit_examples = (పూర్ణాంకం) (loader->B / ASSUMED_అంకె_COMPLETIONS);
    ప్రతిజ్ఞా(label >= 0 && label < ASSUMED_అంకె_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    ప్రతిజ్ఞా(example_batch_సూచీ >= 0 && example_batch_సూచీ < can_fit_examples);
    loader->label[example_batch_సూచీ] = label; // store for output
    // process the number of completions
    పూర్ణాంకం అంకె_completions = (పూర్ణాంకం)loader->బఫరు[1];
    ప్రతిజ్ఞా(అంకె_completions == ASSUMED_అంకె_COMPLETIONS); // we expect 4 completions for now
    ప్రతిజ్ఞా(batch_dim_స్థానభ్రంశం + అంకె_completions <= B); // we expect to fit in the batch
    loader->అంకె_completions = అంకె_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    పూర్ణాంకం context_నిడివి = (పూర్ణాంకం)loader->బఫరు[2];
    uint16_t *context_టోకెన్లు_ప్రారంభం = &loader->బఫరు[3]; // where the tokens start
    ప్రతిజ్ఞా(context_నిడివి > 0 && context_నిడివి < T); // context is non-empty and up to T
    కొరకు (పూర్ణాంకం b = 0; b < అంకె_completions; b++) {
        కొరకు (పూర్ణాంకం i = 0; i < context_నిడివి; i++) {
            పూర్ణాంకం boff = batch_dim_స్థానభ్రంశం + b;
            పూర్ణాంకం tok_cur = (పూర్ణాంకం)context_టోకెన్లు_ప్రారంభం[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->బఫరు + 3 + context_నిడివి;
    కొరకు (పూర్ణాంకం c = 0; c < అంకె_completions; c++) {
        పూర్ణాంకం coff = batch_dim_స్థానభ్రంశం + c;
        పూర్ణాంకం completion_నిడివి = (పూర్ణాంకం)completions_iter[0];
        uint16_t *completion_టోకెన్లు_ప్రారంభం = completions_iter + 1;
        ప్రతిజ్ఞా(completion_నిడివి > 0 && context_నిడివి + completion_నిడివి < T); // things fit?
        కొరకు (పూర్ణాంకం i = 0; i < completion_నిడివి; i++) {
            పూర్ణాంకం tok_cur = (పూర్ణాంకం)completion_టోకెన్లు_ప్రారంభం[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_నిడివి + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_నిడివి + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->ముసుగు[coff * T + context_నిడివి + i - 1] = 1;
        }
        completions_iter += 1 + completion_నిడివి; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_సూచీ += 1;
}

శూన్య evalloader_తదుపరి_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->ముసుగు, 0, B * T * పరిమాణం(అక్షరం));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    పూర్ణాంకం can_fit_examples = (పూర్ణాంకం) (B / ASSUMED_అంకె_COMPLETIONS); // how many examples can we fit in the batch?
    కొరకు (పూర్ణాంకం i = 0; i < can_fit_examples; i++) {
        ఒకవేళ (loader->current_example_సూచీ >= loader->చివర_example_సూచీ) {
            ఆపు; // this process has exhausted its work, noop from here on
        }
        evalloader_తదుపరి_example(loader, i);
    }
}

పూర్ణాంకం evalloader_stat_losses(EvalLoader *loader, దశాంశం* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    పూర్ణాంకం correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    పూర్ణాంకం can_fit_examples = (పూర్ణాంకం) (B / ASSUMED_అంకె_COMPLETIONS);
    కొరకు (పూర్ణాంకం i = 0; i < can_fit_examples; i++) {
        దశాంశం min_loss = 0.0f;
        పూర్ణాంకం min_loss_సూచీ = -1;
        అక్షరం చురుకైన = 0; // is this example active or fully empty?
        // iterate the completions in this example
        కొరకు (పూర్ణాంకం b = 0; b < ASSUMED_అంకె_COMPLETIONS; b++) {
            పూర్ణాంకం boff = i * ASSUMED_అంకె_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            దశాంశం సరాసరి_loss = 0.0f;
            పూర్ణాంకం లెక్క = 0;
            కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
                అక్షరం ముసుగు = loader->ముసుగు[boff * T + t];
                ఒకవేళ (ముసుగు == 1) {
                    చురుకైన = 1;
                    సరాసరి_loss += losses[boff * T + t];
                    లెక్క++;
                }
            }
            ఒకవేళ (లెక్క > 0) { సరాసరి_loss /= లెక్క; }
            ఒకవేళ (b == 0 || సరాసరి_loss < min_loss) {
                min_loss = సరాసరి_loss;
                min_loss_సూచీ = b;
            }
        }
        ఒకవేళ (చురుకైన && (min_loss_సూచీ == loader->label[i])) {
            correct += 1;
        }
    }
    తిరిగి correct;
}

శూన్య evalloader_free(EvalLoader *loader) {
    విడుదల(loader->బఫరు);
    విడుదల(loader->inputs);
    విడుదల(loader->targets);
    విడుదల(loader->ముసుగు);
    విడుదల(loader->label);
    fcloseCheck(loader->eval_file);
}

#అయితే_ముగింపు // DATALOADER_H