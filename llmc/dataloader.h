/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ DATALOADER_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DATALOADER_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#ਅਨ੍ਤਰ੍ਭਾਵ "utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ _WIN32
#ਅਨ੍ਤਰ੍ਭਾਵ <glob.h>
#ਜੇ_ਅੰਤ
// ----------------------------------------------------------------------------
// Distributed Data Loader
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸਿਰਲੇਖ_ਮਾਪ 256

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    ਪੂਰਨ process_rank;
    ਪੂਰਨ ਅੰਕ_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t ਅੰਕ_ਟੋਕਨ; // total number of tokens
    size_t shard_ਅੰਕ_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_ਨਤੀਜਾ; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* ਟੋਕਨ_file;
    // data buffers
    uint16_t* ਬਫਰ; // we fread data from file into this buffer
    ਪੂਰਨ* inputs;  // input tokens into transformer
    ਪੂਰਨ* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    ਪੂਰਨ should_shuffle;
    ਪੂਰਨ* shard_indices;
    ਪੂਰਨ* intra_shard_indices;
    // sizes in bytes
    size_t ਕੁੱਲ_batch_ਮਾਪ_ਬਾਈਟਾਂ;  // total across all processes
    size_t local_batch_ਵਿੱਥ_ਬਾਈਟਾਂ;  // inner-sample offset for this process
    size_t ਸਿਰਲੇਖ_ਬਾਈਟਾਂ;  // header size in bytes
    int64_t file_ਮਾਪ_ਬਾਈਟਾਂ;
} ਅੰਕੜੇ_Loader;

int64_t dataloader_ਲੱਦੋ_shard(ਅੰਕੜੇ_Loader *loader, ਪੂਰਨ shard_ਸੂਚਕਾਂਕ) {
    ਜੇ (loader->should_shuffle) {
        shard_ਸੂਚਕਾਂਕ = loader->shard_indices[shard_ਸੂਚਕਾਂਕ];
    }
    // use the first glob match as the filename for now
    ਅਚਲ ਅੱਖਰ* filename = loader->glob_ਨਤੀਜਾ.gl_pathv[shard_ਸੂਚਕਾਂਕ];
    // open the input file for reading. also only a single file can be opened at a time
    ਜੇ (loader->ਟੋਕਨ_file != ਨਿਰਾ) {
        fcloseCheck(loader->ਟੋਕਨ_file);
    }
    loader->ਟੋਕਨ_file = fopenCheck(filename, "rb");
    // validate the header
    ਪੂਰਨ ਸਿਰਲੇਖ[ਸਿਰਲੇਖ_ਮਾਪ];
    freadCheck(ਸਿਰਲੇਖ, ਆਕਾਰ(ਪੂਰਨ), ਸਿਰਲੇਖ_ਮਾਪ, loader->ਟੋਕਨ_file);
    ਜੇ (ਸਿਰਲੇਖ[0] != 20240520) {
        ਲਿਖੋ("Bad magic in the data file\n");
        ਲਿਖੋ("---> HINT: Are you passing in a correct file?\n");
        ਲਿਖੋ("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    ਜੇ (ਸਿਰਲੇਖ[1] != 1) { ਲਿਖੋ("Bad version in data file\n"); ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE); }
    int64_t ntok = ਸਿਰਲੇਖ[2]; // number of tokens in the file
    ਪ੍ਰਤਿਜ੍ਞਾ(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->ਟੋਕਨ_file, 0, SEEK_END); // seek to end of file
    loader->file_ਮਾਪ_ਬਾਈਟਾਂ = ftell(loader->ਟੋਕਨ_file); // read the offset, i.e. file size
    fseekCheck(loader->ਟੋਕਨ_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_ਮਾਪ = ਸਿਰਲੇਖ_ਮਾਪ * ਆਕਾਰ(ਪੂਰਨ) + ntok * ਆਕਾਰ(uint16_t);
    ਜੇ (loader->file_ਮਾਪ_ਬਾਈਟਾਂ != expected_file_ਮਾਪ) {
        ਲਿਖੋ("Error: file size is not as expected\n");
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_ਅੰਕ_samples = (ntok * ਆਕਾਰ(uint16_t) - ਆਕਾਰ(uint16_t)) / loader->ਕੁੱਲ_batch_ਮਾਪ_ਬਾਈਟਾਂ;
    ਵਾਪਸ ntok;
}

ਖਾਲੀ prepare_intra_shard_indices_(ਅੰਕੜੇ_Loader *loader) {
    // shuffle the examples inside the shards
    ਜੇ (loader->intra_shard_indices != ਨਿਰਾ) {
        // in case shards have different number of samples / sizes
        ਮੁਕਤ(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (ਪੂਰਨ*)mallocCheck(loader->shard_ਅੰਕ_samples * ਆਕਾਰ(ਪੂਰਨ));
    init_ਪਛਾਣ_ਤੱਤ_permutation(loader->intra_shard_indices, (ਪੂਰਨ) loader->shard_ਅੰਕ_samples);
    random_permutation(loader->intra_shard_indices, (ਪੂਰਨ) loader->shard_ਅੰਕ_samples, &loader->shuffle_rng);
}

ਖਾਲੀ dataloader_reset(ਅੰਕੜੇ_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    ਜੇ (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (ਪੂਰਨ) loader->glob_ਨਤੀਜਾ.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_ਲੱਦੋ_shard(loader, (ਪੂਰਨ) loader->current_shard_idx);

    ਜੇ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

ਖਾਲੀ dataloader_advance_(ਅੰਕੜੇ_Loader *loader) {
    ਜੇ (loader->current_shard_idx == loader->glob_ਨਤੀਜਾ.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        ਵਾਪਸ;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_ਨਤੀਜਾ.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_ਲੱਦੋ_shard(loader, (ਪੂਰਨ) loader->current_shard_idx);

    ਜੇ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

ਖਾਲੀ dataloader_init(ਅੰਕੜੇ_Loader *loader,
                     ਅਚਲ ਅੱਖਰ* filename_ਨਮੂਨਾ,
                     size_t B,
                     size_t T,
                     ਪੂਰਨ process_rank,
                     ਪੂਰਨ ਅੰਕ_processes,
                     ਪੂਰਨ should_shuffle) {
    loader->process_rank = process_rank;
    loader->ਅੰਕ_processes = ਅੰਕ_processes;
    loader->B = B;
    loader->T = T;
    loader->ਟੋਕਨ_file = ਨਿਰਾ;
    loader->should_shuffle = should_shuffle;
    loader->ਸਿਰਲੇਖ_ਬਾਈਟਾਂ = ਸਿਰਲੇਖ_ਮਾਪ * ਆਕਾਰ(ਪੂਰਨ);
    loader->ਕੁੱਲ_batch_ਮਾਪ_ਬਾਈਟਾਂ = ((loader->ਅੰਕ_processes * (loader->B * loader->T)) * ਆਕਾਰ(uint16_t));
    loader->local_batch_ਵਿੱਥ_ਬਾਈਟਾਂ = loader->process_rank * loader->B * loader->T * ਆਕਾਰ(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    ਪੂਰਨ glob_ਸਥਿਤੀ = glob(filename_ਨਮੂਨਾ, 0, ਨਿਰਾ, &loader->glob_ਨਤੀਜਾ);
    ਜੇ (glob_ਸਥਿਤੀ != 0) {
        ਲਿਖੋ("Error: failed to glob pattern: %s\n", filename_ਨਮੂਨਾ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    ਜੇ (loader->glob_ਨਤੀਜਾ.gl_pathc == 0) {
        ਲਿਖੋ("Error: no files found matching the pattern: %s\n", filename_ਨਮੂਨਾ);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }

    ਜੇ (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_ਬੀਜ(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (ਪੂਰਨ*)mallocCheck(loader->glob_ਨਤੀਜਾ.gl_pathc * ਆਕਾਰ(ਪੂਰਨ));
        init_ਪਛਾਣ_ਤੱਤ_permutation(loader->shard_indices, (ਪੂਰਨ) loader->glob_ਨਤੀਜਾ.gl_pathc);
        loader->intra_shard_indices = ਨਿਰਾ;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_ਕੁੱਲ = 0;
    ਲਈ (ਪੂਰਨ shard_ਸੂਚਕਾਂਕ = 0; shard_ਸੂਚਕਾਂਕ < loader->glob_ਨਤੀਜਾ.gl_pathc; shard_ਸੂਚਕਾਂਕ++) {
        int64_t shard_ntok = dataloader_ਲੱਦੋ_shard(loader, shard_ਸੂਚਕਾਂਕ);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        ਪ੍ਰਤਿਜ੍ਞਾ(shard_ntok >= (int64_t) (ਅੰਕ_processes * B * T + 1));
        ntok_ਕੁੱਲ += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->ਬਫਰ = (uint16_t*)mallocCheck((B * T + 1) * ਆਕਾਰ(uint16_t));
    loader->inputs = (ਪੂਰਨ*)mallocCheck(B * T * ਆਕਾਰ(ਪੂਰਨ));
    loader->targets = (ਪੂਰਨ*)mallocCheck(B * T * ਆਕਾਰ(ਪੂਰਨ));
    loader->ਅੰਕ_ਟੋਕਨ = ntok_ਕੁੱਲ;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

ਖਾਲੀ dataloader_ਲੱਦੋ_batch(ਅੰਕੜੇ_Loader* loader) {
    ਪ੍ਰਤਿਜ੍ਞਾ(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != ਨਿਰਾ));
    ਪ੍ਰਤਿਜ੍ਞਾ(loader->current_sample_idx < loader->shard_ਅੰਕ_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_ਵਿੱਥ_ਬਾਈਟਾਂ = idx * loader->ਕੁੱਲ_batch_ਮਾਪ_ਬਾਈਟਾਂ;
    int64_t current_ਵਿੱਥ = loader->ਸਿਰਲੇਖ_ਬਾਈਟਾਂ + global_batch_ਵਿੱਥ_ਬਾਈਟਾਂ + loader->local_batch_ਵਿੱਥ_ਬਾਈਟਾਂ;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->ਟੋਕਨ_file, (ਪੂਰਨ) current_ਵਿੱਥ, SEEK_SET);
    freadCheck(loader->ਬਫਰ, ਆਕਾਰ(uint16_t), B*T+1, loader->ਟੋਕਨ_file);
    // decode the buffer into inputs and targets (cast to int)
    ਲਈ (ਪੂਰਨ i = 0; i < B*T; i++) {
        loader->inputs[i] = (ਪੂਰਨ)loader->ਬਫਰ[i];
        loader->targets[i] = (ਪੂਰਨ)loader->ਬਫਰ[i+1];
    }
}

ਖਾਲੀ dataloader_ਅਗਲਾ_batch(ਅੰਕੜੇ_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    ਜੇ (loader->current_sample_idx >= loader->shard_ਅੰਕ_samples) {
        dataloader_advance_(loader);
    }
    dataloader_ਲੱਦੋ_batch(loader);
    loader->current_sample_idx += 1;
}


ਖਾਲੀ dataloader_resume(ਅੰਕੜੇ_Loader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_ਲੱਦੋ_shard(loader, (ਪੂਰਨ) loader->current_shard_idx);
}

ਖਾਲੀ dataloader_free(ਅੰਕੜੇ_Loader *loader) {
    ਮੁਕਤ(loader->ਬਫਰ);
    ਮੁਕਤ(loader->inputs);
    ਮੁਕਤ(loader->targets);
    ਜੇ (loader->should_shuffle) {
        ਮੁਕਤ(loader->shard_indices);
        ਮੁਕਤ(loader->intra_shard_indices);
    }
    fcloseCheck(loader->ਟੋਕਨ_file);
    globfree(&loader->glob_ਨਤੀਜਾ);
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
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ASSUMED_ਅੰਕ_COMPLETIONS 4
// helper macro for ceildiv
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CEIL_DIV(M, N) (((M) + (N)-1) / (N))

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    ਪੂਰਨ process_rank;
    ਪੂਰਨ ਅੰਕ_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* ਬਫਰ; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    ਪੂਰਨ ਅੰਕ_examples; // in total across all processes
    ਪੂਰਨ ਅੰਕ_batches; // to process the entire dataset across all processes
    ਪੂਰਨ ਸ਼ੁਰੂ_example_ਸੂਚਕਾਂਕ; // the assignment of work for this process, start
    ਪੂਰਨ ਸਿਰਾ_example_ਸੂਚਕਾਂਕ; // and end. start is inclusive, end is exclusive
    ਪੂਰਨ current_example_ਸੂਚਕਾਂਕ; // the next example we would read
    ਪੂਰਨ* inputs;  // input tokens into transformer
    ਪੂਰਨ* targets; // target tokens for the transformer
    ਅੱਖਰ* ਨਕਾਬ; // mask=1 at all completion token locations
    ਪੂਰਨ* label; // the correct completion labels
    ਪੂਰਨ ਅੰਕ_completions; // number of completions for this example
} EvalLoader;

ਖਾਲੀ evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    ਪੂਰਨ examples_ਪ੍ਰਤੀ_process = CEIL_DIV(loader->ਅੰਕ_examples, loader->ਅੰਕ_processes);
    ਪੂਰਨ can_fit_examples = (ਪੂਰਨ) (loader->B / ASSUMED_ਅੰਕ_COMPLETIONS);
    ਜੇ (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        ਲਿਖੋ("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_ਅੰਕ_COMPLETIONS);
        ਲਿਖੋ("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    loader->ਅੰਕ_batches = CEIL_DIV(examples_ਪ੍ਰਤੀ_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->ਸ਼ੁਰੂ_example_ਸੂਚਕਾਂਕ = examples_ਪ੍ਰਤੀ_process * loader->process_rank;
    loader->ਸਿਰਾ_example_ਸੂਚਕਾਂਕ = examples_ਪ੍ਰਤੀ_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    ਜੇ (loader->ਸਿਰਾ_example_ਸੂਚਕਾਂਕ > loader->ਅੰਕ_examples) {
        loader->ਸਿਰਾ_example_ਸੂਚਕਾਂਕ = loader->ਅੰਕ_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t ਸਿਰਲੇਖ_ਬਾਈਟਾਂ = ਸਿਰਲੇਖ_ਮਾਪ * ਆਕਾਰ(ਪੂਰਨ);
    fseekCheck(loader->eval_file, (ਪੂਰਨ) ਸਿਰਲੇਖ_ਬਾਈਟਾਂ, SEEK_SET);
    ਲਈ (ਪੂਰਨ i = 0; i < loader->ਸ਼ੁਰੂ_example_ਸੂਚਕਾਂਕ; i++) {
        uint16_t example_ਸਿਰਲੇਖ[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_ਸਿਰਲੇਖ[0], ਆਕਾਰ(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        ਪ੍ਰਤਿਜ੍ਞਾ(example_ਸਿਰਲੇਖ[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        ਪ੍ਰਤਿਜ੍ਞਾ(example_ਸਿਰਲੇਖ[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t ਬਚਿਆ_ਬਾਈਟਾਂ = example_ਸਿਰਲੇਖ[1] - ਆਕਾਰ(uint16_t) * 3;
        ਪ੍ਰਤਿਜ੍ਞਾ(ਬਚਿਆ_ਬਾਈਟਾਂ > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (ਪੂਰਨ) ਬਚਿਆ_ਬਾਈਟਾਂ, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_ਸੂਚਕਾਂਕ = loader->ਸ਼ੁਰੂ_example_ਸੂਚਕਾਂਕ;
}

ਖਾਲੀ evalloader_init(EvalLoader *loader,
                     ਅਚਲ ਅੱਖਰ* filename,
                     size_t B,
                     size_t T,
                     ਪੂਰਨ process_rank,
                     ਪੂਰਨ ਅੰਕ_processes) {
    loader->process_rank = process_rank;
    loader->ਅੰਕ_processes = ਅੰਕ_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    ਪੂਰਨ ਸਿਰਲੇਖ[ਸਿਰਲੇਖ_ਮਾਪ];
    freadCheck(ਸਿਰਲੇਖ, ਆਕਾਰ(ਪੂਰਨ), ਸਿਰਲੇਖ_ਮਾਪ, loader->eval_file);
    ਜੇ (ਸਿਰਲੇਖ[0] != 20240522) { ਲਿਖੋ("Bad magic in eval file\n"); ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE); }
    ਜੇ (ਸਿਰਲੇਖ[1] != 1) { ਲਿਖੋ("Bad version in data file\n"); ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE); }
    loader->ਅੰਕ_examples = ਸਿਰਲੇਖ[2]; // number of examples in the file
    ਪ੍ਰਤਿਜ੍ਞਾ(loader->ਅੰਕ_examples >= ਅੰਕ_processes); // avoid headaches for now
    size_t longest_example_ਬਾਈਟਾਂ = ਸਿਰਲੇਖ[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    ਪ੍ਰਤਿਜ੍ਞਾ(longest_example_ਬਾਈਟਾਂ > 0 && longest_example_ਬਾਈਟਾਂ < (1+ASSUMED_ਅੰਕ_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    ਪੂਰਨ can_fit_examples = (ਪੂਰਨ) (B / ASSUMED_ਅੰਕ_COMPLETIONS);
    loader->ਬਫਰ = (uint16_t*)mallocCheck(longest_example_ਬਾਈਟਾਂ);
    loader->inputs = (ਪੂਰਨ*)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T, ਆਕਾਰ(ਪੂਰਨ));
    loader->targets = (ਪੂਰਨ*)ਸਿਫਰ_ਸਮ੍ਰਿਤੀ_ਦਿਓ(B * T, ਆਕਾਰ(ਪੂਰਨ));
    loader->ਨਕਾਬ = (ਅੱਖਰ*)mallocCheck(B * T * ਆਕਾਰ(ਅੱਖਰ));
    loader->label = (ਪੂਰਨ*)mallocCheck(can_fit_examples * ਆਕਾਰ(ਪੂਰਨ));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

ਖਾਲੀ evalloader_ਅਗਲਾ_example(EvalLoader *loader, ਪੂਰਨ example_batch_ਸੂਚਕਾਂਕ) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    ਪੂਰਨ batch_dim_ਵਿੱਥ = example_batch_ਸੂਚਕਾਂਕ * ASSUMED_ਅੰਕ_COMPLETIONS;
    // read the current example header
    uint16_t example_ਸਿਰਲੇਖ[3];
    freadCheck(&example_ਸਿਰਲੇਖ[0], ਆਕਾਰ(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    ਪ੍ਰਤਿਜ੍ਞਾ(example_ਸਿਰਲੇਖ[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    ਪ੍ਰਤਿਜ੍ਞਾ(example_ਸਿਰਲੇਖ[2] == loader->current_example_ਸੂਚਕਾਂਕ); // <EXAMPLE_INDEX> should match the loop index
    ਪ੍ਰਤਿਜ੍ਞਾ(example_ਸਿਰਲੇਖ[2] >= loader->ਸ਼ੁਰੂ_example_ਸੂਚਕਾਂਕ && example_ਸਿਰਲੇਖ[2] < loader->ਸਿਰਾ_example_ਸੂਚਕਾਂਕ);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_ਬਾਈਟਾਂ = example_ਸਿਰਲੇਖ[1] - ਆਕਾਰ(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->ਬਫਰ, ਆਕਾਰ(ਅੱਖਰ), example_ਬਾਈਟਾਂ, loader->eval_file);
    // process the example label
    ਪੂਰਨ label = (ਪੂਰਨ)loader->ਬਫਰ[0];
    ਪੂਰਨ can_fit_examples = (ਪੂਰਨ) (loader->B / ASSUMED_ਅੰਕ_COMPLETIONS);
    ਪ੍ਰਤਿਜ੍ਞਾ(label >= 0 && label < ASSUMED_ਅੰਕ_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    ਪ੍ਰਤਿਜ੍ਞਾ(example_batch_ਸੂਚਕਾਂਕ >= 0 && example_batch_ਸੂਚਕਾਂਕ < can_fit_examples);
    loader->label[example_batch_ਸੂਚਕਾਂਕ] = label; // store for output
    // process the number of completions
    ਪੂਰਨ ਅੰਕ_completions = (ਪੂਰਨ)loader->ਬਫਰ[1];
    ਪ੍ਰਤਿਜ੍ਞਾ(ਅੰਕ_completions == ASSUMED_ਅੰਕ_COMPLETIONS); // we expect 4 completions for now
    ਪ੍ਰਤਿਜ੍ਞਾ(batch_dim_ਵਿੱਥ + ਅੰਕ_completions <= B); // we expect to fit in the batch
    loader->ਅੰਕ_completions = ਅੰਕ_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    ਪੂਰਨ context_ਲੰਮਾਈ = (ਪੂਰਨ)loader->ਬਫਰ[2];
    uint16_t *context_ਟੋਕਨ_ਸ਼ੁਰੂ = &loader->ਬਫਰ[3]; // where the tokens start
    ਪ੍ਰਤਿਜ੍ਞਾ(context_ਲੰਮਾਈ > 0 && context_ਲੰਮਾਈ < T); // context is non-empty and up to T
    ਲਈ (ਪੂਰਨ b = 0; b < ਅੰਕ_completions; b++) {
        ਲਈ (ਪੂਰਨ i = 0; i < context_ਲੰਮਾਈ; i++) {
            ਪੂਰਨ boff = batch_dim_ਵਿੱਥ + b;
            ਪੂਰਨ tok_cur = (ਪੂਰਨ)context_ਟੋਕਨ_ਸ਼ੁਰੂ[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->ਬਫਰ + 3 + context_ਲੰਮਾਈ;
    ਲਈ (ਪੂਰਨ c = 0; c < ਅੰਕ_completions; c++) {
        ਪੂਰਨ coff = batch_dim_ਵਿੱਥ + c;
        ਪੂਰਨ completion_ਲੰਮਾਈ = (ਪੂਰਨ)completions_iter[0];
        uint16_t *completion_ਟੋਕਨ_ਸ਼ੁਰੂ = completions_iter + 1;
        ਪ੍ਰਤਿਜ੍ਞਾ(completion_ਲੰਮਾਈ > 0 && context_ਲੰਮਾਈ + completion_ਲੰਮਾਈ < T); // things fit?
        ਲਈ (ਪੂਰਨ i = 0; i < completion_ਲੰਮਾਈ; i++) {
            ਪੂਰਨ tok_cur = (ਪੂਰਨ)completion_ਟੋਕਨ_ਸ਼ੁਰੂ[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_ਲੰਮਾਈ + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_ਲੰਮਾਈ + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->ਨਕਾਬ[coff * T + context_ਲੰਮਾਈ + i - 1] = 1;
        }
        completions_iter += 1 + completion_ਲੰਮਾਈ; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_ਸੂਚਕਾਂਕ += 1;
}

ਖਾਲੀ evalloader_ਅਗਲਾ_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->ਨਕਾਬ, 0, B * T * ਆਕਾਰ(ਅੱਖਰ));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    ਪੂਰਨ can_fit_examples = (ਪੂਰਨ) (B / ASSUMED_ਅੰਕ_COMPLETIONS); // how many examples can we fit in the batch?
    ਲਈ (ਪੂਰਨ i = 0; i < can_fit_examples; i++) {
        ਜੇ (loader->current_example_ਸੂਚਕਾਂਕ >= loader->ਸਿਰਾ_example_ਸੂਚਕਾਂਕ) {
            ਰੋਕੋ; // this process has exhausted its work, noop from here on
        }
        evalloader_ਅਗਲਾ_example(loader, i);
    }
}

ਪੂਰਨ evalloader_stat_losses(EvalLoader *loader, ਦਸ਼ਮਲਵ* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    ਪੂਰਨ correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    ਪੂਰਨ can_fit_examples = (ਪੂਰਨ) (B / ASSUMED_ਅੰਕ_COMPLETIONS);
    ਲਈ (ਪੂਰਨ i = 0; i < can_fit_examples; i++) {
        ਦਸ਼ਮਲਵ min_loss = 0.0f;
        ਪੂਰਨ min_loss_ਸੂਚਕਾਂਕ = -1;
        ਅੱਖਰ ਸਰਗਰਮ = 0; // is this example active or fully empty?
        // iterate the completions in this example
        ਲਈ (ਪੂਰਨ b = 0; b < ASSUMED_ਅੰਕ_COMPLETIONS; b++) {
            ਪੂਰਨ boff = i * ASSUMED_ਅੰਕ_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            ਦਸ਼ਮਲਵ ਔਸਤਨ_loss = 0.0f;
            ਪੂਰਨ ਗਿਣਤੀ_2 = 0;
            ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
                ਅੱਖਰ ਨਕਾਬ = loader->ਨਕਾਬ[boff * T + t];
                ਜੇ (ਨਕਾਬ == 1) {
                    ਸਰਗਰਮ = 1;
                    ਔਸਤਨ_loss += losses[boff * T + t];
                    ਗਿਣਤੀ_2++;
                }
            }
            ਜੇ (ਗਿਣਤੀ_2 > 0) { ਔਸਤਨ_loss /= ਗਿਣਤੀ_2; }
            ਜੇ (b == 0 || ਔਸਤਨ_loss < min_loss) {
                min_loss = ਔਸਤਨ_loss;
                min_loss_ਸੂਚਕਾਂਕ = b;
            }
        }
        ਜੇ (ਸਰਗਰਮ && (min_loss_ਸੂਚਕਾਂਕ == loader->label[i])) {
            correct += 1;
        }
    }
    ਵਾਪਸ correct;
}

ਖਾਲੀ evalloader_free(EvalLoader *loader) {
    ਮੁਕਤ(loader->ਬਫਰ);
    ਮੁਕਤ(loader->inputs);
    ਮੁਕਤ(loader->targets);
    ਮੁਕਤ(loader->ਨਕਾਬ);
    ਮੁਕਤ(loader->label);
    fcloseCheck(loader->eval_file);
}

#ਜੇ_ਅੰਤ // DATALOADER_H