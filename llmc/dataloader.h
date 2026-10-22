/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ DATALOADER_H
#ವ್ಯಾಖ್ಯಾನಿಸು DATALOADER_H

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#ಅನ್ತರ್ಭಾವ "utils.h"
#ಅನ್ತರ್ಭಾವ "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ _WIN32
#ಅನ್ತರ್ಭಾವ <glob.h>
#ಆದರೆ_ಅಂತ್ಯ
// ----------------------------------------------------------------------------
// Distributed Data Loader
#ವ್ಯಾಖ್ಯಾನಿಸು ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ 256

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    ಪೂರ್ಣಾಂಕ process_rank;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t ಅಂಕೆ_ಟೋಕನುಗಳು; // total number of tokens
    size_t shard_ಅಂಕೆ_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_ಫಲಿತಾಂಶ; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* ಟೋಕನುಗಳು_file;
    // data buffers
    uint16_t* ಬಫರ್; // we fread data from file into this buffer
    ಪೂರ್ಣಾಂಕ* inputs;  // input tokens into transformer
    ಪೂರ್ಣಾಂಕ* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    ಪೂರ್ಣಾಂಕ should_shuffle;
    ಪೂರ್ಣಾಂಕ* shard_indices;
    ಪೂರ್ಣಾಂಕ* intra_shard_indices;
    // sizes in bytes
    size_t ಒಟ್ಟು_batch_ಗಾತ್ರ_ಬೈಟುಗಳು;  // total across all processes
    size_t local_batch_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು;  // inner-sample offset for this process
    size_t ಶೀರ್ಷಿಕೆ_ಬೈಟುಗಳು;  // header size in bytes
    int64_t file_ಗಾತ್ರ_ಬೈಟುಗಳು;
} ದತ್ತಾಂಶ_Loader;

int64_t dataloader_ತುಂಬು_shard(ದತ್ತಾಂಶ_Loader *loader, ಪೂರ್ಣಾಂಕ shard_ಸೂಚ್ಯಂಕ) {
    ಆದರೆ (loader->should_shuffle) {
        shard_ಸೂಚ್ಯಂಕ = loader->shard_indices[shard_ಸೂಚ್ಯಂಕ];
    }
    // use the first glob match as the filename for now
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* filename = loader->glob_ಫಲಿತಾಂಶ.gl_pathv[shard_ಸೂಚ್ಯಂಕ];
    // open the input file for reading. also only a single file can be opened at a time
    ಆದರೆ (loader->ಟೋಕನುಗಳು_file != ಖಾಲಿ) {
        fcloseCheck(loader->ಟೋಕನುಗಳು_file);
    }
    loader->ಟೋಕನುಗಳು_file = fopenCheck(filename, "rb");
    // validate the header
    ಪೂರ್ಣಾಂಕ ಶೀರ್ಷಿಕೆ[ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ];
    freadCheck(ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ, loader->ಟೋಕನುಗಳು_file);
    ಆದರೆ (ಶೀರ್ಷಿಕೆ[0] != 20240520) {
        ಬರೆ("Bad magic in the data file\n");
        ಬರೆ("---> HINT: Are you passing in a correct file?\n");
        ಬರೆ("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    ಆದರೆ (ಶೀರ್ಷಿಕೆ[1] != 1) { ಬರೆ("Bad version in data file\n"); ನಿರ್ಗಮಿಸು(EXIT_FAILURE); }
    int64_t ntok = ಶೀರ್ಷಿಕೆ[2]; // number of tokens in the file
    ಪ್ರತಿಜ್ಞಾ(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->ಟೋಕನುಗಳು_file, 0, SEEK_END); // seek to end of file
    loader->file_ಗಾತ್ರ_ಬೈಟುಗಳು = ftell(loader->ಟೋಕನುಗಳು_file); // read the offset, i.e. file size
    fseekCheck(loader->ಟೋಕನುಗಳು_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_ಗಾತ್ರ = ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) + ntok * ಗಾತ್ರ(uint16_t);
    ಆದರೆ (loader->file_ಗಾತ್ರ_ಬೈಟುಗಳು != expected_file_ಗಾತ್ರ) {
        ಬರೆ("Error: file size is not as expected\n");
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_ಅಂಕೆ_samples = (ntok * ಗಾತ್ರ(uint16_t) - ಗಾತ್ರ(uint16_t)) / loader->ಒಟ್ಟು_batch_ಗಾತ್ರ_ಬೈಟುಗಳು;
    ಹಿಂತಿರುಗಿ ntok;
}

ಶೂನ್ಯ prepare_intra_shard_indices_(ದತ್ತಾಂಶ_Loader *loader) {
    // shuffle the examples inside the shards
    ಆದರೆ (loader->intra_shard_indices != ಖಾಲಿ) {
        // in case shards have different number of samples / sizes
        ಬಿಡುಗಡೆ(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (ಪೂರ್ಣಾಂಕ*)mallocCheck(loader->shard_ಅಂಕೆ_samples * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    init_ತಟಸ್ಥ_permutation(loader->intra_shard_indices, (ಪೂರ್ಣಾಂಕ) loader->shard_ಅಂಕೆ_samples);
    random_permutation(loader->intra_shard_indices, (ಪೂರ್ಣಾಂಕ) loader->shard_ಅಂಕೆ_samples, &loader->shuffle_rng);
}

ಶೂನ್ಯ dataloader_reset(ದತ್ತಾಂಶ_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    ಆದರೆ (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (ಪೂರ್ಣಾಂಕ) loader->glob_ಫಲಿತಾಂಶ.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_ತುಂಬು_shard(loader, (ಪೂರ್ಣಾಂಕ) loader->current_shard_idx);

    ಆದರೆ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

ಶೂನ್ಯ dataloader_advance_(ದತ್ತಾಂಶ_Loader *loader) {
    ಆದರೆ (loader->current_shard_idx == loader->glob_ಫಲಿತಾಂಶ.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        ಹಿಂತಿರುಗಿ;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_ಫಲಿತಾಂಶ.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_ತುಂಬು_shard(loader, (ಪೂರ್ಣಾಂಕ) loader->current_shard_idx);

    ಆದರೆ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

ಶೂನ್ಯ dataloader_init(ದತ್ತಾಂಶ_Loader *loader,
                     ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* filename_ನಮೂನೆ,
                     size_t B,
                     size_t T,
                     ಪೂರ್ಣಾಂಕ process_rank,
                     ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes,
                     ಪೂರ್ಣಾಂಕ should_shuffle) {
    loader->process_rank = process_rank;
    loader->ಅಂಕೆ_processes = ಅಂಕೆ_processes;
    loader->B = B;
    loader->T = T;
    loader->ಟೋಕನುಗಳು_file = ಖಾಲಿ;
    loader->should_shuffle = should_shuffle;
    loader->ಶೀರ್ಷಿಕೆ_ಬೈಟುಗಳು = ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ);
    loader->ಒಟ್ಟು_batch_ಗಾತ್ರ_ಬೈಟುಗಳು = ((loader->ಅಂಕೆ_processes * (loader->B * loader->T)) * ಗಾತ್ರ(uint16_t));
    loader->local_batch_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು = loader->process_rank * loader->B * loader->T * ಗಾತ್ರ(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    ಪೂರ್ಣಾಂಕ glob_ಸ್ಥಿತಿ = glob(filename_ನಮೂನೆ, 0, ಖಾಲಿ, &loader->glob_ಫಲಿತಾಂಶ);
    ಆದರೆ (glob_ಸ್ಥಿತಿ != 0) {
        ಬರೆ("Error: failed to glob pattern: %s\n", filename_ನಮೂನೆ);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    ಆದರೆ (loader->glob_ಫಲಿತಾಂಶ.gl_pathc == 0) {
        ಬರೆ("Error: no files found matching the pattern: %s\n", filename_ನಮೂನೆ);
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }

    ಆದರೆ (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_ಬೀಜ(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (ಪೂರ್ಣಾಂಕ*)mallocCheck(loader->glob_ಫಲಿತಾಂಶ.gl_pathc * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
        init_ತಟಸ್ಥ_permutation(loader->shard_indices, (ಪೂರ್ಣಾಂಕ) loader->glob_ಫಲಿತಾಂಶ.gl_pathc);
        loader->intra_shard_indices = ಖಾಲಿ;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_ಒಟ್ಟು = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ shard_ಸೂಚ್ಯಂಕ = 0; shard_ಸೂಚ್ಯಂಕ < loader->glob_ಫಲಿತಾಂಶ.gl_pathc; shard_ಸೂಚ್ಯಂಕ++) {
        int64_t shard_ntok = dataloader_ತುಂಬು_shard(loader, shard_ಸೂಚ್ಯಂಕ);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        ಪ್ರತಿಜ್ಞಾ(shard_ntok >= (int64_t) (ಅಂಕೆ_processes * B * T + 1));
        ntok_ಒಟ್ಟು += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->ಬಫರ್ = (uint16_t*)mallocCheck((B * T + 1) * ಗಾತ್ರ(uint16_t));
    loader->inputs = (ಪೂರ್ಣಾಂಕ*)mallocCheck(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    loader->targets = (ಪೂರ್ಣಾಂಕ*)mallocCheck(B * T * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    loader->ಅಂಕೆ_ಟೋಕನುಗಳು = ntok_ಒಟ್ಟು;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

ಶೂನ್ಯ dataloader_ತುಂಬು_batch(ದತ್ತಾಂಶ_Loader* loader) {
    ಪ್ರತಿಜ್ಞಾ(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != ಖಾಲಿ));
    ಪ್ರತಿಜ್ಞಾ(loader->current_sample_idx < loader->shard_ಅಂಕೆ_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು = idx * loader->ಒಟ್ಟು_batch_ಗಾತ್ರ_ಬೈಟುಗಳು;
    int64_t current_ಸ್ಥಾನಾಂತರ = loader->ಶೀರ್ಷಿಕೆ_ಬೈಟುಗಳು + global_batch_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು + loader->local_batch_ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->ಟೋಕನುಗಳು_file, (ಪೂರ್ಣಾಂಕ) current_ಸ್ಥಾನಾಂತರ, SEEK_SET);
    freadCheck(loader->ಬಫರ್, ಗಾತ್ರ(uint16_t), B*T+1, loader->ಟೋಕನುಗಳು_file);
    // decode the buffer into inputs and targets (cast to int)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < B*T; i++) {
        loader->inputs[i] = (ಪೂರ್ಣಾಂಕ)loader->ಬಫರ್[i];
        loader->targets[i] = (ಪೂರ್ಣಾಂಕ)loader->ಬಫರ್[i+1];
    }
}

ಶೂನ್ಯ dataloader_ಮುಂದಿನ_batch(ದತ್ತಾಂಶ_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    ಆದರೆ (loader->current_sample_idx >= loader->shard_ಅಂಕೆ_samples) {
        dataloader_advance_(loader);
    }
    dataloader_ತುಂಬು_batch(loader);
    loader->current_sample_idx += 1;
}


ಶೂನ್ಯ dataloader_resume(ದತ್ತಾಂಶ_Loader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_ತುಂಬು_shard(loader, (ಪೂರ್ಣಾಂಕ) loader->current_shard_idx);
}

ಶೂನ್ಯ dataloader_free(ದತ್ತಾಂಶ_Loader *loader) {
    ಬಿಡುಗಡೆ(loader->ಬಫರ್);
    ಬಿಡುಗಡೆ(loader->inputs);
    ಬಿಡುಗಡೆ(loader->targets);
    ಆದರೆ (loader->should_shuffle) {
        ಬಿಡುಗಡೆ(loader->shard_indices);
        ಬಿಡುಗಡೆ(loader->intra_shard_indices);
    }
    fcloseCheck(loader->ಟೋಕನುಗಳು_file);
    globfree(&loader->glob_ಫಲಿತಾಂಶ);
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
#ವ್ಯಾಖ್ಯಾನಿಸು ASSUMED_ಅಂಕೆ_COMPLETIONS 4
// helper macro for ceildiv
#ವ್ಯಾಖ್ಯಾನಿಸು CEIL_DIV(M, N) (((M) + (N)-1) / (N))

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    ಪೂರ್ಣಾಂಕ process_rank;
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* ಬಫರ್; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_examples; // in total across all processes
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_batches; // to process the entire dataset across all processes
    ಪೂರ್ಣಾಂಕ ಆರಂಭ_example_ಸೂಚ್ಯಂಕ; // the assignment of work for this process, start
    ಪೂರ್ಣಾಂಕ ಕೊನೆ_example_ಸೂಚ್ಯಂಕ; // and end. start is inclusive, end is exclusive
    ಪೂರ್ಣಾಂಕ current_example_ಸೂಚ್ಯಂಕ; // the next example we would read
    ಪೂರ್ಣಾಂಕ* inputs;  // input tokens into transformer
    ಪೂರ್ಣಾಂಕ* targets; // target tokens for the transformer
    ಅಕ್ಷರ* ಮುಖವಾಡ; // mask=1 at all completion token locations
    ಪೂರ್ಣಾಂಕ* label; // the correct completion labels
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_completions; // number of completions for this example
} EvalLoader;

ಶೂನ್ಯ evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    ಪೂರ್ಣಾಂಕ examples_ಪ್ರತಿ_process = CEIL_DIV(loader->ಅಂಕೆ_examples, loader->ಅಂಕೆ_processes);
    ಪೂರ್ಣಾಂಕ can_fit_examples = (ಪೂರ್ಣಾಂಕ) (loader->B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    ಆದರೆ (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        ಬರೆ("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_ಅಂಕೆ_COMPLETIONS);
        ಬರೆ("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    loader->ಅಂಕೆ_batches = CEIL_DIV(examples_ಪ್ರತಿ_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->ಆರಂಭ_example_ಸೂಚ್ಯಂಕ = examples_ಪ್ರತಿ_process * loader->process_rank;
    loader->ಕೊನೆ_example_ಸೂಚ್ಯಂಕ = examples_ಪ್ರತಿ_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    ಆದರೆ (loader->ಕೊನೆ_example_ಸೂಚ್ಯಂಕ > loader->ಅಂಕೆ_examples) {
        loader->ಕೊನೆ_example_ಸೂಚ್ಯಂಕ = loader->ಅಂಕೆ_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t ಶೀರ್ಷಿಕೆ_ಬೈಟುಗಳು = ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ);
    fseekCheck(loader->eval_file, (ಪೂರ್ಣಾಂಕ) ಶೀರ್ಷಿಕೆ_ಬೈಟುಗಳು, SEEK_SET);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < loader->ಆರಂಭ_example_ಸೂಚ್ಯಂಕ; i++) {
        uint16_t example_ಶೀರ್ಷಿಕೆ[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_ಶೀರ್ಷಿಕೆ[0], ಗಾತ್ರ(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        ಪ್ರತಿಜ್ಞಾ(example_ಶೀರ್ಷಿಕೆ[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        ಪ್ರತಿಜ್ಞಾ(example_ಶೀರ್ಷಿಕೆ[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t ಉಳಿದ_ಬೈಟುಗಳು = example_ಶೀರ್ಷಿಕೆ[1] - ಗಾತ್ರ(uint16_t) * 3;
        ಪ್ರತಿಜ್ಞಾ(ಉಳಿದ_ಬೈಟುಗಳು > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (ಪೂರ್ಣಾಂಕ) ಉಳಿದ_ಬೈಟುಗಳು, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_ಸೂಚ್ಯಂಕ = loader->ಆರಂಭ_example_ಸೂಚ್ಯಂಕ;
}

ಶೂನ್ಯ evalloader_init(EvalLoader *loader,
                     ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* filename,
                     size_t B,
                     size_t T,
                     ಪೂರ್ಣಾಂಕ process_rank,
                     ಪೂರ್ಣಾಂಕ ಅಂಕೆ_processes) {
    loader->process_rank = process_rank;
    loader->ಅಂಕೆ_processes = ಅಂಕೆ_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    ಪೂರ್ಣಾಂಕ ಶೀರ್ಷಿಕೆ[ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ];
    freadCheck(ಶೀರ್ಷಿಕೆ, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ), ಶೀರ್ಷಿಕೆ_ಗಾತ್ರ, loader->eval_file);
    ಆದರೆ (ಶೀರ್ಷಿಕೆ[0] != 20240522) { ಬರೆ("Bad magic in eval file\n"); ನಿರ್ಗಮಿಸು(EXIT_FAILURE); }
    ಆದರೆ (ಶೀರ್ಷಿಕೆ[1] != 1) { ಬರೆ("Bad version in data file\n"); ನಿರ್ಗಮಿಸು(EXIT_FAILURE); }
    loader->ಅಂಕೆ_examples = ಶೀರ್ಷಿಕೆ[2]; // number of examples in the file
    ಪ್ರತಿಜ್ಞಾ(loader->ಅಂಕೆ_examples >= ಅಂಕೆ_processes); // avoid headaches for now
    size_t longest_example_ಬೈಟುಗಳು = ಶೀರ್ಷಿಕೆ[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    ಪ್ರತಿಜ್ಞಾ(longest_example_ಬೈಟುಗಳು > 0 && longest_example_ಬೈಟುಗಳು < (1+ASSUMED_ಅಂಕೆ_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    ಪೂರ್ಣಾಂಕ can_fit_examples = (ಪೂರ್ಣಾಂಕ) (B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    loader->ಬಫರ್ = (uint16_t*)mallocCheck(longest_example_ಬೈಟುಗಳು);
    loader->inputs = (ಪೂರ್ಣಾಂಕ*)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(B * T, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    loader->targets = (ಪೂರ್ಣಾಂಕ*)ಸೊನ್ನೆ_ಸ್ಮರಣೆ_ಕೊಡು(B * T, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));
    loader->ಮುಖವಾಡ = (ಅಕ್ಷರ*)mallocCheck(B * T * ಗಾತ್ರ(ಅಕ್ಷರ));
    loader->label = (ಪೂರ್ಣಾಂಕ*)mallocCheck(can_fit_examples * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

ಶೂನ್ಯ evalloader_ಮುಂದಿನ_example(EvalLoader *loader, ಪೂರ್ಣಾಂಕ example_batch_ಸೂಚ್ಯಂಕ) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    ಪೂರ್ಣಾಂಕ batch_dim_ಸ್ಥಾನಾಂತರ = example_batch_ಸೂಚ್ಯಂಕ * ASSUMED_ಅಂಕೆ_COMPLETIONS;
    // read the current example header
    uint16_t example_ಶೀರ್ಷಿಕೆ[3];
    freadCheck(&example_ಶೀರ್ಷಿಕೆ[0], ಗಾತ್ರ(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    ಪ್ರತಿಜ್ಞಾ(example_ಶೀರ್ಷಿಕೆ[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    ಪ್ರತಿಜ್ಞಾ(example_ಶೀರ್ಷಿಕೆ[2] == loader->current_example_ಸೂಚ್ಯಂಕ); // <EXAMPLE_INDEX> should match the loop index
    ಪ್ರತಿಜ್ಞಾ(example_ಶೀರ್ಷಿಕೆ[2] >= loader->ಆರಂಭ_example_ಸೂಚ್ಯಂಕ && example_ಶೀರ್ಷಿಕೆ[2] < loader->ಕೊನೆ_example_ಸೂಚ್ಯಂಕ);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_ಬೈಟುಗಳು = example_ಶೀರ್ಷಿಕೆ[1] - ಗಾತ್ರ(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->ಬಫರ್, ಗಾತ್ರ(ಅಕ್ಷರ), example_ಬೈಟುಗಳು, loader->eval_file);
    // process the example label
    ಪೂರ್ಣಾಂಕ label = (ಪೂರ್ಣಾಂಕ)loader->ಬಫರ್[0];
    ಪೂರ್ಣಾಂಕ can_fit_examples = (ಪೂರ್ಣಾಂಕ) (loader->B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    ಪ್ರತಿಜ್ಞಾ(label >= 0 && label < ASSUMED_ಅಂಕೆ_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    ಪ್ರತಿಜ್ಞಾ(example_batch_ಸೂಚ್ಯಂಕ >= 0 && example_batch_ಸೂಚ್ಯಂಕ < can_fit_examples);
    loader->label[example_batch_ಸೂಚ್ಯಂಕ] = label; // store for output
    // process the number of completions
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_completions = (ಪೂರ್ಣಾಂಕ)loader->ಬಫರ್[1];
    ಪ್ರತಿಜ್ಞಾ(ಅಂಕೆ_completions == ASSUMED_ಅಂಕೆ_COMPLETIONS); // we expect 4 completions for now
    ಪ್ರತಿಜ್ಞಾ(batch_dim_ಸ್ಥಾನಾಂತರ + ಅಂಕೆ_completions <= B); // we expect to fit in the batch
    loader->ಅಂಕೆ_completions = ಅಂಕೆ_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    ಪೂರ್ಣಾಂಕ context_ಉದ್ದಳತೆ = (ಪೂರ್ಣಾಂಕ)loader->ಬಫರ್[2];
    uint16_t *context_ಟೋಕನುಗಳು_ಆರಂಭ = &loader->ಬಫರ್[3]; // where the tokens start
    ಪ್ರತಿಜ್ಞಾ(context_ಉದ್ದಳತೆ > 0 && context_ಉದ್ದಳತೆ < T); // context is non-empty and up to T
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < ಅಂಕೆ_completions; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < context_ಉದ್ದಳತೆ; i++) {
            ಪೂರ್ಣಾಂಕ boff = batch_dim_ಸ್ಥಾನಾಂತರ + b;
            ಪೂರ್ಣಾಂಕ tok_cur = (ಪೂರ್ಣಾಂಕ)context_ಟೋಕನುಗಳು_ಆರಂಭ[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->ಬಫರ್ + 3 + context_ಉದ್ದಳತೆ;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < ಅಂಕೆ_completions; c++) {
        ಪೂರ್ಣಾಂಕ coff = batch_dim_ಸ್ಥಾನಾಂತರ + c;
        ಪೂರ್ಣಾಂಕ completion_ಉದ್ದಳತೆ = (ಪೂರ್ಣಾಂಕ)completions_iter[0];
        uint16_t *completion_ಟೋಕನುಗಳು_ಆರಂಭ = completions_iter + 1;
        ಪ್ರತಿಜ್ಞಾ(completion_ಉದ್ದಳತೆ > 0 && context_ಉದ್ದಳತೆ + completion_ಉದ್ದಳತೆ < T); // things fit?
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < completion_ಉದ್ದಳತೆ; i++) {
            ಪೂರ್ಣಾಂಕ tok_cur = (ಪೂರ್ಣಾಂಕ)completion_ಟೋಕನುಗಳು_ಆರಂಭ[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_ಉದ್ದಳತೆ + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_ಉದ್ದಳತೆ + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->ಮುಖವಾಡ[coff * T + context_ಉದ್ದಳತೆ + i - 1] = 1;
        }
        completions_iter += 1 + completion_ಉದ್ದಳತೆ; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_ಸೂಚ್ಯಂಕ += 1;
}

ಶೂನ್ಯ evalloader_ಮುಂದಿನ_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->ಮುಖವಾಡ, 0, B * T * ಗಾತ್ರ(ಅಕ್ಷರ));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    ಪೂರ್ಣಾಂಕ can_fit_examples = (ಪೂರ್ಣಾಂಕ) (B / ASSUMED_ಅಂಕೆ_COMPLETIONS); // how many examples can we fit in the batch?
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < can_fit_examples; i++) {
        ಆದರೆ (loader->current_example_ಸೂಚ್ಯಂಕ >= loader->ಕೊನೆ_example_ಸೂಚ್ಯಂಕ) {
            ನಿಲ್ಲಿಸು; // this process has exhausted its work, noop from here on
        }
        evalloader_ಮುಂದಿನ_example(loader, i);
    }
}

ಪೂರ್ಣಾಂಕ evalloader_stat_losses(EvalLoader *loader, ದಶಮಾಂಶ* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    ಪೂರ್ಣಾಂಕ correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    ಪೂರ್ಣಾಂಕ can_fit_examples = (ಪೂರ್ಣಾಂಕ) (B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < can_fit_examples; i++) {
        ದಶಮಾಂಶ min_loss = 0.0f;
        ಪೂರ್ಣಾಂಕ min_loss_ಸೂಚ್ಯಂಕ = -1;
        ಅಕ್ಷರ ಸಕ್ರಿಯ = 0; // is this example active or fully empty?
        // iterate the completions in this example
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < ASSUMED_ಅಂಕೆ_COMPLETIONS; b++) {
            ಪೂರ್ಣಾಂಕ boff = i * ASSUMED_ಅಂಕೆ_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            ದಶಮಾಂಶ ಸರಾಸರಿ_ಮೌಲ್ಯ_loss = 0.0f;
            ಪೂರ್ಣಾಂಕ ಎಣಿಕೆ_2 = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
                ಅಕ್ಷರ ಮುಖವಾಡ = loader->ಮುಖವಾಡ[boff * T + t];
                ಆದರೆ (ಮುಖವಾಡ == 1) {
                    ಸಕ್ರಿಯ = 1;
                    ಸರಾಸರಿ_ಮೌಲ್ಯ_loss += losses[boff * T + t];
                    ಎಣಿಕೆ_2++;
                }
            }
            ಆದರೆ (ಎಣಿಕೆ_2 > 0) { ಸರಾಸರಿ_ಮೌಲ್ಯ_loss /= ಎಣಿಕೆ_2; }
            ಆದರೆ (b == 0 || ಸರಾಸರಿ_ಮೌಲ್ಯ_loss < min_loss) {
                min_loss = ಸರಾಸರಿ_ಮೌಲ್ಯ_loss;
                min_loss_ಸೂಚ್ಯಂಕ = b;
            }
        }
        ಆದರೆ (ಸಕ್ರಿಯ && (min_loss_ಸೂಚ್ಯಂಕ == loader->label[i])) {
            correct += 1;
        }
    }
    ಹಿಂತಿರುಗಿ correct;
}

ಶೂನ್ಯ evalloader_free(EvalLoader *loader) {
    ಬಿಡುಗಡೆ(loader->ಬಫರ್);
    ಬಿಡುಗಡೆ(loader->inputs);
    ಬಿಡುಗಡೆ(loader->targets);
    ಬಿಡುಗಡೆ(loader->ಮುಖವಾಡ);
    ಬಿಡುಗಡೆ(loader->label);
    fcloseCheck(loader->eval_file);
}

#ಆದರೆ_ಅಂತ್ಯ // DATALOADER_H