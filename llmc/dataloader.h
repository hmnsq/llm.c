/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

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
#ifndef _WIN32
#ಅನ್ತರ್ಭಾವ <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    ಪೂರ್ಣಾಙ್ಕ process_rank;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t ಅಂಕೆ_tokens; // total number of tokens
    size_t shard_ಅಂಕೆ_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    uint16_t* buffer; // we fread data from file into this buffer
    ಪೂರ್ಣಾಙ್ಕ* inputs;  // input tokens into transformer
    ಪೂರ್ಣಾಙ್ಕ* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    ಪೂರ್ಣಾಙ್ಕ should_shuffle;
    ಪೂರ್ಣಾಙ್ಕ* shard_indices;
    ಪೂರ್ಣಾಙ್ಕ* intra_shard_indices;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_size_bytes;
} DataLoader;

int64_t dataloader_load_shard_(DataLoader *loader, ಪೂರ್ಣಾಙ್ಕ shard_index) {
    ಯದಿ (loader->should_shuffle) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    ಧ್ರುವ ವರ್ಣ* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    ಯದಿ (loader->tokens_file != ರಿಕ್ತ) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    ಪೂರ್ಣಾಙ್ಕ header[HEADER_SIZE];
    freadCheck(header, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), HEADER_SIZE, loader->tokens_file);
    ಯದಿ (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    ಯದಿ (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    int64_t ntok = header[2]; // number of tokens in the file
    ಪ್ರತಿಜ್ಞಾ(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_size_bytes = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = HEADER_SIZE * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) + ntok * ಪರಿಮಾಣ(uint16_t);
    ಯದಿ (loader->file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_ಅಂಕೆ_samples = (ntok * ಪರಿಮಾಣ(uint16_t) - ಪರಿಮಾಣ(uint16_t)) / loader->total_batch_size_bytes;
    ಪ್ರತ್ಯಾಗಮ ntok;
}

ಶೂನ್ಯ prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    ಯದಿ (loader->intra_shard_indices != ರಿಕ್ತ) {
        // in case shards have different number of samples / sizes
        ಮುಕ್ತ(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (ಪೂರ್ಣಾಙ್ಕ*)mallocCheck(loader->shard_ಅಂಕೆ_samples * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    init_identity_permutation(loader->intra_shard_indices, (ಪೂರ್ಣಾಙ್ಕ) loader->shard_ಅಂಕೆ_samples);
    random_permutation(loader->intra_shard_indices, (ಪೂರ್ಣಾಙ್ಕ) loader->shard_ಅಂಕೆ_samples, &loader->shuffle_rng);
}

ಶೂನ್ಯ dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    ಯದಿ (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (ಪೂರ್ಣಾಙ್ಕ) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (ಪೂರ್ಣಾಙ್ಕ) loader->current_shard_idx);

    ಯದಿ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

ಶೂನ್ಯ dataloader_advance_(DataLoader *loader) {
    ಯದಿ (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        ಪ್ರತ್ಯಾಗಮ;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (ಪೂರ್ಣಾಙ್ಕ) loader->current_shard_idx);

    ಯದಿ (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

ಶೂನ್ಯ dataloader_init(DataLoader *loader,
                     ಧ್ರುವ ವರ್ಣ* filename_pattern,
                     size_t B,
                     size_t T,
                     ಪೂರ್ಣಾಙ್ಕ process_rank,
                     ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes,
                     ಪೂರ್ಣಾಙ್ಕ should_shuffle) {
    loader->process_rank = process_rank;
    loader->ಅಂಕೆ_processes = ಅಂಕೆ_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = ರಿಕ್ತ;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_SIZE * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ);
    loader->total_batch_size_bytes = ((loader->ಅಂಕೆ_processes * (loader->B * loader->T)) * ಪರಿಮಾಣ(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * ಪರಿಮಾಣ(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    ಪೂರ್ಣಾಙ್ಕ glob_status = glob(filename_pattern, 0, ರಿಕ್ತ, &loader->glob_result);
    ಯದಿ (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    ಯದಿ (loader->glob_result.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    ಯದಿ (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (ಪೂರ್ಣಾಙ್ಕ*)mallocCheck(loader->glob_result.gl_pathc * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
        init_identity_permutation(loader->shard_indices, (ಪೂರ್ಣಾಙ್ಕ) loader->glob_result.gl_pathc);
        loader->intra_shard_indices = ರಿಕ್ತ;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_total = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        ಪ್ರತಿಜ್ಞಾ(shard_ntok >= (int64_t) (ಅಂಕೆ_processes * B * T + 1));
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = (uint16_t*)mallocCheck((B * T + 1) * ಪರಿಮಾಣ(uint16_t));
    loader->inputs = (ಪೂರ್ಣಾಙ್ಕ*)mallocCheck(B * T * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    loader->targets = (ಪೂರ್ಣಾಙ್ಕ*)mallocCheck(B * T * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    loader->ಅಂಕೆ_tokens = ntok_total;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

ಶೂನ್ಯ dataloader_load_batch(DataLoader* loader) {
    ಪ್ರತಿಜ್ಞಾ(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != ರಿಕ್ತ));
    ಪ್ರತಿಜ್ಞಾ(loader->current_sample_idx < loader->shard_ಅಂಕೆ_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, (ಪೂರ್ಣಾಙ್ಕ) current_offset, SEEK_SET);
    freadCheck(loader->buffer, ಪರಿಮಾಣ(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < B*T; i++) {
        loader->inputs[i] = (ಪೂರ್ಣಾಙ್ಕ)loader->buffer[i];
        loader->targets[i] = (ಪೂರ್ಣಾಙ್ಕ)loader->buffer[i+1];
    }
}

ಶೂನ್ಯ dataloader_next_batch(DataLoader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    ಯದಿ (loader->current_sample_idx >= loader->shard_ಅಂಕೆ_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


ಶೂನ್ಯ dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (ಪೂರ್ಣಾಙ್ಕ) loader->current_shard_idx);
}

ಶೂನ್ಯ dataloader_free(DataLoader *loader) {
    ಮುಕ್ತ(loader->buffer);
    ಮುಕ್ತ(loader->inputs);
    ಮುಕ್ತ(loader->targets);
    ಯದಿ (loader->should_shuffle) {
        ಮುಕ್ತ(loader->shard_indices);
        ಮುಕ್ತ(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_result);
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
#define ASSUMED_ಅಂಕೆ_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    ಪೂರ್ಣಾಙ್ಕ process_rank;
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_examples; // in total across all processes
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_batches; // to process the entire dataset across all processes
    ಪೂರ್ಣಾಙ್ಕ start_example_index; // the assignment of work for this process, start
    ಪೂರ್ಣಾಙ್ಕ end_example_index; // and end. start is inclusive, end is exclusive
    ಪೂರ್ಣಾಙ್ಕ current_example_index; // the next example we would read
    ಪೂರ್ಣಾಙ್ಕ* inputs;  // input tokens into transformer
    ಪೂರ್ಣಾಙ್ಕ* targets; // target tokens for the transformer
    ವರ್ಣ* mask; // mask=1 at all completion token locations
    ಪೂರ್ಣಾಙ್ಕ* label; // the correct completion labels
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_completions; // number of completions for this example
} EvalLoader;

ಶೂನ್ಯ evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    ಪೂರ್ಣಾಙ್ಕ examples_per_process = CEIL_DIV(loader->ಅಂಕೆ_examples, loader->ಅಂಕೆ_processes);
    ಪೂರ್ಣಾಙ್ಕ can_fit_examples = (ಪೂರ್ಣಾಙ್ಕ) (loader->B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    ಯದಿ (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_ಅಂಕೆ_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->ಅಂಕೆ_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    ಯದಿ (loader->end_example_index > loader->ಅಂಕೆ_examples) {
        loader->end_example_index = loader->ಅಂಕೆ_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ);
    fseekCheck(loader->eval_file, (ಪೂರ್ಣಾಙ್ಕ) header_bytes, SEEK_SET);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < loader->start_example_index; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], ಪರಿಮಾಣ(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        ಪ್ರತಿಜ್ಞಾ(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        ಪ್ರತಿಜ್ಞಾ(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - ಪರಿಮಾಣ(uint16_t) * 3;
        ಪ್ರತಿಜ್ಞಾ(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (ಪೂರ್ಣಾಙ್ಕ) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

ಶೂನ್ಯ evalloader_init(EvalLoader *loader,
                     ಧ್ರುವ ವರ್ಣ* filename,
                     size_t B,
                     size_t T,
                     ಪೂರ್ಣಾಙ್ಕ process_rank,
                     ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_processes) {
    loader->process_rank = process_rank;
    loader->ಅಂಕೆ_processes = ಅಂಕೆ_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    ಪೂರ್ಣಾಙ್ಕ header[HEADER_SIZE];
    freadCheck(header, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ), HEADER_SIZE, loader->eval_file);
    ಯದಿ (header[0] != 20240522) { printf("Bad magic in eval file\n"); exit(EXIT_FAILURE); }
    ಯದಿ (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    loader->ಅಂಕೆ_examples = header[2]; // number of examples in the file
    ಪ್ರತಿಜ್ಞಾ(loader->ಅಂಕೆ_examples >= ಅಂಕೆ_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    ಪ್ರತಿಜ್ಞಾ(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_ಅಂಕೆ_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    ಪೂರ್ಣಾಙ್ಕ can_fit_examples = (ಪೂರ್ಣಾಙ್ಕ) (B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    loader->buffer = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (ಪೂರ್ಣಾಙ್ಕ*)calloc(B * T, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    loader->targets = (ಪೂರ್ಣಾಙ್ಕ*)calloc(B * T, ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));
    loader->mask = (ವರ್ಣ*)mallocCheck(B * T * ಪರಿಮಾಣ(ವರ್ಣ));
    loader->label = (ಪೂರ್ಣಾಙ್ಕ*)mallocCheck(can_fit_examples * ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

ಶೂನ್ಯ evalloader_next_example_(EvalLoader *loader, ಪೂರ್ಣಾಙ್ಕ example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    ಪೂರ್ಣಾಙ್ಕ batch_dim_offset = example_batch_index * ASSUMED_ಅಂಕೆ_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], ಪರಿಮಾಣ(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    ಪ್ರತಿಜ್ಞಾ(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    ಪ್ರತಿಜ್ಞಾ(example_header[2] == loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    ಪ್ರತಿಜ್ಞಾ(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_bytes = example_header[1] - ಪರಿಮಾಣ(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, ಪರಿಮಾಣ(ವರ್ಣ), example_bytes, loader->eval_file);
    // process the example label
    ಪೂರ್ಣಾಙ್ಕ label = (ಪೂರ್ಣಾಙ್ಕ)loader->buffer[0];
    ಪೂರ್ಣಾಙ್ಕ can_fit_examples = (ಪೂರ್ಣಾಙ್ಕ) (loader->B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    ಪ್ರತಿಜ್ಞಾ(label >= 0 && label < ASSUMED_ಅಂಕೆ_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    ಪ್ರತಿಜ್ಞಾ(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_completions = (ಪೂರ್ಣಾಙ್ಕ)loader->buffer[1];
    ಪ್ರತಿಜ್ಞಾ(ಅಂಕೆ_completions == ASSUMED_ಅಂಕೆ_COMPLETIONS); // we expect 4 completions for now
    ಪ್ರತಿಜ್ಞಾ(batch_dim_offset + ಅಂಕೆ_completions <= B); // we expect to fit in the batch
    loader->ಅಂಕೆ_completions = ಅಂಕೆ_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    ಪೂರ್ಣಾಙ್ಕ context_length = (ಪೂರ್ಣಾಙ್ಕ)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3]; // where the tokens start
    ಪ್ರತಿಜ್ಞಾ(context_length > 0 && context_length < T); // context is non-empty and up to T
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < ಅಂಕೆ_completions; b++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < context_length; i++) {
            ಪೂರ್ಣಾಙ್ಕ boff = batch_dim_offset + b;
            ಪೂರ್ಣಾಙ್ಕ tok_cur = (ಪೂರ್ಣಾಙ್ಕ)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->buffer + 3 + context_length;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c = 0; c < ಅಂಕೆ_completions; c++) {
        ಪೂರ್ಣಾಙ್ಕ coff = batch_dim_offset + c;
        ಪೂರ್ಣಾಙ್ಕ completion_length = (ಪೂರ್ಣಾಙ್ಕ)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        ಪ್ರತಿಜ್ಞಾ(completion_length > 0 && context_length + completion_length < T); // things fit?
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < completion_length; i++) {
            ಪೂರ್ಣಾಙ್ಕ tok_cur = (ಪೂರ್ಣಾಙ್ಕ)completion_tokens_start[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_length + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_length + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_length + i - 1] = 1;
        }
        completions_iter += 1 + completion_length; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_index += 1;
}

ಶೂನ್ಯ evalloader_next_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * ಪರಿಮಾಣ(ವರ್ಣ));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    ಪೂರ್ಣಾಙ್ಕ can_fit_examples = (ಪೂರ್ಣಾಙ್ಕ) (B / ASSUMED_ಅಂಕೆ_COMPLETIONS); // how many examples can we fit in the batch?
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < can_fit_examples; i++) {
        ಯದಿ (loader->current_example_index >= loader->end_example_index) {
            ಭಙ್ಗ; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

ಪೂರ್ಣಾಙ್ಕ evalloader_stat_losses(EvalLoader *loader, ಭಿನ್ನ* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    ಪೂರ್ಣಾಙ್ಕ correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    ಪೂರ್ಣಾಙ್ಕ can_fit_examples = (ಪೂರ್ಣಾಙ್ಕ) (B / ASSUMED_ಅಂಕೆ_COMPLETIONS);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < can_fit_examples; i++) {
        ಭಿನ್ನ min_loss = 0.0f;
        ಪೂರ್ಣಾಙ್ಕ min_loss_index = -1;
        ವರ್ಣ active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < ASSUMED_ಅಂಕೆ_COMPLETIONS; b++) {
            ಪೂರ್ಣಾಙ್ಕ boff = i * ASSUMED_ಅಂಕೆ_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            ಭಿನ್ನ average_loss = 0.0f;
            ಪೂರ್ಣಾಙ್ಕ ಎಣಿಕೆ = 0;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < T; t++) {
                ವರ್ಣ mask = loader->mask[boff * T + t];
                ಯದಿ (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    ಎಣಿಕೆ++;
                }
            }
            ಯದಿ (ಎಣಿಕೆ > 0) { average_loss /= ಎಣಿಕೆ; }
            ಯದಿ (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        ಯದಿ (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    ಪ್ರತ್ಯಾಗಮ correct;
}

ಶೂನ್ಯ evalloader_free(EvalLoader *loader) {
    ಮುಕ್ತ(loader->buffer);
    ಮುಕ್ತ(loader->inputs);
    ಮುಕ್ತ(loader->targets);
    ಮುಕ್ತ(loader->mask);
    ಮುಕ್ತ(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H