/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#அந்தர்பாவ "utils.h"
#அந்தர்பாவ "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#அந்தர்பாவ <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

வகைவரை கட்டமைப்பு {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    முழு process_rank;
    முழு இலக்கம்_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t இலக்கம்_tokens; // total number of tokens
    size_t shard_இலக்கம்_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    uint16_t* buffer; // we fread data from file into this buffer
    முழு* inputs;  // input tokens into transformer
    முழு* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    முழு should_shuffle;
    முழு* shard_indices;
    முழு* intra_shard_indices;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_size_bytes;
} DataLoader;

int64_t dataloader_load_shard_(DataLoader *loader, முழு shard_index) {
    எனில் (loader->should_shuffle) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    மாறிலி எழுத்து* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    எனில் (loader->tokens_file != வெற்று) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    முழு header[HEADER_SIZE];
    freadCheck(header, அளவு(முழு), HEADER_SIZE, loader->tokens_file);
    எனில் (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    எனில் (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    int64_t ntok = header[2]; // number of tokens in the file
    ப்ரதிஜ்ஞா(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_size_bytes = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = HEADER_SIZE * அளவு(முழு) + ntok * அளவு(uint16_t);
    எனில் (loader->file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_இலக்கம்_samples = (ntok * அளவு(uint16_t) - அளவு(uint16_t)) / loader->total_batch_size_bytes;
    திரும்பு ntok;
}

வெறுமை prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    எனில் (loader->intra_shard_indices != வெற்று) {
        // in case shards have different number of samples / sizes
        விடுவி(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (முழு*)mallocCheck(loader->shard_இலக்கம்_samples * அளவு(முழு));
    init_identity_permutation(loader->intra_shard_indices, (முழு) loader->shard_இலக்கம்_samples);
    random_permutation(loader->intra_shard_indices, (முழு) loader->shard_இலக்கம்_samples, &loader->shuffle_rng);
}

வெறுமை dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    எனில் (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (முழு) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (முழு) loader->current_shard_idx);

    எனில் (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

வெறுமை dataloader_advance_(DataLoader *loader) {
    எனில் (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        திரும்பு;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (முழு) loader->current_shard_idx);

    எனில் (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

வெறுமை dataloader_init(DataLoader *loader,
                     மாறிலி எழுத்து* filename_pattern,
                     size_t B,
                     size_t T,
                     முழு process_rank,
                     முழு இலக்கம்_processes,
                     முழு should_shuffle) {
    loader->process_rank = process_rank;
    loader->இலக்கம்_processes = இலக்கம்_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = வெற்று;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_SIZE * அளவு(முழு);
    loader->total_batch_size_bytes = ((loader->இலக்கம்_processes * (loader->B * loader->T)) * அளவு(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * அளவு(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    முழு glob_status = glob(filename_pattern, 0, வெற்று, &loader->glob_result);
    எனில் (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    எனில் (loader->glob_result.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    எனில் (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (முழு*)mallocCheck(loader->glob_result.gl_pathc * அளவு(முழு));
        init_identity_permutation(loader->shard_indices, (முழு) loader->glob_result.gl_pathc);
        loader->intra_shard_indices = வெற்று;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_total = 0;
    க்கு (முழு shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        ப்ரதிஜ்ஞா(shard_ntok >= (int64_t) (இலக்கம்_processes * B * T + 1));
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = (uint16_t*)mallocCheck((B * T + 1) * அளவு(uint16_t));
    loader->inputs = (முழு*)mallocCheck(B * T * அளவு(முழு));
    loader->targets = (முழு*)mallocCheck(B * T * அளவு(முழு));
    loader->இலக்கம்_tokens = ntok_total;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

வெறுமை dataloader_load_batch(DataLoader* loader) {
    ப்ரதிஜ்ஞா(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != வெற்று));
    ப்ரதிஜ்ஞா(loader->current_sample_idx < loader->shard_இலக்கம்_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, (முழு) current_offset, SEEK_SET);
    freadCheck(loader->buffer, அளவு(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    க்கு (முழு i = 0; i < B*T; i++) {
        loader->inputs[i] = (முழு)loader->buffer[i];
        loader->targets[i] = (முழு)loader->buffer[i+1];
    }
}

வெறுமை dataloader_next_batch(DataLoader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    எனில் (loader->current_sample_idx >= loader->shard_இலக்கம்_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


வெறுமை dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (முழு) loader->current_shard_idx);
}

வெறுமை dataloader_free(DataLoader *loader) {
    விடுவி(loader->buffer);
    விடுவி(loader->inputs);
    விடுவி(loader->targets);
    எனில் (loader->should_shuffle) {
        விடுவி(loader->shard_indices);
        விடுவி(loader->intra_shard_indices);
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
#define ASSUMED_இலக்கம்_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

வகைவரை கட்டமைப்பு {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    முழு process_rank;
    முழு இலக்கம்_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    முழு இலக்கம்_examples; // in total across all processes
    முழு இலக்கம்_batches; // to process the entire dataset across all processes
    முழு start_example_index; // the assignment of work for this process, start
    முழு end_example_index; // and end. start is inclusive, end is exclusive
    முழு current_example_index; // the next example we would read
    முழு* inputs;  // input tokens into transformer
    முழு* targets; // target tokens for the transformer
    எழுத்து* mask; // mask=1 at all completion token locations
    முழு* label; // the correct completion labels
    முழு இலக்கம்_completions; // number of completions for this example
} EvalLoader;

வெறுமை evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    முழு examples_per_process = CEIL_DIV(loader->இலக்கம்_examples, loader->இலக்கம்_processes);
    முழு can_fit_examples = (முழு) (loader->B / ASSUMED_இலக்கம்_COMPLETIONS);
    எனில் (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_இலக்கம்_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->இலக்கம்_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    எனில் (loader->end_example_index > loader->இலக்கம்_examples) {
        loader->end_example_index = loader->இலக்கம்_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * அளவு(முழு);
    fseekCheck(loader->eval_file, (முழு) header_bytes, SEEK_SET);
    க்கு (முழு i = 0; i < loader->start_example_index; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], அளவு(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        ப்ரதிஜ்ஞா(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        ப்ரதிஜ்ஞா(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - அளவு(uint16_t) * 3;
        ப்ரதிஜ்ஞா(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (முழு) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

வெறுமை evalloader_init(EvalLoader *loader,
                     மாறிலி எழுத்து* filename,
                     size_t B,
                     size_t T,
                     முழு process_rank,
                     முழு இலக்கம்_processes) {
    loader->process_rank = process_rank;
    loader->இலக்கம்_processes = இலக்கம்_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    முழு header[HEADER_SIZE];
    freadCheck(header, அளவு(முழு), HEADER_SIZE, loader->eval_file);
    எனில் (header[0] != 20240522) { printf("Bad magic in eval file\n"); exit(EXIT_FAILURE); }
    எனில் (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    loader->இலக்கம்_examples = header[2]; // number of examples in the file
    ப்ரதிஜ்ஞா(loader->இலக்கம்_examples >= இலக்கம்_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    ப்ரதிஜ்ஞா(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_இலக்கம்_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    முழு can_fit_examples = (முழு) (B / ASSUMED_இலக்கம்_COMPLETIONS);
    loader->buffer = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (முழு*)calloc(B * T, அளவு(முழு));
    loader->targets = (முழு*)calloc(B * T, அளவு(முழு));
    loader->mask = (எழுத்து*)mallocCheck(B * T * அளவு(எழுத்து));
    loader->label = (முழு*)mallocCheck(can_fit_examples * அளவு(முழு));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

வெறுமை evalloader_next_example_(EvalLoader *loader, முழு example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    முழு batch_dim_offset = example_batch_index * ASSUMED_இலக்கம்_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], அளவு(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    ப்ரதிஜ்ஞா(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    ப்ரதிஜ்ஞா(example_header[2] == loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    ப்ரதிஜ்ஞா(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_bytes = example_header[1] - அளவு(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, அளவு(எழுத்து), example_bytes, loader->eval_file);
    // process the example label
    முழு label = (முழு)loader->buffer[0];
    முழு can_fit_examples = (முழு) (loader->B / ASSUMED_இலக்கம்_COMPLETIONS);
    ப்ரதிஜ்ஞா(label >= 0 && label < ASSUMED_இலக்கம்_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    ப்ரதிஜ்ஞா(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    முழு இலக்கம்_completions = (முழு)loader->buffer[1];
    ப்ரதிஜ்ஞா(இலக்கம்_completions == ASSUMED_இலக்கம்_COMPLETIONS); // we expect 4 completions for now
    ப்ரதிஜ்ஞா(batch_dim_offset + இலக்கம்_completions <= B); // we expect to fit in the batch
    loader->இலக்கம்_completions = இலக்கம்_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    முழு context_length = (முழு)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3]; // where the tokens start
    ப்ரதிஜ்ஞா(context_length > 0 && context_length < T); // context is non-empty and up to T
    க்கு (முழு b = 0; b < இலக்கம்_completions; b++) {
        க்கு (முழு i = 0; i < context_length; i++) {
            முழு boff = batch_dim_offset + b;
            முழு tok_cur = (முழு)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->buffer + 3 + context_length;
    க்கு (முழு c = 0; c < இலக்கம்_completions; c++) {
        முழு coff = batch_dim_offset + c;
        முழு completion_length = (முழு)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        ப்ரதிஜ்ஞா(completion_length > 0 && context_length + completion_length < T); // things fit?
        க்கு (முழு i = 0; i < completion_length; i++) {
            முழு tok_cur = (முழு)completion_tokens_start[i];
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

வெறுமை evalloader_next_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * அளவு(எழுத்து));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    முழு can_fit_examples = (முழு) (B / ASSUMED_இலக்கம்_COMPLETIONS); // how many examples can we fit in the batch?
    க்கு (முழு i = 0; i < can_fit_examples; i++) {
        எனில் (loader->current_example_index >= loader->end_example_index) {
            நிறுத்து; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

முழு evalloader_stat_losses(EvalLoader *loader, தசம* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    முழு correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    முழு can_fit_examples = (முழு) (B / ASSUMED_இலக்கம்_COMPLETIONS);
    க்கு (முழு i = 0; i < can_fit_examples; i++) {
        தசம min_loss = 0.0f;
        முழு min_loss_index = -1;
        எழுத்து active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        க்கு (முழு b = 0; b < ASSUMED_இலக்கம்_COMPLETIONS; b++) {
            முழு boff = i * ASSUMED_இலக்கம்_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            தசம average_loss = 0.0f;
            முழு மொத்தம் = 0;
            க்கு (முழு t = 0; t < T; t++) {
                எழுத்து mask = loader->mask[boff * T + t];
                எனில் (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    மொத்தம்++;
                }
            }
            எனில் (மொத்தம் > 0) { average_loss /= மொத்தம்; }
            எனில் (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        எனில் (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    திரும்பு correct;
}

வெறுமை evalloader_free(EvalLoader *loader) {
    விடுவி(loader->buffer);
    விடுவி(loader->inputs);
    விடுவி(loader->targets);
    விடுவி(loader->mask);
    விடுவி(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H