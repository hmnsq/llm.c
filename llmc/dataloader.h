/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#अन्तर्भाव "utils.h"
#अन्तर्भाव "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#अन्तर्भाव <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

प्रकारनिर्देश संरचना {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    पूर्णाङ्क process_rank;
    पूर्णाङ्क अङ्क_processes;
    // batch and token information
    माप_प्रकार B;
    माप_प्रकार T;
    माप_प्रकार अङ्क_tokens; // total number of tokens
    माप_प्रकार shard_अङ्क_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    माप_प्रकार current_shard_idx; // the current shard we are reading from
    माप_प्रकार current_sample_idx; // the current sample we are reading from
    // file handle
    खाता* tokens_file;
    // data buffers
    uint16_t* buffer; // we fread data from file into this buffer
    पूर्णाङ्क* inputs;  // input tokens into transformer
    पूर्णाङ्क* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    पूर्णाङ्क should_shuffle;
    पूर्णाङ्क* shard_indices;
    पूर्णाङ्क* intra_shard_indices;
    // sizes in bytes
    माप_प्रकार total_batch_size_bytes;  // total across all processes
    माप_प्रकार local_batch_offset_bytes;  // inner-sample offset for this process
    माप_प्रकार header_bytes;  // header size in bytes
    int64_t file_size_bytes;
} DataLoader;

int64_t dataloader_load_shard_(DataLoader *loader, पूर्णाङ्क shard_index) {
    यदि (loader->should_shuffle) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    अचर वर्ण* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    यदि (loader->tokens_file != रिक्त) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    पूर्णाङ्क header[HEADER_SIZE];
    freadCheck(header, परिमाण(पूर्णाङ्क), HEADER_SIZE, loader->tokens_file);
    यदि (header[0] != 20240520) {
        म_लिखो("Bad magic in the data file\n");
        म_लिखो("---> HINT: Are you passing in a correct file?\n");
        म_लिखो("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        निकास(निकास_त्रुटि);
    }
    यदि (header[1] != 1) { म_लिखो("Bad version in data file\n"); निकास(निकास_त्रुटि); }
    int64_t ntok = header[2]; // number of tokens in the file
    प्रतिज्ञा(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, अंत_से); // seek to end of file
    loader->file_size_bytes = ख_बताओ(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, शुरू_से); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = HEADER_SIZE * परिमाण(पूर्णाङ्क) + ntok * परिमाण(uint16_t);
    यदि (loader->file_size_bytes != expected_file_size) {
        म_लिखो("Error: file size is not as expected\n");
        निकास(निकास_त्रुटि);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_अङ्क_samples = (ntok * परिमाण(uint16_t) - परिमाण(uint16_t)) / loader->total_batch_size_bytes;
    फर्काऊ ntok;
}

शून्य prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    यदि (loader->intra_shard_indices != रिक्त) {
        // in case shards have different number of samples / sizes
        मुक्त(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (पूर्णाङ्क*)mallocCheck(loader->shard_अङ्क_samples * परिमाण(पूर्णाङ्क));
    init_identity_permutation(loader->intra_shard_indices, (पूर्णाङ्क) loader->shard_अङ्क_samples);
    random_permutation(loader->intra_shard_indices, (पूर्णाङ्क) loader->shard_अङ्क_samples, &loader->shuffle_rng);
}

शून्य dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    यदि (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (पूर्णाङ्क) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (पूर्णाङ्क) loader->current_shard_idx);

    यदि (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

शून्य dataloader_advance_(DataLoader *loader) {
    यदि (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        फर्काऊ;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (पूर्णाङ्क) loader->current_shard_idx);

    यदि (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

शून्य dataloader_init(DataLoader *loader,
                     अचर वर्ण* filename_pattern,
                     माप_प्रकार B,
                     माप_प्रकार T,
                     पूर्णाङ्क process_rank,
                     पूर्णाङ्क अङ्क_processes,
                     पूर्णाङ्क should_shuffle) {
    loader->process_rank = process_rank;
    loader->अङ्क_processes = अङ्क_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = रिक्त;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_SIZE * परिमाण(पूर्णाङ्क);
    loader->total_batch_size_bytes = ((loader->अङ्क_processes * (loader->B * loader->T)) * परिमाण(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * परिमाण(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    पूर्णाङ्क glob_status = glob(filename_pattern, 0, रिक्त, &loader->glob_result);
    यदि (glob_status != 0) {
        म_लिखो("Error: failed to glob pattern: %s\n", filename_pattern);
        निकास(निकास_त्रुटि);
    }
    यदि (loader->glob_result.gl_pathc == 0) {
        म_लिखो("Error: no files found matching the pattern: %s\n", filename_pattern);
        निकास(निकास_त्रुटि);
    }

    यदि (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (पूर्णाङ्क*)mallocCheck(loader->glob_result.gl_pathc * परिमाण(पूर्णाङ्क));
        init_identity_permutation(loader->shard_indices, (पूर्णाङ्क) loader->glob_result.gl_pathc);
        loader->intra_shard_indices = रिक्त;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_total = 0;
    लागि (पूर्णाङ्क shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        प्रतिज्ञा(shard_ntok >= (int64_t) (अङ्क_processes * B * T + 1));
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = (uint16_t*)mallocCheck((B * T + 1) * परिमाण(uint16_t));
    loader->inputs = (पूर्णाङ्क*)mallocCheck(B * T * परिमाण(पूर्णाङ्क));
    loader->targets = (पूर्णाङ्क*)mallocCheck(B * T * परिमाण(पूर्णाङ्क));
    loader->अङ्क_tokens = ntok_total;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

शून्य dataloader_load_batch(DataLoader* loader) {
    प्रतिज्ञा(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != रिक्त));
    प्रतिज्ञा(loader->current_sample_idx < loader->shard_अङ्क_samples);
    माप_प्रकार idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    माप_प्रकार global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, (पूर्णाङ्क) current_offset, शुरू_से);
    freadCheck(loader->buffer, परिमाण(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    लागि (पूर्णाङ्क i = 0; i < B*T; i++) {
        loader->inputs[i] = (पूर्णाङ्क)loader->buffer[i];
        loader->targets[i] = (पूर्णाङ्क)loader->buffer[i+1];
    }
}

शून्य dataloader_next_batch(DataLoader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    यदि (loader->current_sample_idx >= loader->shard_अङ्क_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


शून्य dataloader_resume(DataLoader *loader, माप_प्रकार current_shard_idx, माप_प्रकार current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (पूर्णाङ्क) loader->current_shard_idx);
}

शून्य dataloader_free(DataLoader *loader) {
    मुक्त(loader->buffer);
    मुक्त(loader->inputs);
    मुक्त(loader->targets);
    यदि (loader->should_shuffle) {
        मुक्त(loader->shard_indices);
        मुक्त(loader->intra_shard_indices);
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
#define ASSUMED_अङ्क_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

प्रकारनिर्देश संरचना {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    पूर्णाङ्क process_rank;
    पूर्णाङ्क अङ्क_processes;
    // hyperparameters. use size_t to prevent overflow
    माप_प्रकार B; // (micro) batch size dimension of the tensor that feeds into the model
    माप_प्रकार T; // maximum context length of the model
    // input handling and its state
    खाता* eval_file;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    पूर्णाङ्क अङ्क_examples; // in total across all processes
    पूर्णाङ्क अङ्क_batches; // to process the entire dataset across all processes
    पूर्णाङ्क start_example_index; // the assignment of work for this process, start
    पूर्णाङ्क end_example_index; // and end. start is inclusive, end is exclusive
    पूर्णाङ्क current_example_index; // the next example we would read
    पूर्णाङ्क* inputs;  // input tokens into transformer
    पूर्णाङ्क* targets; // target tokens for the transformer
    वर्ण* mask; // mask=1 at all completion token locations
    पूर्णाङ्क* label; // the correct completion labels
    पूर्णाङ्क अङ्क_completions; // number of completions for this example
} EvalLoader;

शून्य evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    पूर्णाङ्क examples_per_process = CEIL_DIV(loader->अङ्क_examples, loader->अङ्क_processes);
    पूर्णाङ्क can_fit_examples = (पूर्णाङ्क) (loader->B / ASSUMED_अङ्क_COMPLETIONS);
    यदि (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        म_लिखो("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_अङ्क_COMPLETIONS);
        म_लिखो("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        निकास(निकास_त्रुटि);
    }
    loader->अङ्क_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    यदि (loader->end_example_index > loader->अङ्क_examples) {
        loader->end_example_index = loader->अङ्क_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * परिमाण(पूर्णाङ्क);
    fseekCheck(loader->eval_file, (पूर्णाङ्क) header_bytes, शुरू_से);
    लागि (पूर्णाङ्क i = 0; i < loader->start_example_index; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], परिमाण(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        प्रतिज्ञा(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        प्रतिज्ञा(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        माप_प्रकार remaining_bytes = example_header[1] - परिमाण(uint16_t) * 3;
        प्रतिज्ञा(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (पूर्णाङ्क) remaining_bytes, प्रस्तुत_से);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

शून्य evalloader_init(EvalLoader *loader,
                     अचर वर्ण* filename,
                     माप_प्रकार B,
                     माप_प्रकार T,
                     पूर्णाङ्क process_rank,
                     पूर्णाङ्क अङ्क_processes) {
    loader->process_rank = process_rank;
    loader->अङ्क_processes = अङ्क_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    पूर्णाङ्क header[HEADER_SIZE];
    freadCheck(header, परिमाण(पूर्णाङ्क), HEADER_SIZE, loader->eval_file);
    यदि (header[0] != 20240522) { म_लिखो("Bad magic in eval file\n"); निकास(निकास_त्रुटि); }
    यदि (header[1] != 1) { म_लिखो("Bad version in data file\n"); निकास(निकास_त्रुटि); }
    loader->अङ्क_examples = header[2]; // number of examples in the file
    प्रतिज्ञा(loader->अङ्क_examples >= अङ्क_processes); // avoid headaches for now
    माप_प्रकार longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    प्रतिज्ञा(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_अङ्क_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    पूर्णाङ्क can_fit_examples = (पूर्णाङ्क) (B / ASSUMED_अङ्क_COMPLETIONS);
    loader->buffer = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (पूर्णाङ्क*)सुस्मृति(B * T, परिमाण(पूर्णाङ्क));
    loader->targets = (पूर्णाङ्क*)सुस्मृति(B * T, परिमाण(पूर्णाङ्क));
    loader->mask = (वर्ण*)mallocCheck(B * T * परिमाण(वर्ण));
    loader->label = (पूर्णाङ्क*)mallocCheck(can_fit_examples * परिमाण(पूर्णाङ्क));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

शून्य evalloader_next_example_(EvalLoader *loader, पूर्णाङ्क example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    पूर्णाङ्क batch_dim_offset = example_batch_index * ASSUMED_अङ्क_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], परिमाण(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    प्रतिज्ञा(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    प्रतिज्ञा(example_header[2] == loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    प्रतिज्ञा(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    माप_प्रकार example_bytes = example_header[1] - परिमाण(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, परिमाण(वर्ण), example_bytes, loader->eval_file);
    // process the example label
    पूर्णाङ्क label = (पूर्णाङ्क)loader->buffer[0];
    पूर्णाङ्क can_fit_examples = (पूर्णाङ्क) (loader->B / ASSUMED_अङ्क_COMPLETIONS);
    प्रतिज्ञा(label >= 0 && label < ASSUMED_अङ्क_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    प्रतिज्ञा(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    पूर्णाङ्क अङ्क_completions = (पूर्णाङ्क)loader->buffer[1];
    प्रतिज्ञा(अङ्क_completions == ASSUMED_अङ्क_COMPLETIONS); // we expect 4 completions for now
    प्रतिज्ञा(batch_dim_offset + अङ्क_completions <= B); // we expect to fit in the batch
    loader->अङ्क_completions = अङ्क_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    पूर्णाङ्क context_length = (पूर्णाङ्क)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3]; // where the tokens start
    प्रतिज्ञा(context_length > 0 && context_length < T); // context is non-empty and up to T
    लागि (पूर्णाङ्क b = 0; b < अङ्क_completions; b++) {
        लागि (पूर्णाङ्क i = 0; i < context_length; i++) {
            पूर्णाङ्क boff = batch_dim_offset + b;
            पूर्णाङ्क tok_cur = (पूर्णाङ्क)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->buffer + 3 + context_length;
    लागि (पूर्णाङ्क c = 0; c < अङ्क_completions; c++) {
        पूर्णाङ्क coff = batch_dim_offset + c;
        पूर्णाङ्क completion_length = (पूर्णाङ्क)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        प्रतिज्ञा(completion_length > 0 && context_length + completion_length < T); // things fit?
        लागि (पूर्णाङ्क i = 0; i < completion_length; i++) {
            पूर्णाङ्क tok_cur = (पूर्णाङ्क)completion_tokens_start[i];
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

शून्य evalloader_next_batch(EvalLoader *loader) {
    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    स_रखो(loader->mask, 0, B * T * परिमाण(वर्ण));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    पूर्णाङ्क can_fit_examples = (पूर्णाङ्क) (B / ASSUMED_अङ्क_COMPLETIONS); // how many examples can we fit in the batch?
    लागि (पूर्णाङ्क i = 0; i < can_fit_examples; i++) {
        यदि (loader->current_example_index >= loader->end_example_index) {
            रोक; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

पूर्णाङ्क evalloader_stat_losses(EvalLoader *loader, भिन्न* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    पूर्णाङ्क correct = 0;
    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    // iterate the examples in this batch
    पूर्णाङ्क can_fit_examples = (पूर्णाङ्क) (B / ASSUMED_अङ्क_COMPLETIONS);
    लागि (पूर्णाङ्क i = 0; i < can_fit_examples; i++) {
        भिन्न min_loss = 0.0f;
        पूर्णाङ्क min_loss_index = -1;
        वर्ण active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        लागि (पूर्णाङ्क b = 0; b < ASSUMED_अङ्क_COMPLETIONS; b++) {
            पूर्णाङ्क boff = i * ASSUMED_अङ्क_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            भिन्न average_loss = 0.0f;
            पूर्णाङ्क गन्ती = 0;
            लागि (पूर्णाङ्क t = 0; t < T; t++) {
                वर्ण mask = loader->mask[boff * T + t];
                यदि (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    गन्ती++;
                }
            }
            यदि (गन्ती > 0) { average_loss /= गन्ती; }
            यदि (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        यदि (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    फर्काऊ correct;
}

शून्य evalloader_free(EvalLoader *loader) {
    मुक्त(loader->buffer);
    मुक्त(loader->inputs);
    मुक्त(loader->targets);
    मुक्त(loader->mask);
    मुक्त(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H