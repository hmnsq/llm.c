/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#अगर_अघोषित DATALOADER_H
#घोषणा DATALOADER_H

#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <stddef.h>
#समावेश <stdint.h>
#समावेश <assert.h>
#समावेश <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#समावेश "utils.h"
#समावेश "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#अगर_अघोषित _WIN32
#समावेश <glob.h>
#पूर्ण_अगर
// ----------------------------------------------------------------------------
// Distributed Data Loader
#घोषणा HEADER_आकार 256

प्रकार काष्ठा {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    पूर्णांक process_rank;
    पूर्णांक अंक_processes;
    // batch and token information
    माप_प्रकार B;
    माप_प्रकार T;
    माप_प्रकार अंक_tokens; // total number of tokens
    माप_प्रकार shard_अंक_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_परिणाम; // stores the result of glob, for all shards we want to iterate
    माप_प्रकार current_shard_idx; // the current shard we are reading from
    माप_प्रकार current_sample_idx; // the current sample we are reading from
    // file handle
    खाता* tokens_file;
    // data buffers
    uint16_t* बफर; // we fread data from file into this buffer
    पूर्णांक* inputs;  // input tokens into transformer
    पूर्णांक* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    पूर्णांक should_shuffle;
    पूर्णांक* shard_indices;
    पूर्णांक* intra_shard_indices;
    // sizes in bytes
    माप_प्रकार योग_batch_आकार_bytes;  // total across all processes
    माप_प्रकार local_batch_offset_bytes;  // inner-sample offset for this process
    माप_प्रकार header_bytes;  // header size in bytes
    int64_t file_आकार_bytes;
} तथ्य_Loader;

int64_t dataloader_load_shard_(तथ्य_Loader *loader, पूर्णांक shard_सूचकांक) {
    अगर (loader->should_shuffle) {
        shard_सूचकांक = loader->shard_indices[shard_सूचकांक];
    }
    // use the first glob match as the filename for now
    स्थिर अक्षर* filename = loader->glob_परिणाम.gl_pathv[shard_सूचकांक];
    // open the input file for reading. also only a single file can be opened at a time
    अगर (loader->tokens_file != शून्य) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    पूर्णांक header[HEADER_आकार];
    freadCheck(header, माप(पूर्णांक), HEADER_आकार, loader->tokens_file);
    अगर (header[0] != 20240520) {
        म_लिखो("Bad magic in the data file\n");
        म_लिखो("---> HINT: Are you passing in a correct file?\n");
        म_लिखो("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        निकास(निकास_त्रुटि);
    }
    अगर (header[1] != 1) { म_लिखो("Bad version in data file\n"); निकास(निकास_त्रुटि); }
    int64_t ntok = header[2]; // number of tokens in the file
    निश्चित(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, अंत_से); // seek to end of file
    loader->file_आकार_bytes = ख_बताओ(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, शुरू_से); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_आकार = HEADER_आकार * माप(पूर्णांक) + ntok * माप(uint16_t);
    अगर (loader->file_आकार_bytes != expected_file_आकार) {
        म_लिखो("Error: file size is not as expected\n");
        निकास(निकास_त्रुटि);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_अंक_samples = (ntok * माप(uint16_t) - माप(uint16_t)) / loader->योग_batch_आकार_bytes;
    वापस ntok;
}

व्योम prepare_intra_shard_indices_(तथ्य_Loader *loader) {
    // shuffle the examples inside the shards
    अगर (loader->intra_shard_indices != शून्य) {
        // in case shards have different number of samples / sizes
        मुक्त(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (पूर्णांक*)mallocCheck(loader->shard_अंक_samples * माप(पूर्णांक));
    init_identity_permutation(loader->intra_shard_indices, (पूर्णांक) loader->shard_अंक_samples);
    random_permutation(loader->intra_shard_indices, (पूर्णांक) loader->shard_अंक_samples, &loader->shuffle_rng);
}

व्योम dataloader_reset(तथ्य_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    अगर (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (पूर्णांक) loader->glob_परिणाम.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (पूर्णांक) loader->current_shard_idx);

    अगर (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

व्योम dataloader_advance_(तथ्य_Loader *loader) {
    अगर (loader->current_shard_idx == loader->glob_परिणाम.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        वापस;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_परिणाम.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (पूर्णांक) loader->current_shard_idx);

    अगर (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

व्योम dataloader_init(तथ्य_Loader *loader,
                     स्थिर अक्षर* filename_pattern,
                     माप_प्रकार B,
                     माप_प्रकार T,
                     पूर्णांक process_rank,
                     पूर्णांक अंक_processes,
                     पूर्णांक should_shuffle) {
    loader->process_rank = process_rank;
    loader->अंक_processes = अंक_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = शून्य;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_आकार * माप(पूर्णांक);
    loader->योग_batch_आकार_bytes = ((loader->अंक_processes * (loader->B * loader->T)) * माप(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * माप(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    पूर्णांक glob_status = glob(filename_pattern, 0, शून्य, &loader->glob_परिणाम);
    अगर (glob_status != 0) {
        म_लिखो("Error: failed to glob pattern: %s\n", filename_pattern);
        निकास(निकास_त्रुटि);
    }
    अगर (loader->glob_परिणाम.gl_pathc == 0) {
        म_लिखो("Error: no files found matching the pattern: %s\n", filename_pattern);
        निकास(निकास_त्रुटि);
    }

    अगर (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (पूर्णांक*)mallocCheck(loader->glob_परिणाम.gl_pathc * माप(पूर्णांक));
        init_identity_permutation(loader->shard_indices, (पूर्णांक) loader->glob_परिणाम.gl_pathc);
        loader->intra_shard_indices = शून्य;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_योग = 0;
    क्रम (पूर्णांक shard_सूचकांक = 0; shard_सूचकांक < loader->glob_परिणाम.gl_pathc; shard_सूचकांक++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_सूचकांक);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        निश्चित(shard_ntok >= (int64_t) (अंक_processes * B * T + 1));
        ntok_योग += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->बफर = (uint16_t*)mallocCheck((B * T + 1) * माप(uint16_t));
    loader->inputs = (पूर्णांक*)mallocCheck(B * T * माप(पूर्णांक));
    loader->targets = (पूर्णांक*)mallocCheck(B * T * माप(पूर्णांक));
    loader->अंक_tokens = ntok_योग;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

व्योम dataloader_load_batch(तथ्य_Loader* loader) {
    निश्चित(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != शून्य));
    निश्चित(loader->current_sample_idx < loader->shard_अंक_samples);
    माप_प्रकार idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    माप_प्रकार global_batch_offset_bytes = idx * loader->योग_batch_आकार_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, (पूर्णांक) current_offset, शुरू_से);
    freadCheck(loader->बफर, माप(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    क्रम (पूर्णांक i = 0; i < B*T; i++) {
        loader->inputs[i] = (पूर्णांक)loader->बफर[i];
        loader->targets[i] = (पूर्णांक)loader->बफर[i+1];
    }
}

व्योम dataloader_अग्र_batch(तथ्य_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    अगर (loader->current_sample_idx >= loader->shard_अंक_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


व्योम dataloader_resume(तथ्य_Loader *loader, माप_प्रकार current_shard_idx, माप_प्रकार current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (पूर्णांक) loader->current_shard_idx);
}

व्योम dataloader_free(तथ्य_Loader *loader) {
    मुक्त(loader->बफर);
    मुक्त(loader->inputs);
    मुक्त(loader->targets);
    अगर (loader->should_shuffle) {
        मुक्त(loader->shard_indices);
        मुक्त(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_परिणाम);
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
#घोषणा ASSUMED_अंक_COMPLETIONS 4
// helper macro for ceildiv
#घोषणा CEIL_DIV(M, N) (((M) + (N)-1) / (N))

प्रकार काष्ठा {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    पूर्णांक process_rank;
    पूर्णांक अंक_processes;
    // hyperparameters. use size_t to prevent overflow
    माप_प्रकार B; // (micro) batch size dimension of the tensor that feeds into the model
    माप_प्रकार T; // maximum context length of the model
    // input handling and its state
    खाता* eval_file;
    uint16_t* बफर; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    पूर्णांक अंक_examples; // in total across all processes
    पूर्णांक अंक_batches; // to process the entire dataset across all processes
    पूर्णांक start_example_सूचकांक; // the assignment of work for this process, start
    पूर्णांक end_example_सूचकांक; // and end. start is inclusive, end is exclusive
    पूर्णांक current_example_सूचकांक; // the next example we would read
    पूर्णांक* inputs;  // input tokens into transformer
    पूर्णांक* targets; // target tokens for the transformer
    अक्षर* mask; // mask=1 at all completion token locations
    पूर्णांक* label; // the correct completion labels
    पूर्णांक अंक_completions; // number of completions for this example
} EvalLoader;

व्योम evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    पूर्णांक examples_per_process = CEIL_DIV(loader->अंक_examples, loader->अंक_processes);
    पूर्णांक can_fit_examples = (पूर्णांक) (loader->B / ASSUMED_अंक_COMPLETIONS);
    अगर (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        म_लिखो("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_अंक_COMPLETIONS);
        म_लिखो("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        निकास(निकास_त्रुटि);
    }
    loader->अंक_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_सूचकांक = examples_per_process * loader->process_rank;
    loader->end_example_सूचकांक = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    अगर (loader->end_example_सूचकांक > loader->अंक_examples) {
        loader->end_example_सूचकांक = loader->अंक_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_आकार * माप(पूर्णांक);
    fseekCheck(loader->eval_file, (पूर्णांक) header_bytes, शुरू_से);
    क्रम (पूर्णांक i = 0; i < loader->start_example_सूचकांक; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], माप(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        निश्चित(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        निश्चित(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        माप_प्रकार remaining_bytes = example_header[1] - माप(uint16_t) * 3;
        निश्चित(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (पूर्णांक) remaining_bytes, प्रस्तुत_से);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_सूचकांक = loader->start_example_सूचकांक;
}

व्योम evalloader_init(EvalLoader *loader,
                     स्थिर अक्षर* filename,
                     माप_प्रकार B,
                     माप_प्रकार T,
                     पूर्णांक process_rank,
                     पूर्णांक अंक_processes) {
    loader->process_rank = process_rank;
    loader->अंक_processes = अंक_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    पूर्णांक header[HEADER_आकार];
    freadCheck(header, माप(पूर्णांक), HEADER_आकार, loader->eval_file);
    अगर (header[0] != 20240522) { म_लिखो("Bad magic in eval file\n"); निकास(निकास_त्रुटि); }
    अगर (header[1] != 1) { म_लिखो("Bad version in data file\n"); निकास(निकास_त्रुटि); }
    loader->अंक_examples = header[2]; // number of examples in the file
    निश्चित(loader->अंक_examples >= अंक_processes); // avoid headaches for now
    माप_प्रकार longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    निश्चित(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_अंक_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    पूर्णांक can_fit_examples = (पूर्णांक) (B / ASSUMED_अंक_COMPLETIONS);
    loader->बफर = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (पूर्णांक*)सुस्मृति(B * T, माप(पूर्णांक));
    loader->targets = (पूर्णांक*)सुस्मृति(B * T, माप(पूर्णांक));
    loader->mask = (अक्षर*)mallocCheck(B * T * माप(अक्षर));
    loader->label = (पूर्णांक*)mallocCheck(can_fit_examples * माप(पूर्णांक));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

व्योम evalloader_अग्र_example(EvalLoader *loader, पूर्णांक example_batch_सूचकांक) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    पूर्णांक batch_dim_offset = example_batch_सूचकांक * ASSUMED_अंक_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], माप(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    निश्चित(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    निश्चित(example_header[2] == loader->current_example_सूचकांक); // <EXAMPLE_INDEX> should match the loop index
    निश्चित(example_header[2] >= loader->start_example_सूचकांक && example_header[2] < loader->end_example_सूचकांक);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    माप_प्रकार example_bytes = example_header[1] - माप(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->बफर, माप(अक्षर), example_bytes, loader->eval_file);
    // process the example label
    पूर्णांक label = (पूर्णांक)loader->बफर[0];
    पूर्णांक can_fit_examples = (पूर्णांक) (loader->B / ASSUMED_अंक_COMPLETIONS);
    निश्चित(label >= 0 && label < ASSUMED_अंक_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    निश्चित(example_batch_सूचकांक >= 0 && example_batch_सूचकांक < can_fit_examples);
    loader->label[example_batch_सूचकांक] = label; // store for output
    // process the number of completions
    पूर्णांक अंक_completions = (पूर्णांक)loader->बफर[1];
    निश्चित(अंक_completions == ASSUMED_अंक_COMPLETIONS); // we expect 4 completions for now
    निश्चित(batch_dim_offset + अंक_completions <= B); // we expect to fit in the batch
    loader->अंक_completions = अंक_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    पूर्णांक context_लंबाई = (पूर्णांक)loader->बफर[2];
    uint16_t *context_tokens_start = &loader->बफर[3]; // where the tokens start
    निश्चित(context_लंबाई > 0 && context_लंबाई < T); // context is non-empty and up to T
    क्रम (पूर्णांक b = 0; b < अंक_completions; b++) {
        क्रम (पूर्णांक i = 0; i < context_लंबाई; i++) {
            पूर्णांक boff = batch_dim_offset + b;
            पूर्णांक tok_cur = (पूर्णांक)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->बफर + 3 + context_लंबाई;
    क्रम (पूर्णांक c = 0; c < अंक_completions; c++) {
        पूर्णांक coff = batch_dim_offset + c;
        पूर्णांक completion_लंबाई = (पूर्णांक)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        निश्चित(completion_लंबाई > 0 && context_लंबाई + completion_लंबाई < T); // things fit?
        क्रम (पूर्णांक i = 0; i < completion_लंबाई; i++) {
            पूर्णांक tok_cur = (पूर्णांक)completion_tokens_start[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_लंबाई + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_लंबाई + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_लंबाई + i - 1] = 1;
        }
        completions_iter += 1 + completion_लंबाई; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_सूचकांक += 1;
}

व्योम evalloader_अग्र_batch(EvalLoader *loader) {
    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    स_रखो(loader->mask, 0, B * T * माप(अक्षर));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    पूर्णांक can_fit_examples = (पूर्णांक) (B / ASSUMED_अंक_COMPLETIONS); // how many examples can we fit in the batch?
    क्रम (पूर्णांक i = 0; i < can_fit_examples; i++) {
        अगर (loader->current_example_सूचकांक >= loader->end_example_सूचकांक) {
            अवरोध; // this process has exhausted its work, noop from here on
        }
        evalloader_अग्र_example(loader, i);
    }
}

पूर्णांक evalloader_stat_losses(EvalLoader *loader, भग्न* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    पूर्णांक correct = 0;
    माप_प्रकार B = loader->B;
    माप_प्रकार T = loader->T;
    // iterate the examples in this batch
    पूर्णांक can_fit_examples = (पूर्णांक) (B / ASSUMED_अंक_COMPLETIONS);
    क्रम (पूर्णांक i = 0; i < can_fit_examples; i++) {
        भग्न min_loss = 0.0f;
        पूर्णांक min_loss_सूचकांक = -1;
        अक्षर active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        क्रम (पूर्णांक b = 0; b < ASSUMED_अंक_COMPLETIONS; b++) {
            पूर्णांक boff = i * ASSUMED_अंक_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            भग्न average_loss = 0.0f;
            पूर्णांक गिनती = 0;
            क्रम (पूर्णांक t = 0; t < T; t++) {
                अक्षर mask = loader->mask[boff * T + t];
                अगर (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    गिनती++;
                }
            }
            अगर (गिनती > 0) { average_loss /= गिनती; }
            अगर (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_सूचकांक = b;
            }
        }
        अगर (active && (min_loss_सूचकांक == loader->label[i])) {
            correct += 1;
        }
    }
    वापस correct;
}

व्योम evalloader_free(EvalLoader *loader) {
    मुक्त(loader->बफर);
    मुक्त(loader->inputs);
    मुक्त(loader->targets);
    मुक्त(loader->mask);
    मुक्त(loader->label);
    fcloseCheck(loader->eval_file);
}

#पूर्ण_अगर // DATALOADER_H