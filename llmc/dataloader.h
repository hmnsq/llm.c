/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#অন্তর্ভাব "utils.h"
#অন্তর্ভাব "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#অন্তর্ভাব <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_আকার 256

প্রকারলেখ কাঠামো {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    পূর্ণসংখ্যা process_rank;
    পূর্ণসংখ্যা অঙ্ক_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t অঙ্ক_tokens; // total number of tokens
    size_t shard_অঙ্ক_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_ফলাফল; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    uint16_t* বাফার; // we fread data from file into this buffer
    পূর্ণসংখ্যা* inputs;  // input tokens into transformer
    পূর্ণসংখ্যা* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    পূর্ণসংখ্যা should_shuffle;
    পূর্ণসংখ্যা* shard_indices;
    পূর্ণসংখ্যা* intra_shard_indices;
    // sizes in bytes
    size_t যোগ_batch_আকার_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_আকার_bytes;
} তথ্য_Loader;

int64_t dataloader_load_shard_(তথ্য_Loader *loader, পূর্ণসংখ্যা shard_সূচক) {
    যদি (loader->should_shuffle) {
        shard_সূচক = loader->shard_indices[shard_সূচক];
    }
    // use the first glob match as the filename for now
    ধ্রুবক অক্ষর* filename = loader->glob_ফলাফল.gl_pathv[shard_সূচক];
    // open the input file for reading. also only a single file can be opened at a time
    যদি (loader->tokens_file != ফাঁকা) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    // validate the header
    পূর্ণসংখ্যা header[HEADER_আকার];
    freadCheck(header, আকার(পূর্ণসংখ্যা), HEADER_আকার, loader->tokens_file);
    যদি (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    যদি (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    int64_t ntok = header[2]; // number of tokens in the file
    প্রতিজ্ঞা(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->tokens_file, 0, SEEK_END); // seek to end of file
    loader->file_আকার_bytes = ftell(loader->tokens_file); // read the offset, i.e. file size
    fseekCheck(loader->tokens_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_আকার = HEADER_আকার * আকার(পূর্ণসংখ্যা) + ntok * আকার(uint16_t);
    যদি (loader->file_আকার_bytes != expected_file_আকার) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_অঙ্ক_samples = (ntok * আকার(uint16_t) - আকার(uint16_t)) / loader->যোগ_batch_আকার_bytes;
    ফেরত ntok;
}

শূন্য prepare_intra_shard_indices_(তথ্য_Loader *loader) {
    // shuffle the examples inside the shards
    যদি (loader->intra_shard_indices != ফাঁকা) {
        // in case shards have different number of samples / sizes
        মুক্ত(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (পূর্ণসংখ্যা*)mallocCheck(loader->shard_অঙ্ক_samples * আকার(পূর্ণসংখ্যা));
    init_identity_permutation(loader->intra_shard_indices, (পূর্ণসংখ্যা) loader->shard_অঙ্ক_samples);
    random_permutation(loader->intra_shard_indices, (পূর্ণসংখ্যা) loader->shard_অঙ্ক_samples, &loader->shuffle_rng);
}

শূন্য dataloader_reset(তথ্য_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    যদি (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (পূর্ণসংখ্যা) loader->glob_ফলাফল.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (পূর্ণসংখ্যা) loader->current_shard_idx);

    যদি (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

শূন্য dataloader_advance_(তথ্য_Loader *loader) {
    যদি (loader->current_shard_idx == loader->glob_ফলাফল.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        ফেরত;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_ফলাফল.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (পূর্ণসংখ্যা) loader->current_shard_idx);

    যদি (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

শূন্য dataloader_init(তথ্য_Loader *loader,
                     ধ্রুবক অক্ষর* filename_pattern,
                     size_t B,
                     size_t T,
                     পূর্ণসংখ্যা process_rank,
                     পূর্ণসংখ্যা অঙ্ক_processes,
                     পূর্ণসংখ্যা should_shuffle) {
    loader->process_rank = process_rank;
    loader->অঙ্ক_processes = অঙ্ক_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = ফাঁকা;
    loader->should_shuffle = should_shuffle;
    loader->header_bytes = HEADER_আকার * আকার(পূর্ণসংখ্যা);
    loader->যোগ_batch_আকার_bytes = ((loader->অঙ্ক_processes * (loader->B * loader->T)) * আকার(uint16_t));
    loader->local_batch_offset_bytes = loader->process_rank * loader->B * loader->T * আকার(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    পূর্ণসংখ্যা glob_status = glob(filename_pattern, 0, ফাঁকা, &loader->glob_ফলাফল);
    যদি (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    যদি (loader->glob_ফলাফল.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }

    যদি (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (পূর্ণসংখ্যা*)mallocCheck(loader->glob_ফলাফল.gl_pathc * আকার(পূর্ণসংখ্যা));
        init_identity_permutation(loader->shard_indices, (পূর্ণসংখ্যা) loader->glob_ফলাফল.gl_pathc);
        loader->intra_shard_indices = ফাঁকা;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_যোগ = 0;
    জন্য (পূর্ণসংখ্যা shard_সূচক = 0; shard_সূচক < loader->glob_ফলাফল.gl_pathc; shard_সূচক++) {
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_সূচক);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        প্রতিজ্ঞা(shard_ntok >= (int64_t) (অঙ্ক_processes * B * T + 1));
        ntok_যোগ += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->বাফার = (uint16_t*)mallocCheck((B * T + 1) * আকার(uint16_t));
    loader->inputs = (পূর্ণসংখ্যা*)mallocCheck(B * T * আকার(পূর্ণসংখ্যা));
    loader->targets = (পূর্ণসংখ্যা*)mallocCheck(B * T * আকার(পূর্ণসংখ্যা));
    loader->অঙ্ক_tokens = ntok_যোগ;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

শূন্য dataloader_load_batch(তথ্য_Loader* loader) {
    প্রতিজ্ঞা(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != ফাঁকা));
    প্রতিজ্ঞা(loader->current_sample_idx < loader->shard_অঙ্ক_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->যোগ_batch_আকার_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->tokens_file, (পূর্ণসংখ্যা) current_offset, SEEK_SET);
    freadCheck(loader->বাফার, আকার(uint16_t), B*T+1, loader->tokens_file);
    // decode the buffer into inputs and targets (cast to int)
    জন্য (পূর্ণসংখ্যা i = 0; i < B*T; i++) {
        loader->inputs[i] = (পূর্ণসংখ্যা)loader->বাফার[i];
        loader->targets[i] = (পূর্ণসংখ্যা)loader->বাফার[i+1];
    }
}

শূন্য dataloader_পরবর্তী_batch(তথ্য_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    যদি (loader->current_sample_idx >= loader->shard_অঙ্ক_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


শূন্য dataloader_resume(তথ্য_Loader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (পূর্ণসংখ্যা) loader->current_shard_idx);
}

শূন্য dataloader_free(তথ্য_Loader *loader) {
    মুক্ত(loader->বাফার);
    মুক্ত(loader->inputs);
    মুক্ত(loader->targets);
    যদি (loader->should_shuffle) {
        মুক্ত(loader->shard_indices);
        মুক্ত(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_ফলাফল);
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
#define ASSUMED_অঙ্ক_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

প্রকারলেখ কাঠামো {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    পূর্ণসংখ্যা process_rank;
    পূর্ণসংখ্যা অঙ্ক_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* বাফার; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    পূর্ণসংখ্যা অঙ্ক_examples; // in total across all processes
    পূর্ণসংখ্যা অঙ্ক_batches; // to process the entire dataset across all processes
    পূর্ণসংখ্যা start_example_সূচক; // the assignment of work for this process, start
    পূর্ণসংখ্যা end_example_সূচক; // and end. start is inclusive, end is exclusive
    পূর্ণসংখ্যা current_example_সূচক; // the next example we would read
    পূর্ণসংখ্যা* inputs;  // input tokens into transformer
    পূর্ণসংখ্যা* targets; // target tokens for the transformer
    অক্ষর* mask; // mask=1 at all completion token locations
    পূর্ণসংখ্যা* label; // the correct completion labels
    পূর্ণসংখ্যা অঙ্ক_completions; // number of completions for this example
} EvalLoader;

শূন্য evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    পূর্ণসংখ্যা examples_per_process = CEIL_DIV(loader->অঙ্ক_examples, loader->অঙ্ক_processes);
    পূর্ণসংখ্যা can_fit_examples = (পূর্ণসংখ্যা) (loader->B / ASSUMED_অঙ্ক_COMPLETIONS);
    যদি (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_অঙ্ক_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->অঙ্ক_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_সূচক = examples_per_process * loader->process_rank;
    loader->end_example_সূচক = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    যদি (loader->end_example_সূচক > loader->অঙ্ক_examples) {
        loader->end_example_সূচক = loader->অঙ্ক_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_আকার * আকার(পূর্ণসংখ্যা);
    fseekCheck(loader->eval_file, (পূর্ণসংখ্যা) header_bytes, SEEK_SET);
    জন্য (পূর্ণসংখ্যা i = 0; i < loader->start_example_সূচক; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], আকার(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        প্রতিজ্ঞা(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        প্রতিজ্ঞা(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - আকার(uint16_t) * 3;
        প্রতিজ্ঞা(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (পূর্ণসংখ্যা) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_সূচক = loader->start_example_সূচক;
}

শূন্য evalloader_init(EvalLoader *loader,
                     ধ্রুবক অক্ষর* filename,
                     size_t B,
                     size_t T,
                     পূর্ণসংখ্যা process_rank,
                     পূর্ণসংখ্যা অঙ্ক_processes) {
    loader->process_rank = process_rank;
    loader->অঙ্ক_processes = অঙ্ক_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    পূর্ণসংখ্যা header[HEADER_আকার];
    freadCheck(header, আকার(পূর্ণসংখ্যা), HEADER_আকার, loader->eval_file);
    যদি (header[0] != 20240522) { printf("Bad magic in eval file\n"); exit(EXIT_FAILURE); }
    যদি (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    loader->অঙ্ক_examples = header[2]; // number of examples in the file
    প্রতিজ্ঞা(loader->অঙ্ক_examples >= অঙ্ক_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    প্রতিজ্ঞা(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_অঙ্ক_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    পূর্ণসংখ্যা can_fit_examples = (পূর্ণসংখ্যা) (B / ASSUMED_অঙ্ক_COMPLETIONS);
    loader->বাফার = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (পূর্ণসংখ্যা*)calloc(B * T, আকার(পূর্ণসংখ্যা));
    loader->targets = (পূর্ণসংখ্যা*)calloc(B * T, আকার(পূর্ণসংখ্যা));
    loader->mask = (অক্ষর*)mallocCheck(B * T * আকার(অক্ষর));
    loader->label = (পূর্ণসংখ্যা*)mallocCheck(can_fit_examples * আকার(পূর্ণসংখ্যা));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

শূন্য evalloader_পরবর্তী_example(EvalLoader *loader, পূর্ণসংখ্যা example_batch_সূচক) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    পূর্ণসংখ্যা batch_dim_offset = example_batch_সূচক * ASSUMED_অঙ্ক_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], আকার(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    প্রতিজ্ঞা(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    প্রতিজ্ঞা(example_header[2] == loader->current_example_সূচক); // <EXAMPLE_INDEX> should match the loop index
    প্রতিজ্ঞা(example_header[2] >= loader->start_example_সূচক && example_header[2] < loader->end_example_সূচক);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_bytes = example_header[1] - আকার(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->বাফার, আকার(অক্ষর), example_bytes, loader->eval_file);
    // process the example label
    পূর্ণসংখ্যা label = (পূর্ণসংখ্যা)loader->বাফার[0];
    পূর্ণসংখ্যা can_fit_examples = (পূর্ণসংখ্যা) (loader->B / ASSUMED_অঙ্ক_COMPLETIONS);
    প্রতিজ্ঞা(label >= 0 && label < ASSUMED_অঙ্ক_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    প্রতিজ্ঞা(example_batch_সূচক >= 0 && example_batch_সূচক < can_fit_examples);
    loader->label[example_batch_সূচক] = label; // store for output
    // process the number of completions
    পূর্ণসংখ্যা অঙ্ক_completions = (পূর্ণসংখ্যা)loader->বাফার[1];
    প্রতিজ্ঞা(অঙ্ক_completions == ASSUMED_অঙ্ক_COMPLETIONS); // we expect 4 completions for now
    প্রতিজ্ঞা(batch_dim_offset + অঙ্ক_completions <= B); // we expect to fit in the batch
    loader->অঙ্ক_completions = অঙ্ক_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    পূর্ণসংখ্যা context_দৈর্ঘ্য = (পূর্ণসংখ্যা)loader->বাফার[2];
    uint16_t *context_tokens_start = &loader->বাফার[3]; // where the tokens start
    প্রতিজ্ঞা(context_দৈর্ঘ্য > 0 && context_দৈর্ঘ্য < T); // context is non-empty and up to T
    জন্য (পূর্ণসংখ্যা b = 0; b < অঙ্ক_completions; b++) {
        জন্য (পূর্ণসংখ্যা i = 0; i < context_দৈর্ঘ্য; i++) {
            পূর্ণসংখ্যা boff = batch_dim_offset + b;
            পূর্ণসংখ্যা tok_cur = (পূর্ণসংখ্যা)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->বাফার + 3 + context_দৈর্ঘ্য;
    জন্য (পূর্ণসংখ্যা c = 0; c < অঙ্ক_completions; c++) {
        পূর্ণসংখ্যা coff = batch_dim_offset + c;
        পূর্ণসংখ্যা completion_দৈর্ঘ্য = (পূর্ণসংখ্যা)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        প্রতিজ্ঞা(completion_দৈর্ঘ্য > 0 && context_দৈর্ঘ্য + completion_দৈর্ঘ্য < T); // things fit?
        জন্য (পূর্ণসংখ্যা i = 0; i < completion_দৈর্ঘ্য; i++) {
            পূর্ণসংখ্যা tok_cur = (পূর্ণসংখ্যা)completion_tokens_start[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_দৈর্ঘ্য + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_দৈর্ঘ্য + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_দৈর্ঘ্য + i - 1] = 1;
        }
        completions_iter += 1 + completion_দৈর্ঘ্য; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_সূচক += 1;
}

শূন্য evalloader_পরবর্তী_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * আকার(অক্ষর));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    পূর্ণসংখ্যা can_fit_examples = (পূর্ণসংখ্যা) (B / ASSUMED_অঙ্ক_COMPLETIONS); // how many examples can we fit in the batch?
    জন্য (পূর্ণসংখ্যা i = 0; i < can_fit_examples; i++) {
        যদি (loader->current_example_সূচক >= loader->end_example_সূচক) {
            বিরতি; // this process has exhausted its work, noop from here on
        }
        evalloader_পরবর্তী_example(loader, i);
    }
}

পূর্ণসংখ্যা evalloader_stat_losses(EvalLoader *loader, দশমিক* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    পূর্ণসংখ্যা correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    পূর্ণসংখ্যা can_fit_examples = (পূর্ণসংখ্যা) (B / ASSUMED_অঙ্ক_COMPLETIONS);
    জন্য (পূর্ণসংখ্যা i = 0; i < can_fit_examples; i++) {
        দশমিক min_loss = 0.0f;
        পূর্ণসংখ্যা min_loss_সূচক = -1;
        অক্ষর active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        জন্য (পূর্ণসংখ্যা b = 0; b < ASSUMED_অঙ্ক_COMPLETIONS; b++) {
            পূর্ণসংখ্যা boff = i * ASSUMED_অঙ্ক_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            দশমিক average_loss = 0.0f;
            পূর্ণসংখ্যা গুনতি = 0;
            জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
                অক্ষর mask = loader->mask[boff * T + t];
                যদি (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    গুনতি++;
                }
            }
            যদি (গুনতি > 0) { average_loss /= গুনতি; }
            যদি (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_সূচক = b;
            }
        }
        যদি (active && (min_loss_সূচক == loader->label[i])) {
            correct += 1;
        }
    }
    ফেরত correct;
}

শূন্য evalloader_free(EvalLoader *loader) {
    মুক্ত(loader->বাফার);
    মুক্ত(loader->inputs);
    মুক্ত(loader->targets);
    মুক্ত(loader->mask);
    মুক্ত(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H