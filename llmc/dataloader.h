/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#جے_بنا_تعریف DATALOADER_H
#تعریف_کرو DATALOADER_H

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <stddef.h>
#ضمن <stdint.h>
#ضمن <assert.h>
#ضمن <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#ضمن "utils.h"
#ضمن "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#جے_بنا_تعریف _WIN32
#ضمن <glob.h>
#جے_انت
// ----------------------------------------------------------------------------
// Distributed Data Loader
#تعریف_کرو سرخی_ناپ 256

تعريفنوع بنية {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    صحيح process_rank;
    صحيح عدد_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t عدد_ٹوکن; // total number of tokens
    size_t shard_عدد_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_نتیجہ; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* ٹوکن_file;
    // data buffers
    uint16_t* بفر; // we fread data from file into this buffer
    صحيح* inputs;  // input tokens into transformer
    صحيح* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    صحيح should_shuffle;
    صحيح* shard_indices;
    صحيح* intra_shard_indices;
    // sizes in bytes
    size_t کل_batch_ناپ_بائٹاں;  // total across all processes
    size_t local_batch_وتھ_بائٹاں;  // inner-sample offset for this process
    size_t سرخی_بائٹاں;  // header size in bytes
    int64_t file_ناپ_بائٹاں;
} انکڑے_Loader;

int64_t dataloader_لدو_shard(انکڑے_Loader *loader, صحيح shard_اشاریہ) {
    إذا (loader->should_shuffle) {
        shard_اشاریہ = loader->shard_indices[shard_اشاریہ];
    }
    // use the first glob match as the filename for now
    ثابت حرف* filename = loader->glob_نتیجہ.gl_pathv[shard_اشاریہ];
    // open the input file for reading. also only a single file can be opened at a time
    إذا (loader->ٹوکن_file != عدم) {
        fcloseCheck(loader->ٹوکن_file);
    }
    loader->ٹوکن_file = fopenCheck(filename, "rb");
    // validate the header
    صحيح سرخی[سرخی_ناپ];
    freadCheck(سرخی, حجم(صحيح), سرخی_ناپ, loader->ٹوکن_file);
    إذا (سرخی[0] != 20240520) {
        لکھو("Bad magic in the data file\n");
        لکھو("---> HINT: Are you passing in a correct file?\n");
        لکھو("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    إذا (سرخی[1] != 1) { لکھو("Bad version in data file\n"); باہر_نکلو(EXIT_FAILURE); }
    int64_t ntok = سرخی[2]; // number of tokens in the file
    أكد(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->ٹوکن_file, 0, SEEK_END); // seek to end of file
    loader->file_ناپ_بائٹاں = ftell(loader->ٹوکن_file); // read the offset, i.e. file size
    fseekCheck(loader->ٹوکن_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_ناپ = سرخی_ناپ * حجم(صحيح) + ntok * حجم(uint16_t);
    إذا (loader->file_ناپ_بائٹاں != expected_file_ناپ) {
        لکھو("Error: file size is not as expected\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_عدد_samples = (ntok * حجم(uint16_t) - حجم(uint16_t)) / loader->کل_batch_ناپ_بائٹاں;
    أرجع ntok;
}

فراغ prepare_intra_shard_indices_(انکڑے_Loader *loader) {
    // shuffle the examples inside the shards
    إذا (loader->intra_shard_indices != عدم) {
        // in case shards have different number of samples / sizes
        حرر(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (صحيح*)mallocCheck(loader->shard_عدد_samples * حجم(صحيح));
    init_پچھان_تت_permutation(loader->intra_shard_indices, (صحيح) loader->shard_عدد_samples);
    random_permutation(loader->intra_shard_indices, (صحيح) loader->shard_عدد_samples, &loader->shuffle_rng);
}

فراغ dataloader_reset(انکڑے_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    إذا (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (صحيح) loader->glob_نتیجہ.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_لدو_shard(loader, (صحيح) loader->current_shard_idx);

    إذا (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

فراغ dataloader_advance_(انکڑے_Loader *loader) {
    إذا (loader->current_shard_idx == loader->glob_نتیجہ.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        أرجع;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_نتیجہ.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_لدو_shard(loader, (صحيح) loader->current_shard_idx);

    إذا (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

فراغ dataloader_init(انکڑے_Loader *loader,
                     ثابت حرف* filename_نمونہ,
                     size_t B,
                     size_t T,
                     صحيح process_rank,
                     صحيح عدد_processes,
                     صحيح should_shuffle) {
    loader->process_rank = process_rank;
    loader->عدد_processes = عدد_processes;
    loader->B = B;
    loader->T = T;
    loader->ٹوکن_file = عدم;
    loader->should_shuffle = should_shuffle;
    loader->سرخی_بائٹاں = سرخی_ناپ * حجم(صحيح);
    loader->کل_batch_ناپ_بائٹاں = ((loader->عدد_processes * (loader->B * loader->T)) * حجم(uint16_t));
    loader->local_batch_وتھ_بائٹاں = loader->process_rank * loader->B * loader->T * حجم(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    صحيح glob_حالت = glob(filename_نمونہ, 0, عدم, &loader->glob_نتیجہ);
    إذا (glob_حالت != 0) {
        لکھو("Error: failed to glob pattern: %s\n", filename_نمونہ);
        باہر_نکلو(EXIT_FAILURE);
    }
    إذا (loader->glob_نتیجہ.gl_pathc == 0) {
        لکھو("Error: no files found matching the pattern: %s\n", filename_نمونہ);
        باہر_نکلو(EXIT_FAILURE);
    }

    إذا (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_بیج(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (صحيح*)mallocCheck(loader->glob_نتیجہ.gl_pathc * حجم(صحيح));
        init_پچھان_تت_permutation(loader->shard_indices, (صحيح) loader->glob_نتیجہ.gl_pathc);
        loader->intra_shard_indices = عدم;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_کل = 0;
    لكل (صحيح shard_اشاریہ = 0; shard_اشاریہ < loader->glob_نتیجہ.gl_pathc; shard_اشاریہ++) {
        int64_t shard_ntok = dataloader_لدو_shard(loader, shard_اشاریہ);
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        أكد(shard_ntok >= (int64_t) (عدد_processes * B * T + 1));
        ntok_کل += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->بفر = (uint16_t*)mallocCheck((B * T + 1) * حجم(uint16_t));
    loader->inputs = (صحيح*)mallocCheck(B * T * حجم(صحيح));
    loader->targets = (صحيح*)mallocCheck(B * T * حجم(صحيح));
    loader->عدد_ٹوکن = ntok_کل;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

فراغ dataloader_لدو_batch(انکڑے_Loader* loader) {
    أكد(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != عدم));
    أكد(loader->current_sample_idx < loader->shard_عدد_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_وتھ_بائٹاں = idx * loader->کل_batch_ناپ_بائٹاں;
    int64_t current_وتھ = loader->سرخی_بائٹاں + global_batch_وتھ_بائٹاں + loader->local_batch_وتھ_بائٹاں;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->ٹوکن_file, (صحيح) current_وتھ, SEEK_SET);
    freadCheck(loader->بفر, حجم(uint16_t), B*T+1, loader->ٹوکن_file);
    // decode the buffer into inputs and targets (cast to int)
    لكل (صحيح i = 0; i < B*T; i++) {
        loader->inputs[i] = (صحيح)loader->بفر[i];
        loader->targets[i] = (صحيح)loader->بفر[i+1];
    }
}

فراغ dataloader_اگلا_batch(انکڑے_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    إذا (loader->current_sample_idx >= loader->shard_عدد_samples) {
        dataloader_advance_(loader);
    }
    dataloader_لدو_batch(loader);
    loader->current_sample_idx += 1;
}


فراغ dataloader_resume(انکڑے_Loader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_لدو_shard(loader, (صحيح) loader->current_shard_idx);
}

فراغ dataloader_free(انکڑے_Loader *loader) {
    حرر(loader->بفر);
    حرر(loader->inputs);
    حرر(loader->targets);
    إذا (loader->should_shuffle) {
        حرر(loader->shard_indices);
        حرر(loader->intra_shard_indices);
    }
    fcloseCheck(loader->ٹوکن_file);
    globfree(&loader->glob_نتیجہ);
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
#تعریف_کرو ASSUMED_عدد_COMPLETIONS 4
// helper macro for ceildiv
#تعریف_کرو CEIL_DIV(M, N) (((M) + (N)-1) / (N))

تعريفنوع بنية {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    صحيح process_rank;
    صحيح عدد_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* بفر; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    صحيح عدد_examples; // in total across all processes
    صحيح عدد_batches; // to process the entire dataset across all processes
    صحيح شروع_example_اشاریہ; // the assignment of work for this process, start
    صحيح سرا_example_اشاریہ; // and end. start is inclusive, end is exclusive
    صحيح current_example_اشاریہ; // the next example we would read
    صحيح* inputs;  // input tokens into transformer
    صحيح* targets; // target tokens for the transformer
    حرف* نقاب; // mask=1 at all completion token locations
    صحيح* label; // the correct completion labels
    صحيح عدد_completions; // number of completions for this example
} EvalLoader;

فراغ evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    صحيح examples_فی_process = CEIL_DIV(loader->عدد_examples, loader->عدد_processes);
    صحيح can_fit_examples = (صحيح) (loader->B / ASSUMED_عدد_COMPLETIONS);
    إذا (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        لکھو("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_عدد_COMPLETIONS);
        لکھو("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    loader->عدد_batches = CEIL_DIV(examples_فی_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->شروع_example_اشاریہ = examples_فی_process * loader->process_rank;
    loader->سرا_example_اشاریہ = examples_فی_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    إذا (loader->سرا_example_اشاریہ > loader->عدد_examples) {
        loader->سرا_example_اشاریہ = loader->عدد_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t سرخی_بائٹاں = سرخی_ناپ * حجم(صحيح);
    fseekCheck(loader->eval_file, (صحيح) سرخی_بائٹاں, SEEK_SET);
    لكل (صحيح i = 0; i < loader->شروع_example_اشاریہ; i++) {
        uint16_t example_سرخی[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_سرخی[0], حجم(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        أكد(example_سرخی[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        أكد(example_سرخی[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t بچیا_بائٹاں = example_سرخی[1] - حجم(uint16_t) * 3;
        أكد(بچیا_بائٹاں > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (صحيح) بچیا_بائٹاں, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_اشاریہ = loader->شروع_example_اشاریہ;
}

فراغ evalloader_init(EvalLoader *loader,
                     ثابت حرف* filename,
                     size_t B,
                     size_t T,
                     صحيح process_rank,
                     صحيح عدد_processes) {
    loader->process_rank = process_rank;
    loader->عدد_processes = عدد_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    صحيح سرخی[سرخی_ناپ];
    freadCheck(سرخی, حجم(صحيح), سرخی_ناپ, loader->eval_file);
    إذا (سرخی[0] != 20240522) { لکھو("Bad magic in eval file\n"); باہر_نکلو(EXIT_FAILURE); }
    إذا (سرخی[1] != 1) { لکھو("Bad version in data file\n"); باہر_نکلو(EXIT_FAILURE); }
    loader->عدد_examples = سرخی[2]; // number of examples in the file
    أكد(loader->عدد_examples >= عدد_processes); // avoid headaches for now
    size_t longest_example_بائٹاں = سرخی[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    أكد(longest_example_بائٹاں > 0 && longest_example_بائٹاں < (1+ASSUMED_عدد_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    صحيح can_fit_examples = (صحيح) (B / ASSUMED_عدد_COMPLETIONS);
    loader->بفر = (uint16_t*)mallocCheck(longest_example_بائٹاں);
    loader->inputs = (صحيح*)صفر_حافظہ_دیو(B * T, حجم(صحيح));
    loader->targets = (صحيح*)صفر_حافظہ_دیو(B * T, حجم(صحيح));
    loader->نقاب = (حرف*)mallocCheck(B * T * حجم(حرف));
    loader->label = (صحيح*)mallocCheck(can_fit_examples * حجم(صحيح));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

فراغ evalloader_اگلا_example(EvalLoader *loader, صحيح example_batch_اشاریہ) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    صحيح batch_dim_وتھ = example_batch_اشاریہ * ASSUMED_عدد_COMPLETIONS;
    // read the current example header
    uint16_t example_سرخی[3];
    freadCheck(&example_سرخی[0], حجم(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    أكد(example_سرخی[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    أكد(example_سرخی[2] == loader->current_example_اشاریہ); // <EXAMPLE_INDEX> should match the loop index
    أكد(example_سرخی[2] >= loader->شروع_example_اشاریہ && example_سرخی[2] < loader->سرا_example_اشاریہ);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_بائٹاں = example_سرخی[1] - حجم(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->بفر, حجم(حرف), example_بائٹاں, loader->eval_file);
    // process the example label
    صحيح label = (صحيح)loader->بفر[0];
    صحيح can_fit_examples = (صحيح) (loader->B / ASSUMED_عدد_COMPLETIONS);
    أكد(label >= 0 && label < ASSUMED_عدد_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    أكد(example_batch_اشاریہ >= 0 && example_batch_اشاریہ < can_fit_examples);
    loader->label[example_batch_اشاریہ] = label; // store for output
    // process the number of completions
    صحيح عدد_completions = (صحيح)loader->بفر[1];
    أكد(عدد_completions == ASSUMED_عدد_COMPLETIONS); // we expect 4 completions for now
    أكد(batch_dim_وتھ + عدد_completions <= B); // we expect to fit in the batch
    loader->عدد_completions = عدد_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    صحيح context_لمائی = (صحيح)loader->بفر[2];
    uint16_t *context_ٹوکن_شروع = &loader->بفر[3]; // where the tokens start
    أكد(context_لمائی > 0 && context_لمائی < T); // context is non-empty and up to T
    لكل (صحيح b = 0; b < عدد_completions; b++) {
        لكل (صحيح i = 0; i < context_لمائی; i++) {
            صحيح boff = batch_dim_وتھ + b;
            صحيح tok_cur = (صحيح)context_ٹوکن_شروع[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->بفر + 3 + context_لمائی;
    لكل (صحيح c = 0; c < عدد_completions; c++) {
        صحيح coff = batch_dim_وتھ + c;
        صحيح completion_لمائی = (صحيح)completions_iter[0];
        uint16_t *completion_ٹوکن_شروع = completions_iter + 1;
        أكد(completion_لمائی > 0 && context_لمائی + completion_لمائی < T); // things fit?
        لكل (صحيح i = 0; i < completion_لمائی; i++) {
            صحيح tok_cur = (صحيح)completion_ٹوکن_شروع[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_لمائی + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_لمائی + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->نقاب[coff * T + context_لمائی + i - 1] = 1;
        }
        completions_iter += 1 + completion_لمائی; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_اشاریہ += 1;
}

فراغ evalloader_اگلا_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->نقاب, 0, B * T * حجم(حرف));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    صحيح can_fit_examples = (صحيح) (B / ASSUMED_عدد_COMPLETIONS); // how many examples can we fit in the batch?
    لكل (صحيح i = 0; i < can_fit_examples; i++) {
        إذا (loader->current_example_اشاریہ >= loader->سرا_example_اشاریہ) {
            اكسر; // this process has exhausted its work, noop from here on
        }
        evalloader_اگلا_example(loader, i);
    }
}

صحيح evalloader_stat_losses(EvalLoader *loader, عائم* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    صحيح correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    صحيح can_fit_examples = (صحيح) (B / ASSUMED_عدد_COMPLETIONS);
    لكل (صحيح i = 0; i < can_fit_examples; i++) {
        عائم min_loss = 0.0f;
        صحيح min_loss_اشاریہ = -1;
        حرف سرگرم = 0; // is this example active or fully empty?
        // iterate the completions in this example
        لكل (صحيح b = 0; b < ASSUMED_عدد_COMPLETIONS; b++) {
            صحيح boff = i * ASSUMED_عدد_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            عائم اوسطن_loss = 0.0f;
            صحيح گنتی = 0;
            لكل (صحيح t = 0; t < T; t++) {
                حرف نقاب = loader->نقاب[boff * T + t];
                إذا (نقاب == 1) {
                    سرگرم = 1;
                    اوسطن_loss += losses[boff * T + t];
                    گنتی++;
                }
            }
            إذا (گنتی > 0) { اوسطن_loss /= گنتی; }
            إذا (b == 0 || اوسطن_loss < min_loss) {
                min_loss = اوسطن_loss;
                min_loss_اشاریہ = b;
            }
        }
        إذا (سرگرم && (min_loss_اشاریہ == loader->label[i])) {
            correct += 1;
        }
    }
    أرجع correct;
}

فراغ evalloader_free(EvalLoader *loader) {
    حرر(loader->بفر);
    حرر(loader->inputs);
    حرر(loader->targets);
    حرر(loader->نقاب);
    حرر(loader->label);
    fcloseCheck(loader->eval_file);
}

#جے_انت // DATALOADER_H