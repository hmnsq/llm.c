/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#اگر_غیر_تعریف_شدہ DATALOADER_H
#تعریف_کرو DATALOADER_H

#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <stddef.h>
#شامل <stdint.h>
#شامل <assert.h>
#شامل <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#شامل "utils.h"
#شامل "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#اگر_غیر_تعریف_شدہ _WIN32
#شامل <glob.h>
#اگر_ختم
// ----------------------------------------------------------------------------
// Distributed Data Loader
#تعریف_کرو سرخی_سائز 256

قسمتعریف ساخت {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    صحیح process_rank;
    صحیح عدد_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t عدد_علامات; // total number of tokens
    size_t shard_عدد_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_نتیجہ; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* علامات_file;
    // data buffers
    uint16_t* بفر; // we fread data from file into this buffer
    صحیح* inputs;  // input tokens into transformer
    صحیح* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    صحیح should_shuffle;
    صحیح* shard_indices;
    صحیح* intra_shard_indices;
    // sizes in bytes
    size_t کل_batch_سائز_بائٹیں;  // total across all processes
    size_t local_batch_انحراف_بائٹیں;  // inner-sample offset for this process
    size_t سرخی_بائٹیں;  // header size in bytes
    int64_t file_سائز_بائٹیں;
} ڈیٹا_Loader;

int64_t dataloader_بھرو_shard(ڈیٹا_Loader *loader, صحیح shard_اشاریہ) {
    اگر (loader->should_shuffle) {
        shard_اشاریہ = loader->shard_indices[shard_اشاریہ];
    }
    // use the first glob match as the filename for now
    ثابت حرف* filename = loader->glob_نتیجہ.gl_pathv[shard_اشاریہ];
    // open the input file for reading. also only a single file can be opened at a time
    اگر (loader->علامات_file != تہی) {
        fcloseCheck(loader->علامات_file);
    }
    loader->علامات_file = fopenCheck(filename, "rb");
    // validate the header
    صحیح سرخی[سرخی_سائز];
    freadCheck(سرخی, حجم(صحیح), سرخی_سائز, loader->علامات_file);
    اگر (سرخی[0] != 20240520) {
        لکھو("Bad magic in the data file\n");
        لکھو("---> HINT: Are you passing in a correct file?\n");
        لکھو("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    اگر (سرخی[1] != 1) { لکھو("Bad version in data file\n"); باہر_نکلو(EXIT_FAILURE); }
    int64_t ntok = سرخی[2]; // number of tokens in the file
    أكد(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(loader->علامات_file, 0, SEEK_END); // seek to end of file
    loader->file_سائز_بائٹیں = ftell(loader->علامات_file); // read the offset, i.e. file size
    fseekCheck(loader->علامات_file, 0, SEEK_SET); // seek back to the beginning
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_سائز = سرخی_سائز * حجم(صحیح) + ntok * حجم(uint16_t);
    اگر (loader->file_سائز_بائٹیں != expected_file_سائز) {
        لکھو("Error: file size is not as expected\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader->shard_عدد_samples = (ntok * حجم(uint16_t) - حجم(uint16_t)) / loader->کل_batch_سائز_بائٹیں;
    واپس ntok;
}

خالی prepare_intra_shard_indices_(ڈیٹا_Loader *loader) {
    // shuffle the examples inside the shards
    اگر (loader->intra_shard_indices != تہی) {
        // in case shards have different number of samples / sizes
        آزاد(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (صحیح*)mallocCheck(loader->shard_عدد_samples * حجم(صحیح));
    init_شناختی_عنصر_permutation(loader->intra_shard_indices, (صحیح) loader->shard_عدد_samples);
    random_permutation(loader->intra_shard_indices, (صحیح) loader->shard_عدد_samples, &loader->shuffle_rng);
}

خالی dataloader_reset(ڈیٹا_Loader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    اگر (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (صحیح) loader->glob_نتیجہ.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_بھرو_shard(loader, (صحیح) loader->current_shard_idx);

    اگر (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

خالی dataloader_advance_(ڈیٹا_Loader *loader) {
    اگر (loader->current_shard_idx == loader->glob_نتیجہ.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        واپس;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_نتیجہ.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_بھرو_shard(loader, (صحیح) loader->current_shard_idx);

    اگر (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

خالی dataloader_init(ڈیٹا_Loader *loader,
                     ثابت حرف* filename_نمونہ,
                     size_t B,
                     size_t T,
                     صحیح process_rank,
                     صحیح عدد_processes,
                     صحیح should_shuffle) {
    loader->process_rank = process_rank;
    loader->عدد_processes = عدد_processes;
    loader->B = B;
    loader->T = T;
    loader->علامات_file = تہی;
    loader->should_shuffle = should_shuffle;
    loader->سرخی_بائٹیں = سرخی_سائز * حجم(صحیح);
    loader->کل_batch_سائز_بائٹیں = ((loader->عدد_processes * (loader->B * loader->T)) * حجم(uint16_t));
    loader->local_batch_انحراف_بائٹیں = loader->process_rank * loader->B * loader->T * حجم(uint16_t);

    // glob to get the list of files matching the pattern, these are our data shards
    صحیح glob_حالت = glob(filename_نمونہ, 0, تہی, &loader->glob_نتیجہ);
    اگر (glob_حالت != 0) {
        لکھو("Error: failed to glob pattern: %s\n", filename_نمونہ);
        باہر_نکلو(EXIT_FAILURE);
    }
    اگر (loader->glob_نتیجہ.gl_pathc == 0) {
        لکھو("Error: no files found matching the pattern: %s\n", filename_نمونہ);
        باہر_نکلو(EXIT_FAILURE);
    }

    اگر (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_بیج(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (صحیح*)mallocCheck(loader->glob_نتیجہ.gl_pathc * حجم(صحیح));
        init_شناختی_عنصر_permutation(loader->shard_indices, (صحیح) loader->glob_نتیجہ.gl_pathc);
        loader->intra_shard_indices = تہی;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_کل = 0;
    برائے (صحیح shard_اشاریہ = 0; shard_اشاریہ < loader->glob_نتیجہ.gl_pathc; shard_اشاریہ++) {
        int64_t shard_ntok = dataloader_بھرو_shard(loader, shard_اشاریہ);
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
    loader->inputs = (صحیح*)mallocCheck(B * T * حجم(صحیح));
    loader->targets = (صحیح*)mallocCheck(B * T * حجم(صحیح));
    loader->عدد_علامات = ntok_کل;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

خالی dataloader_بھرو_batch(ڈیٹا_Loader* loader) {
    أكد(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != تہی));
    أكد(loader->current_sample_idx < loader->shard_عدد_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_انحراف_بائٹیں = idx * loader->کل_batch_سائز_بائٹیں;
    int64_t current_انحراف = loader->سرخی_بائٹیں + global_batch_انحراف_بائٹیں + loader->local_batch_انحراف_بائٹیں;

    size_t B = loader->B;
    size_t T = loader->T;
    // read B*T+1 uint16_t tokens from the file into buffer
    fseekCheck(loader->علامات_file, (صحیح) current_انحراف, SEEK_SET);
    freadCheck(loader->بفر, حجم(uint16_t), B*T+1, loader->علامات_file);
    // decode the buffer into inputs and targets (cast to int)
    برائے (صحیح i = 0; i < B*T; i++) {
        loader->inputs[i] = (صحیح)loader->بفر[i];
        loader->targets[i] = (صحیح)loader->بفر[i+1];
    }
}

خالی dataloader_اگلا_batch(ڈیٹا_Loader *loader) {
    // if the next batch would go past the end of the file, advance the loader
    اگر (loader->current_sample_idx >= loader->shard_عدد_samples) {
        dataloader_advance_(loader);
    }
    dataloader_بھرو_batch(loader);
    loader->current_sample_idx += 1;
}


خالی dataloader_resume(ڈیٹا_Loader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_بھرو_shard(loader, (صحیح) loader->current_shard_idx);
}

خالی dataloader_free(ڈیٹا_Loader *loader) {
    آزاد(loader->بفر);
    آزاد(loader->inputs);
    آزاد(loader->targets);
    اگر (loader->should_shuffle) {
        آزاد(loader->shard_indices);
        آزاد(loader->intra_shard_indices);
    }
    fcloseCheck(loader->علامات_file);
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

قسمتعریف ساخت {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    صحیح process_rank;
    صحیح عدد_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* بفر; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    صحیح عدد_examples; // in total across all processes
    صحیح عدد_batches; // to process the entire dataset across all processes
    صحیح آغاز_example_اشاریہ; // the assignment of work for this process, start
    صحیح انتہا_example_اشاریہ; // and end. start is inclusive, end is exclusive
    صحیح current_example_اشاریہ; // the next example we would read
    صحیح* inputs;  // input tokens into transformer
    صحیح* targets; // target tokens for the transformer
    حرف* نقاب; // mask=1 at all completion token locations
    صحیح* label; // the correct completion labels
    صحیح عدد_completions; // number of completions for this example
} EvalLoader;

خالی evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    صحیح examples_فی_process = CEIL_DIV(loader->عدد_examples, loader->عدد_processes);
    صحیح can_fit_examples = (صحیح) (loader->B / ASSUMED_عدد_COMPLETIONS);
    اگر (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        لکھو("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_عدد_COMPLETIONS);
        لکھو("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        باہر_نکلو(EXIT_FAILURE);
    }
    loader->عدد_batches = CEIL_DIV(examples_فی_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->آغاز_example_اشاریہ = examples_فی_process * loader->process_rank;
    loader->انتہا_example_اشاریہ = examples_فی_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    اگر (loader->انتہا_example_اشاریہ > loader->عدد_examples) {
        loader->انتہا_example_اشاریہ = loader->عدد_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t سرخی_بائٹیں = سرخی_سائز * حجم(صحیح);
    fseekCheck(loader->eval_file, (صحیح) سرخی_بائٹیں, SEEK_SET);
    برائے (صحیح i = 0; i < loader->آغاز_example_اشاریہ; i++) {
        uint16_t example_سرخی[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_سرخی[0], حجم(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        أكد(example_سرخی[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        أكد(example_سرخی[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t بقایا_بائٹیں = example_سرخی[1] - حجم(uint16_t) * 3;
        أكد(بقایا_بائٹیں > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (صحیح) بقایا_بائٹیں, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_اشاریہ = loader->آغاز_example_اشاریہ;
}

خالی evalloader_init(EvalLoader *loader,
                     ثابت حرف* filename,
                     size_t B,
                     size_t T,
                     صحیح process_rank,
                     صحیح عدد_processes) {
    loader->process_rank = process_rank;
    loader->عدد_processes = عدد_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    صحیح سرخی[سرخی_سائز];
    freadCheck(سرخی, حجم(صحیح), سرخی_سائز, loader->eval_file);
    اگر (سرخی[0] != 20240522) { لکھو("Bad magic in eval file\n"); باہر_نکلو(EXIT_FAILURE); }
    اگر (سرخی[1] != 1) { لکھو("Bad version in data file\n"); باہر_نکلو(EXIT_FAILURE); }
    loader->عدد_examples = سرخی[2]; // number of examples in the file
    أكد(loader->عدد_examples >= عدد_processes); // avoid headaches for now
    size_t longest_example_بائٹیں = سرخی[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    أكد(longest_example_بائٹیں > 0 && longest_example_بائٹیں < (1+ASSUMED_عدد_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    صحیح can_fit_examples = (صحیح) (B / ASSUMED_عدد_COMPLETIONS);
    loader->بفر = (uint16_t*)mallocCheck(longest_example_بائٹیں);
    loader->inputs = (صحیح*)صفر_حافظہ_دو(B * T, حجم(صحیح));
    loader->targets = (صحیح*)صفر_حافظہ_دو(B * T, حجم(صحیح));
    loader->نقاب = (حرف*)mallocCheck(B * T * حجم(حرف));
    loader->label = (صحیح*)mallocCheck(can_fit_examples * حجم(صحیح));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

خالی evalloader_اگلا_example(EvalLoader *loader, صحیح example_batch_اشاریہ) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    صحیح batch_dim_انحراف = example_batch_اشاریہ * ASSUMED_عدد_COMPLETIONS;
    // read the current example header
    uint16_t example_سرخی[3];
    freadCheck(&example_سرخی[0], حجم(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    أكد(example_سرخی[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    أكد(example_سرخی[2] == loader->current_example_اشاریہ); // <EXAMPLE_INDEX> should match the loop index
    أكد(example_سرخی[2] >= loader->آغاز_example_اشاریہ && example_سرخی[2] < loader->انتہا_example_اشاریہ);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_بائٹیں = example_سرخی[1] - حجم(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->بفر, حجم(حرف), example_بائٹیں, loader->eval_file);
    // process the example label
    صحیح label = (صحیح)loader->بفر[0];
    صحیح can_fit_examples = (صحیح) (loader->B / ASSUMED_عدد_COMPLETIONS);
    أكد(label >= 0 && label < ASSUMED_عدد_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    أكد(example_batch_اشاریہ >= 0 && example_batch_اشاریہ < can_fit_examples);
    loader->label[example_batch_اشاریہ] = label; // store for output
    // process the number of completions
    صحیح عدد_completions = (صحیح)loader->بفر[1];
    أكد(عدد_completions == ASSUMED_عدد_COMPLETIONS); // we expect 4 completions for now
    أكد(batch_dim_انحراف + عدد_completions <= B); // we expect to fit in the batch
    loader->عدد_completions = عدد_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    صحیح context_لمبائی = (صحیح)loader->بفر[2];
    uint16_t *context_علامات_آغاز = &loader->بفر[3]; // where the tokens start
    أكد(context_لمبائی > 0 && context_لمبائی < T); // context is non-empty and up to T
    برائے (صحیح b = 0; b < عدد_completions; b++) {
        برائے (صحیح i = 0; i < context_لمبائی; i++) {
            صحیح boff = batch_dim_انحراف + b;
            صحیح tok_cur = (صحیح)context_علامات_آغاز[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->بفر + 3 + context_لمبائی;
    برائے (صحیح c = 0; c < عدد_completions; c++) {
        صحیح coff = batch_dim_انحراف + c;
        صحیح completion_لمبائی = (صحیح)completions_iter[0];
        uint16_t *completion_علامات_آغاز = completions_iter + 1;
        أكد(completion_لمبائی > 0 && context_لمبائی + completion_لمبائی < T); // things fit?
        برائے (صحیح i = 0; i < completion_لمبائی; i++) {
            صحیح tok_cur = (صحیح)completion_علامات_آغاز[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_لمبائی + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_لمبائی + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->نقاب[coff * T + context_لمبائی + i - 1] = 1;
        }
        completions_iter += 1 + completion_لمبائی; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_اشاریہ += 1;
}

خالی evalloader_اگلا_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->نقاب, 0, B * T * حجم(حرف));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    صحیح can_fit_examples = (صحیح) (B / ASSUMED_عدد_COMPLETIONS); // how many examples can we fit in the batch?
    برائے (صحیح i = 0; i < can_fit_examples; i++) {
        اگر (loader->current_example_اشاریہ >= loader->انتہا_example_اشاریہ) {
            توڑو; // this process has exhausted its work, noop from here on
        }
        evalloader_اگلا_example(loader, i);
    }
}

صحیح evalloader_stat_losses(EvalLoader *loader, اعشاری* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    صحیح correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    صحیح can_fit_examples = (صحیح) (B / ASSUMED_عدد_COMPLETIONS);
    برائے (صحیح i = 0; i < can_fit_examples; i++) {
        اعشاری min_loss = 0.0f;
        صحیح min_loss_اشاریہ = -1;
        حرف فعال = 0; // is this example active or fully empty?
        // iterate the completions in this example
        برائے (صحیح b = 0; b < ASSUMED_عدد_COMPLETIONS; b++) {
            صحیح boff = i * ASSUMED_عدد_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            اعشاری اوسط_قدر_loss = 0.0f;
            صحیح گنتی = 0;
            برائے (صحیح t = 0; t < T; t++) {
                حرف نقاب = loader->نقاب[boff * T + t];
                اگر (نقاب == 1) {
                    فعال = 1;
                    اوسط_قدر_loss += losses[boff * T + t];
                    گنتی++;
                }
            }
            اگر (گنتی > 0) { اوسط_قدر_loss /= گنتی; }
            اگر (b == 0 || اوسط_قدر_loss < min_loss) {
                min_loss = اوسط_قدر_loss;
                min_loss_اشاریہ = b;
            }
        }
        اگر (فعال && (min_loss_اشاریہ == loader->label[i])) {
            correct += 1;
        }
    }
    واپس correct;
}

خالی evalloader_free(EvalLoader *loader) {
    آزاد(loader->بفر);
    آزاد(loader->inputs);
    آزاد(loader->targets);
    آزاد(loader->نقاب);
    آزاد(loader->label);
    fcloseCheck(loader->eval_file);
}

#اگر_ختم // DATALOADER_H