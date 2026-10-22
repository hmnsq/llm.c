/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <string.h>

തരനിർവചനം ഘടന {
    സ്ഥിരം അക്ഷരം* type;
    ദശാംശം learning_rate;
    പൂർണ്ണസംഖ്യ warmup_iterations;
    പൂർണ്ണസംഖ്യ train_അക്കം_batches;
    ദശാംശം final_learning_rate_frac;
} LearningRateScheduler;

ശൂന്യം lr_scheduler_init(LearningRateScheduler *scheduler, സ്ഥിരം അക്ഷരം* scheduler_type, ദശാംശം learning_rate, പൂർണ്ണസംഖ്യ warmup_iterations, പൂർണ്ണസംഖ്യ train_അക്കം_batches, ദശാംശം final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_അക്കം_batches = train_അക്കം_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
ദശാംശം get_learning_rate_cosine(LearningRateScheduler *scheduler, പൂർണ്ണസംഖ്യ step) {
    ദശാംശം lr = scheduler->learning_rate;
    എങ്കിൽ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ദശാംശം)(step + 1)) / scheduler->warmup_iterations;
    } അല്ലെങ്കിൽ {
        ദശാംശം decay_ratio = ((ദശാംശം)(step - scheduler->warmup_iterations)) / (scheduler->train_അക്കം_batches - scheduler->warmup_iterations);
        പ്രതിജ്ഞാ(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ദശാംശം coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        പ്രതിജ്ഞാ(0.0f <= coeff && coeff <= 1.0f);
        ദശാംശം min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    തിരികെ lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
ദശാംശം get_learning_rate_linear(LearningRateScheduler *scheduler, പൂർണ്ണസംഖ്യ step) {
    ദശാംശം lr = scheduler->learning_rate;
    എങ്കിൽ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ദശാംശം)(step + 1)) / scheduler->warmup_iterations;
    } അല്ലെങ്കിൽ {
        ദശാംശം decay_ratio = ((ദശാംശം)(step - scheduler->warmup_iterations)) / (scheduler->train_അക്കം_batches - scheduler->warmup_iterations);
        പ്രതിജ്ഞാ(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ദശാംശം min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    തിരികെ lr;
}

// constant
ദശാംശം get_learning_rate_constant(LearningRateScheduler *scheduler, പൂർണ്ണസംഖ്യ step) {
    തിരികെ scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
ദശാംശം get_learning_rate_wsd(LearningRateScheduler *scheduler, പൂർണ്ണസംഖ്യ step) {
    പൂർണ്ണസംഖ്യ decay_point = (പൂർണ്ണസംഖ്യ)(0.8f * scheduler->train_അക്കം_batches);
    ദശാംശം max_lr = scheduler->learning_rate;
    ദശാംശം lr = max_lr;
    എങ്കിൽ (step < scheduler->warmup_iterations) {
        ദശാംശം decay_ratio = ((ദശാംശം)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } അല്ലെങ്കിൽ എങ്കിൽ (step < decay_point) {
        // noop, keep lr constant
    } അല്ലെങ്കിൽ {
        ദശാംശം decay_ratio = ((ദശാംശം)(step - decay_point)) / (scheduler->train_അക്കം_batches - decay_point);
        പ്രതിജ്ഞാ(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ദശാംശം min_lr = max_lr * scheduler->final_learning_rate_frac;
        തിരികെ min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    തിരികെ lr;
}

// return the learning rate at a given step
ദശാംശം get_learning_rate(LearningRateScheduler *scheduler, പൂർണ്ണസംഖ്യ step) {
    ദശാംശം step_learning_rate;
    എങ്കിൽ (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } അല്ലെങ്കിൽ എങ്കിൽ (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } അല്ലെങ്കിൽ എങ്കിൽ (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } അല്ലെങ്കിൽ എങ്കിൽ (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } അല്ലെങ്കിൽ {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    തിരികെ step_learning_rate;
}

#endif // SCHEDULERS_H
