/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#ضمن <assert.h>
#ضمن <math.h>
#ضمن <string.h>

تعريفنوع بنية {
    ثابت حرف* type;
    عائم learning_rate;
    صحيح warmup_iterations;
    صحيح train_num_batches;
    عائم final_learning_rate_frac;
} LearningRateScheduler;

فراغ lr_scheduler_init(LearningRateScheduler *scheduler, ثابت حرف* scheduler_type, عائم learning_rate, صحيح warmup_iterations, صحيح train_num_batches, عائم final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
عائم get_learning_rate_cosine(LearningRateScheduler *scheduler, صحيح step) {
    عائم lr = scheduler->learning_rate;
    𐤀𐤌 (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((عائم)(step + 1)) / scheduler->warmup_iterations;
    } وإلا {
        عائم decay_ratio = ((عائم)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        أكد(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        عائم coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        أكد(0.0f <= coeff && coeff <= 1.0f);
        عائم min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    أرجع lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
عائم get_learning_rate_linear(LearningRateScheduler *scheduler, صحيح step) {
    عائم lr = scheduler->learning_rate;
    𐤀𐤌 (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((عائم)(step + 1)) / scheduler->warmup_iterations;
    } وإلا {
        عائم decay_ratio = ((عائم)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        أكد(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        عائم min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    أرجع lr;
}

// constant
عائم get_learning_rate_constant(LearningRateScheduler *scheduler, صحيح step) {
    أرجع scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
عائم get_learning_rate_wsd(LearningRateScheduler *scheduler, صحيح step) {
    صحيح decay_point = (صحيح)(0.8f * scheduler->train_num_batches);
    عائم max_lr = scheduler->learning_rate;
    عائم lr = max_lr;
    𐤀𐤌 (step < scheduler->warmup_iterations) {
        عائم decay_ratio = ((عائم)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } وإلا 𐤀𐤌 (step < decay_point) {
        // noop, keep lr constant
    } وإلا {
        عائم decay_ratio = ((عائم)(step - decay_point)) / (scheduler->train_num_batches - decay_point);
        أكد(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        عائم min_lr = max_lr * scheduler->final_learning_rate_frac;
        أرجع min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    أرجع lr;
}

// return the learning rate at a given step
عائم get_learning_rate(LearningRateScheduler *scheduler, صحيح step) {
    عائم step_learning_rate;
    𐤀𐤌 (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } وإلا 𐤀𐤌 (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } وإلا 𐤀𐤌 (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } وإلا 𐤀𐤌 (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } وإلا {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    أرجع step_learning_rate;
}

#endif // SCHEDULERS_H
