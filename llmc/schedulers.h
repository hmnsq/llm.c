/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#include <assert.h>
#include <math.h>
#include <string.h>

typedef sa_nracanaa {
    const var_na* type;
    float learning_rate;
    ຈຳນວນເຕັມ warmup_iterations;
    ຈຳນວນເຕັມ train_num_batches;
    float final_learning_rate_frac;
} LearningRateScheduler;

void lr_scheduler_init(LearningRateScheduler *scheduler, const var_na* scheduler_type, float learning_rate, ຈຳນວນເຕັມ warmup_iterations, ຈຳນວນເຕັມ train_num_batches, float final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
float get_learning_rate_cosine(LearningRateScheduler *scheduler, ຈຳນວນເຕັມ step) {
    float lr = scheduler->learning_rate;
    ຖ້າ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } ບໍ່ດັ່ງນັ້ນ {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        assert(0.0f <= coeff && coeff <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    ສົ່ງຄືນ lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
float get_learning_rate_linear(LearningRateScheduler *scheduler, ຈຳນວນເຕັມ step) {
    float lr = scheduler->learning_rate;
    ຖ້າ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } ບໍ່ດັ່ງນັ້ນ {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    ສົ່ງຄືນ lr;
}

// constant
float get_learning_rate_constant(LearningRateScheduler *scheduler, ຈຳນວນເຕັມ step) {
    ສົ່ງຄືນ scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
float get_learning_rate_wsd(LearningRateScheduler *scheduler, ຈຳນວນເຕັມ step) {
    ຈຳນວນເຕັມ decay_point = (ຈຳນວນເຕັມ)(0.8f * scheduler->train_num_batches);
    float max_lr = scheduler->learning_rate;
    float lr = max_lr;
    ຖ້າ (step < scheduler->warmup_iterations) {
        float decay_ratio = ((float)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (step < decay_point) {
        // noop, keep lr constant
    } ບໍ່ດັ່ງນັ້ນ {
        float decay_ratio = ((float)(step - decay_point)) / (scheduler->train_num_batches - decay_point);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float min_lr = max_lr * scheduler->final_learning_rate_frac;
        ສົ່ງຄືນ min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    ສົ່ງຄືນ lr;
}

// return the learning rate at a given step
float get_learning_rate(LearningRateScheduler *scheduler, ຈຳນວນເຕັມ step) {
    float step_learning_rate;
    ຖ້າ (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } ບໍ່ດັ່ງນັ້ນ ຖ້າ (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } ບໍ່ດັ່ງນັ້ນ {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    ສົ່ງຄືນ step_learning_rate;
}

#endif // SCHEDULERS_H
