/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    ಧ್ರುವ ವರ್ಣ* type;
    ಭಿನ್ನ learning_rate;
    ಪೂರ್ಣಾಙ್ಕ warmup_iterations;
    ಪೂರ್ಣಾಙ್ಕ train_ಅಂಕೆ_batches;
    ಭಿನ್ನ final_learning_rate_frac;
} LearningRateScheduler;

ಶೂನ್ಯ lr_scheduler_init(LearningRateScheduler *scheduler, ಧ್ರುವ ವರ್ಣ* scheduler_type, ಭಿನ್ನ learning_rate, ಪೂರ್ಣಾಙ್ಕ warmup_iterations, ಪೂರ್ಣಾಙ್ಕ train_ಅಂಕೆ_batches, ಭಿನ್ನ final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_ಅಂಕೆ_batches = train_ಅಂಕೆ_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
ಭಿನ್ನ get_learning_rate_cosine(LearningRateScheduler *scheduler, ಪೂರ್ಣಾಙ್ಕ step) {
    ಭಿನ್ನ lr = scheduler->learning_rate;
    ಯದಿ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ಭಿನ್ನ)(step + 1)) / scheduler->warmup_iterations;
    } ಅನ್ಯಥಾ {
        ಭಿನ್ನ decay_ratio = ((ಭಿನ್ನ)(step - scheduler->warmup_iterations)) / (scheduler->train_ಅಂಕೆ_batches - scheduler->warmup_iterations);
        ಪ್ರತಿಜ್ಞಾ(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ಭಿನ್ನ coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        ಪ್ರತಿಜ್ಞಾ(0.0f <= coeff && coeff <= 1.0f);
        ಭಿನ್ನ min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    ಪ್ರತ್ಯಾಗಮ lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
ಭಿನ್ನ get_learning_rate_linear(LearningRateScheduler *scheduler, ಪೂರ್ಣಾಙ್ಕ step) {
    ಭಿನ್ನ lr = scheduler->learning_rate;
    ಯದಿ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ಭಿನ್ನ)(step + 1)) / scheduler->warmup_iterations;
    } ಅನ್ಯಥಾ {
        ಭಿನ್ನ decay_ratio = ((ಭಿನ್ನ)(step - scheduler->warmup_iterations)) / (scheduler->train_ಅಂಕೆ_batches - scheduler->warmup_iterations);
        ಪ್ರತಿಜ್ಞಾ(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ಭಿನ್ನ min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    ಪ್ರತ್ಯಾಗಮ lr;
}

// constant
ಭಿನ್ನ get_learning_rate_constant(LearningRateScheduler *scheduler, ಪೂರ್ಣಾಙ್ಕ step) {
    ಪ್ರತ್ಯಾಗಮ scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
ಭಿನ್ನ get_learning_rate_wsd(LearningRateScheduler *scheduler, ಪೂರ್ಣಾಙ್ಕ step) {
    ಪೂರ್ಣಾಙ್ಕ decay_point = (ಪೂರ್ಣಾಙ್ಕ)(0.8f * scheduler->train_ಅಂಕೆ_batches);
    ಭಿನ್ನ max_lr = scheduler->learning_rate;
    ಭಿನ್ನ lr = max_lr;
    ಯದಿ (step < scheduler->warmup_iterations) {
        ಭಿನ್ನ decay_ratio = ((ಭಿನ್ನ)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } ಅನ್ಯಥಾ ಯದಿ (step < decay_point) {
        // noop, keep lr constant
    } ಅನ್ಯಥಾ {
        ಭಿನ್ನ decay_ratio = ((ಭಿನ್ನ)(step - decay_point)) / (scheduler->train_ಅಂಕೆ_batches - decay_point);
        ಪ್ರತಿಜ್ಞಾ(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ಭಿನ್ನ min_lr = max_lr * scheduler->final_learning_rate_frac;
        ಪ್ರತ್ಯಾಗಮ min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    ಪ್ರತ್ಯಾಗಮ lr;
}

// return the learning rate at a given step
ಭಿನ್ನ get_learning_rate(LearningRateScheduler *scheduler, ಪೂರ್ಣಾಙ್ಕ step) {
    ಭಿನ್ನ step_learning_rate;
    ಯದಿ (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } ಅನ್ಯಥಾ ಯದಿ (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } ಅನ್ಯಥಾ ಯದಿ (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } ಅನ್ಯಥಾ ಯದಿ (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } ಅನ್ಯಥಾ {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    ಪ್ರತ್ಯಾಗಮ step_learning_rate;
}

#endif // SCHEDULERS_H
