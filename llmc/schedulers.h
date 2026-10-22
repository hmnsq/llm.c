/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#شامل <assert.h>
#شامل <math.h>
#شامل <string.h>

تعریفنوع ساختار {
    ثابت نویسه* type;
    اعشاری learning_rate;
    صحیح warmup_iterations;
    صحیح train_رقم_batches;
    اعشاری final_learning_rate_frac;
} LearningRateScheduler;

تهی lr_scheduler_init(LearningRateScheduler *scheduler, ثابت نویسه* scheduler_type, اعشاری learning_rate, صحیح warmup_iterations, صحیح train_رقم_batches, اعشاری final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_رقم_batches = train_رقم_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
اعشاری get_learning_rate_cosine(LearningRateScheduler *scheduler, صحیح step) {
    اعشاری lr = scheduler->learning_rate;
    اگر (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((اعشاری)(step + 1)) / scheduler->warmup_iterations;
    } وگرنه {
        اعشاری decay_ratio = ((اعشاری)(step - scheduler->warmup_iterations)) / (scheduler->train_رقم_batches - scheduler->warmup_iterations);
        أكد(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        اعشاری coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        أكد(0.0f <= coeff && coeff <= 1.0f);
        اعشاری min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    بازگشت lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
اعشاری get_learning_rate_linear(LearningRateScheduler *scheduler, صحیح step) {
    اعشاری lr = scheduler->learning_rate;
    اگر (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((اعشاری)(step + 1)) / scheduler->warmup_iterations;
    } وگرنه {
        اعشاری decay_ratio = ((اعشاری)(step - scheduler->warmup_iterations)) / (scheduler->train_رقم_batches - scheduler->warmup_iterations);
        أكد(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        اعشاری min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    بازگشت lr;
}

// constant
اعشاری get_learning_rate_constant(LearningRateScheduler *scheduler, صحیح step) {
    بازگشت scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
اعشاری get_learning_rate_wsd(LearningRateScheduler *scheduler, صحیح step) {
    صحیح decay_point = (صحیح)(0.8f * scheduler->train_رقم_batches);
    اعشاری max_lr = scheduler->learning_rate;
    اعشاری lr = max_lr;
    اگر (step < scheduler->warmup_iterations) {
        اعشاری decay_ratio = ((اعشاری)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } وگرنه اگر (step < decay_point) {
        // noop, keep lr constant
    } وگرنه {
        اعشاری decay_ratio = ((اعشاری)(step - decay_point)) / (scheduler->train_رقم_batches - decay_point);
        أكد(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        اعشاری min_lr = max_lr * scheduler->final_learning_rate_frac;
        بازگشت min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    بازگشت lr;
}

// return the learning rate at a given step
اعشاری get_learning_rate(LearningRateScheduler *scheduler, صحیح step) {
    اعشاری step_learning_rate;
    اگر (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } وگرنه اگر (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } وگرنه اگر (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } وگرنه اگر (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } وگرنه {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    بازگشت step_learning_rate;
}

#endif // SCHEDULERS_H
