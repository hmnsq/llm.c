/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <string.h>

প্রকারনির্দেশ সংরচনা {
    ধ্রুব বর্ণ* type;
    ভিন্ন learning_rate;
    পূর্ণাঙ্ক warmup_iterations;
    পূর্ণাঙ্ক train_অঙ্ক_batches;
    ভিন্ন final_learning_rate_frac;
} LearningRateScheduler;

শূন্য lr_scheduler_init(LearningRateScheduler *scheduler, ধ্রুব বর্ণ* scheduler_type, ভিন্ন learning_rate, পূর্ণাঙ্ক warmup_iterations, পূর্ণাঙ্ক train_অঙ্ক_batches, ভিন্ন final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_অঙ্ক_batches = train_অঙ্ক_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
ভিন্ন get_learning_rate_cosine(LearningRateScheduler *scheduler, পূর্ণাঙ্ক step) {
    ভিন্ন lr = scheduler->learning_rate;
    যদি (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ভিন্ন)(step + 1)) / scheduler->warmup_iterations;
    } অন্যথা {
        ভিন্ন decay_ratio = ((ভিন্ন)(step - scheduler->warmup_iterations)) / (scheduler->train_অঙ্ক_batches - scheduler->warmup_iterations);
        প্রতিজ্ঞা(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ভিন্ন coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        প্রতিজ্ঞা(0.0f <= coeff && coeff <= 1.0f);
        ভিন্ন min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    প্রত্যাগম lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
ভিন্ন get_learning_rate_linear(LearningRateScheduler *scheduler, পূর্ণাঙ্ক step) {
    ভিন্ন lr = scheduler->learning_rate;
    যদি (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ভিন্ন)(step + 1)) / scheduler->warmup_iterations;
    } অন্যথা {
        ভিন্ন decay_ratio = ((ভিন্ন)(step - scheduler->warmup_iterations)) / (scheduler->train_অঙ্ক_batches - scheduler->warmup_iterations);
        প্রতিজ্ঞা(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ভিন্ন min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    প্রত্যাগম lr;
}

// constant
ভিন্ন get_learning_rate_constant(LearningRateScheduler *scheduler, পূর্ণাঙ্ক step) {
    প্রত্যাগম scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
ভিন্ন get_learning_rate_wsd(LearningRateScheduler *scheduler, পূর্ণাঙ্ক step) {
    পূর্ণাঙ্ক decay_point = (পূর্ণাঙ্ক)(0.8f * scheduler->train_অঙ্ক_batches);
    ভিন্ন max_lr = scheduler->learning_rate;
    ভিন্ন lr = max_lr;
    যদি (step < scheduler->warmup_iterations) {
        ভিন্ন decay_ratio = ((ভিন্ন)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } অন্যথা যদি (step < decay_point) {
        // noop, keep lr constant
    } অন্যথা {
        ভিন্ন decay_ratio = ((ভিন্ন)(step - decay_point)) / (scheduler->train_অঙ্ক_batches - decay_point);
        প্রতিজ্ঞা(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ভিন্ন min_lr = max_lr * scheduler->final_learning_rate_frac;
        প্রত্যাগম min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    প্রত্যাগম lr;
}

// return the learning rate at a given step
ভিন্ন get_learning_rate(LearningRateScheduler *scheduler, পূর্ণাঙ্ক step) {
    ভিন্ন step_learning_rate;
    যদি (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } অন্যথা যদি (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } অন্যথা যদি (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } অন্যথা যদি (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } অন্যথা {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    প্রত্যাগম step_learning_rate;
}

#endif // SCHEDULERS_H
