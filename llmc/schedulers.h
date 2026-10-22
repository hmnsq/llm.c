/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ଅଚଳ ଅକ୍ଷର* type;
    ଦଶମିକ learning_rate;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା warmup_iterations;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା train_ଅଙ୍କ_batches;
    ଦଶମିକ final_learning_rate_frac;
} LearningRateScheduler;

ଶୂନ୍ୟ lr_scheduler_init(LearningRateScheduler *scheduler, ଅଚଳ ଅକ୍ଷର* scheduler_type, ଦଶମିକ learning_rate, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warmup_iterations, ପୂର୍ଣ୍ଣସଂଖ୍ୟା train_ଅଙ୍କ_batches, ଦଶମିକ final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_ଅଙ୍କ_batches = train_ଅଙ୍କ_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
ଦଶମିକ get_learning_rate_cosine(LearningRateScheduler *scheduler, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step) {
    ଦଶମିକ lr = scheduler->learning_rate;
    ଯଦି (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ଦଶମିକ)(step + 1)) / scheduler->warmup_iterations;
    } ଅନ୍ୟଥା {
        ଦଶମିକ decay_ratio = ((ଦଶମିକ)(step - scheduler->warmup_iterations)) / (scheduler->train_ଅଙ୍କ_batches - scheduler->warmup_iterations);
        ପ୍ରତିଜ୍ଞା(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ଦଶମିକ coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        ପ୍ରତିଜ୍ଞା(0.0f <= coeff && coeff <= 1.0f);
        ଦଶମିକ min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    ଫେରନ୍ତୁ lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
ଦଶମିକ get_learning_rate_linear(LearningRateScheduler *scheduler, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step) {
    ଦଶମିକ lr = scheduler->learning_rate;
    ଯଦି (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((ଦଶମିକ)(step + 1)) / scheduler->warmup_iterations;
    } ଅନ୍ୟଥା {
        ଦଶମିକ decay_ratio = ((ଦଶମିକ)(step - scheduler->warmup_iterations)) / (scheduler->train_ଅଙ୍କ_batches - scheduler->warmup_iterations);
        ପ୍ରତିଜ୍ଞା(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ଦଶମିକ min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    ଫେରନ୍ତୁ lr;
}

// constant
ଦଶମିକ get_learning_rate_constant(LearningRateScheduler *scheduler, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step) {
    ଫେରନ୍ତୁ scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
ଦଶମିକ get_learning_rate_wsd(LearningRateScheduler *scheduler, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା decay_point = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)(0.8f * scheduler->train_ଅଙ୍କ_batches);
    ଦଶମିକ max_lr = scheduler->learning_rate;
    ଦଶମିକ lr = max_lr;
    ଯଦି (step < scheduler->warmup_iterations) {
        ଦଶମିକ decay_ratio = ((ଦଶମିକ)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } ଅନ୍ୟଥା ଯଦି (step < decay_point) {
        // noop, keep lr constant
    } ଅନ୍ୟଥା {
        ଦଶମିକ decay_ratio = ((ଦଶମିକ)(step - decay_point)) / (scheduler->train_ଅଙ୍କ_batches - decay_point);
        ପ୍ରତିଜ୍ଞା(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        ଦଶମିକ min_lr = max_lr * scheduler->final_learning_rate_frac;
        ଫେରନ୍ତୁ min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    ଫେରନ୍ତୁ lr;
}

// return the learning rate at a given step
ଦଶମିକ get_learning_rate(LearningRateScheduler *scheduler, ପୂର୍ଣ୍ଣସଂଖ୍ୟା step) {
    ଦଶମିକ step_learning_rate;
    ଯଦି (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } ଅନ୍ୟଥା ଯଦି (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } ଅନ୍ୟଥା ଯଦି (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } ଅନ୍ୟଥା ଯଦି (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } ଅନ୍ୟଥା {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    ଫେରନ୍ତୁ step_learning_rate;
}

#endif // SCHEDULERS_H
