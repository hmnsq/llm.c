/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <string.h>

પ્રકારવ્યાખ્યા સંરચના {
    અચલ અક્ષર* type;
    દશાંશ learning_rate;
    પૂર્ણાંક warmup_iterations;
    પૂર્ણાંક train_અંક_batches;
    દશાંશ final_learning_rate_frac;
} LearningRateScheduler;

શૂન્ય lr_scheduler_init(LearningRateScheduler *scheduler, અચલ અક્ષર* scheduler_type, દશાંશ learning_rate, પૂર્ણાંક warmup_iterations, પૂર્ણાંક train_અંક_batches, દશાંશ final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_અંક_batches = train_અંક_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
દશાંશ get_learning_rate_cosine(LearningRateScheduler *scheduler, પૂર્ણાંક step) {
    દશાંશ lr = scheduler->learning_rate;
    જો (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((દશાંશ)(step + 1)) / scheduler->warmup_iterations;
    } નહીંતર {
        દશાંશ decay_ratio = ((દશાંશ)(step - scheduler->warmup_iterations)) / (scheduler->train_અંક_batches - scheduler->warmup_iterations);
        પ્રતિજ્ઞા(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        દશાંશ coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        પ્રતિજ્ઞા(0.0f <= coeff && coeff <= 1.0f);
        દશાંશ min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    પાછા lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
દશાંશ get_learning_rate_linear(LearningRateScheduler *scheduler, પૂર્ણાંક step) {
    દશાંશ lr = scheduler->learning_rate;
    જો (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((દશાંશ)(step + 1)) / scheduler->warmup_iterations;
    } નહીંતર {
        દશાંશ decay_ratio = ((દશાંશ)(step - scheduler->warmup_iterations)) / (scheduler->train_અંક_batches - scheduler->warmup_iterations);
        પ્રતિજ્ઞા(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        દશાંશ min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    પાછા lr;
}

// constant
દશાંશ get_learning_rate_constant(LearningRateScheduler *scheduler, પૂર્ણાંક step) {
    પાછા scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
દશાંશ get_learning_rate_wsd(LearningRateScheduler *scheduler, પૂર્ણાંક step) {
    પૂર્ણાંક decay_point = (પૂર્ણાંક)(0.8f * scheduler->train_અંક_batches);
    દશાંશ max_lr = scheduler->learning_rate;
    દશાંશ lr = max_lr;
    જો (step < scheduler->warmup_iterations) {
        દશાંશ decay_ratio = ((દશાંશ)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } નહીંતર જો (step < decay_point) {
        // noop, keep lr constant
    } નહીંતર {
        દશાંશ decay_ratio = ((દશાંશ)(step - decay_point)) / (scheduler->train_અંક_batches - decay_point);
        પ્રતિજ્ઞા(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        દશાંશ min_lr = max_lr * scheduler->final_learning_rate_frac;
        પાછા min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    પાછા lr;
}

// return the learning rate at a given step
દશાંશ get_learning_rate(LearningRateScheduler *scheduler, પૂર્ણાંક step) {
    દશાંશ step_learning_rate;
    જો (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } નહીંતર જો (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } નહીંતર જો (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } નહીંતર જો (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } નહીંતર {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    પાછા step_learning_rate;
}

#endif // SCHEDULERS_H
