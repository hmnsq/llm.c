/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

प्रकारनिर्देश संरचना {
    ध्रुव वर्ण* type;
    भिन्न learning_rate;
    पूर्णाङ्क warmup_iterations;
    पूर्णाङ्क train_अङ्कः_batches;
    भिन्न final_learning_rate_frac;
} LearningRateScheduler;

शून्य lr_scheduler_init(LearningRateScheduler *scheduler, ध्रुव वर्ण* scheduler_type, भिन्न learning_rate, पूर्णाङ्क warmup_iterations, पूर्णाङ्क train_अङ्कः_batches, भिन्न final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_अङ्कः_batches = train_अङ्कः_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
भिन्न get_learning_rate_cosine(LearningRateScheduler *scheduler, पूर्णाङ्क step) {
    भिन्न lr = scheduler->learning_rate;
    जइ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((भिन्न)(step + 1)) / scheduler->warmup_iterations;
    } अण्णहा {
        भिन्न decay_ratio = ((भिन्न)(step - scheduler->warmup_iterations)) / (scheduler->train_अङ्कः_batches - scheduler->warmup_iterations);
        प्रतिज्ञा(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        भिन्न coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        प्रतिज्ञा(0.0f <= coeff && coeff <= 1.0f);
        भिन्न min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    प्रत्यागम lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
भिन्न get_learning_rate_linear(LearningRateScheduler *scheduler, पूर्णाङ्क step) {
    भिन्न lr = scheduler->learning_rate;
    जइ (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((भिन्न)(step + 1)) / scheduler->warmup_iterations;
    } अण्णहा {
        भिन्न decay_ratio = ((भिन्न)(step - scheduler->warmup_iterations)) / (scheduler->train_अङ्कः_batches - scheduler->warmup_iterations);
        प्रतिज्ञा(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        भिन्न min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    प्रत्यागम lr;
}

// constant
भिन्न get_learning_rate_constant(LearningRateScheduler *scheduler, पूर्णाङ्क step) {
    प्रत्यागम scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
भिन्न get_learning_rate_wsd(LearningRateScheduler *scheduler, पूर्णाङ्क step) {
    पूर्णाङ्क decay_point = (पूर्णाङ्क)(0.8f * scheduler->train_अङ्कः_batches);
    भिन्न max_lr = scheduler->learning_rate;
    भिन्न lr = max_lr;
    जइ (step < scheduler->warmup_iterations) {
        भिन्न decay_ratio = ((भिन्न)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } अण्णहा जइ (step < decay_point) {
        // noop, keep lr constant
    } अण्णहा {
        भिन्न decay_ratio = ((भिन्न)(step - decay_point)) / (scheduler->train_अङ्कः_batches - decay_point);
        प्रतिज्ञा(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        भिन्न min_lr = max_lr * scheduler->final_learning_rate_frac;
        प्रत्यागम min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    प्रत्यागम lr;
}

// return the learning rate at a given step
भिन्न get_learning_rate(LearningRateScheduler *scheduler, पूर्णाङ्क step) {
    भिन्न step_learning_rate;
    जइ (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } अण्णहा जइ (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } अण्णहा जइ (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } अण्णहा जइ (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } अण्णहा {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    प्रत्यागम step_learning_rate;
}

#endif // SCHEDULERS_H
