/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <string.h>

வகைவரை கட்டமைப்பு {
    மாறிலி எழுத்து* type;
    தசம learning_rate;
    முழு warmup_iterations;
    முழு train_இலக்கம்_batches;
    தசம final_learning_rate_frac;
} LearningRateScheduler;

வெறுமை lr_scheduler_init(LearningRateScheduler *scheduler, மாறிலி எழுத்து* scheduler_type, தசம learning_rate, முழு warmup_iterations, முழு train_இலக்கம்_batches, தசம final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_இலக்கம்_batches = train_இலக்கம்_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
தசம get_learning_rate_cosine(LearningRateScheduler *scheduler, முழு step) {
    தசம lr = scheduler->learning_rate;
    எனில் (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((தசம)(step + 1)) / scheduler->warmup_iterations;
    } இல்லையெனில் {
        தசம decay_ratio = ((தசம)(step - scheduler->warmup_iterations)) / (scheduler->train_இலக்கம்_batches - scheduler->warmup_iterations);
        ப்ரதிஜ்ஞா(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        தசம coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        ப்ரதிஜ்ஞா(0.0f <= coeff && coeff <= 1.0f);
        தசம min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    திரும்பு lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
தசம get_learning_rate_linear(LearningRateScheduler *scheduler, முழு step) {
    தசம lr = scheduler->learning_rate;
    எனில் (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((தசம)(step + 1)) / scheduler->warmup_iterations;
    } இல்லையெனில் {
        தசம decay_ratio = ((தசம)(step - scheduler->warmup_iterations)) / (scheduler->train_இலக்கம்_batches - scheduler->warmup_iterations);
        ப்ரதிஜ்ஞா(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        தசம min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    திரும்பு lr;
}

// constant
தசம get_learning_rate_constant(LearningRateScheduler *scheduler, முழு step) {
    திரும்பு scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
தசம get_learning_rate_wsd(LearningRateScheduler *scheduler, முழு step) {
    முழு decay_point = (முழு)(0.8f * scheduler->train_இலக்கம்_batches);
    தசம max_lr = scheduler->learning_rate;
    தசம lr = max_lr;
    எனில் (step < scheduler->warmup_iterations) {
        தசம decay_ratio = ((தசம)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } இல்லையெனில் எனில் (step < decay_point) {
        // noop, keep lr constant
    } இல்லையெனில் {
        தசம decay_ratio = ((தசம)(step - decay_point)) / (scheduler->train_இலக்கம்_batches - decay_point);
        ப்ரதிஜ்ஞா(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        தசம min_lr = max_lr * scheduler->final_learning_rate_frac;
        திரும்பு min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    திரும்பு lr;
}

// return the learning rate at a given step
தசம get_learning_rate(LearningRateScheduler *scheduler, முழு step) {
    தசம step_learning_rate;
    எனில் (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } இல்லையெனில் எனில் (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } இல்லையெனில் எனில் (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } இல்லையெனில் எனில் (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } இல்லையெனில் {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    திரும்பு step_learning_rate;
}

#endif // SCHEDULERS_H
