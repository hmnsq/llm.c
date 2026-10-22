/*
Implements various learning rate schedulers.
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SCHEDULERS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCHEDULERS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਅਚਲ ਅੱਖਰ* type;
    ਦਸ਼ਮਲਵ learning_ਦਰ;
    ਪੂਰਨ warmup_iterations;
    ਪੂਰਨ train_ਅੰਕ_batches;
    ਦਸ਼ਮਲਵ final_learning_ਦਰ_frac;
} Learning_ਦਰ_Scheduler;

ਖਾਲੀ lr_scheduler_init(Learning_ਦਰ_Scheduler *scheduler, ਅਚਲ ਅੱਖਰ* scheduler_type, ਦਸ਼ਮਲਵ learning_ਦਰ, ਪੂਰਨ warmup_iterations, ਪੂਰਨ train_ਅੰਕ_batches, ਦਸ਼ਮਲਵ final_learning_ਦਰ_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_ਦਰ = learning_ਦਰ;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_ਅੰਕ_batches = train_ਅੰਕ_batches;
    scheduler->final_learning_ਦਰ_frac = final_learning_ਦਰ_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
ਦਸ਼ਮਲਵ get_learning_ਦਰ_cosine(Learning_ਦਰ_Scheduler *scheduler, ਪੂਰਨ ਕਦਮ) {
    ਦਸ਼ਮਲਵ lr = scheduler->learning_ਦਰ;
    ਜੇ (ਕਦਮ < scheduler->warmup_iterations) {
        lr = scheduler->learning_ਦਰ * ((ਦਸ਼ਮਲਵ)(ਕਦਮ + 1)) / scheduler->warmup_iterations;
    } ਨਹੀਂਤਾਂ {
        ਦਸ਼ਮਲਵ decay_ਅਨੁਪਾਤ = ((ਦਸ਼ਮਲਵ)(ਕਦਮ - scheduler->warmup_iterations)) / (scheduler->train_ਅੰਕ_batches - scheduler->warmup_iterations);
        ਪ੍ਰਤਿਜ੍ਞਾ(0.0f <= decay_ਅਨੁਪਾਤ && decay_ਅਨੁਪਾਤ <= 1.0f);
        ਦਸ਼ਮਲਵ coeff = 0.5f * (1.0f + cosf(M_PI * decay_ਅਨੁਪਾਤ)); // coeff starts at 1 and goes to 0
        ਪ੍ਰਤਿਜ੍ਞਾ(0.0f <= coeff && coeff <= 1.0f);
        ਦਸ਼ਮਲਵ min_lr = scheduler->learning_ਦਰ * scheduler->final_learning_ਦਰ_frac;
        lr = min_lr + coeff * (scheduler->learning_ਦਰ - min_lr);
    }
    ਵਾਪਸ lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
ਦਸ਼ਮਲਵ get_learning_ਦਰ_ਰੇਖਿਕ(Learning_ਦਰ_Scheduler *scheduler, ਪੂਰਨ ਕਦਮ) {
    ਦਸ਼ਮਲਵ lr = scheduler->learning_ਦਰ;
    ਜੇ (ਕਦਮ < scheduler->warmup_iterations) {
        lr = scheduler->learning_ਦਰ * ((ਦਸ਼ਮਲਵ)(ਕਦਮ + 1)) / scheduler->warmup_iterations;
    } ਨਹੀਂਤਾਂ {
        ਦਸ਼ਮਲਵ decay_ਅਨੁਪਾਤ = ((ਦਸ਼ਮਲਵ)(ਕਦਮ - scheduler->warmup_iterations)) / (scheduler->train_ਅੰਕ_batches - scheduler->warmup_iterations);
        ਪ੍ਰਤਿਜ੍ਞਾ(0.0f <= decay_ਅਨੁਪਾਤ && decay_ਅਨੁਪਾਤ <= 1.0f);
        ਦਸ਼ਮਲਵ min_lr = scheduler->learning_ਦਰ * scheduler->final_learning_ਦਰ_frac;
        lr = scheduler->learning_ਦਰ - decay_ਅਨੁਪਾਤ * (scheduler->learning_ਦਰ - min_lr);
    }
    ਵਾਪਸ lr;
}

// constant
ਦਸ਼ਮਲਵ get_learning_ਦਰ_constant(Learning_ਦਰ_Scheduler *scheduler, ਪੂਰਨ ਕਦਮ) {
    ਵਾਪਸ scheduler->learning_ਦਰ;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
ਦਸ਼ਮਲਵ get_learning_ਦਰ_wsd(Learning_ਦਰ_Scheduler *scheduler, ਪੂਰਨ ਕਦਮ) {
    ਪੂਰਨ decay_ਬਿੰਦੂ = (ਪੂਰਨ)(0.8f * scheduler->train_ਅੰਕ_batches);
    ਦਸ਼ਮਲਵ max_lr = scheduler->learning_ਦਰ;
    ਦਸ਼ਮਲਵ lr = max_lr;
    ਜੇ (ਕਦਮ < scheduler->warmup_iterations) {
        ਦਸ਼ਮਲਵ decay_ਅਨੁਪਾਤ = ((ਦਸ਼ਮਲਵ)(ਕਦਮ + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ਅਨੁਪਾਤ;
    } ਨਹੀਂਤਾਂ ਜੇ (ਕਦਮ < decay_ਬਿੰਦੂ) {
        // noop, keep lr constant
    } ਨਹੀਂਤਾਂ {
        ਦਸ਼ਮਲਵ decay_ਅਨੁਪਾਤ = ((ਦਸ਼ਮਲਵ)(ਕਦਮ - decay_ਬਿੰਦੂ)) / (scheduler->train_ਅੰਕ_batches - decay_ਬਿੰਦੂ);
        ਪ੍ਰਤਿਜ੍ਞਾ(0.0f <= decay_ਅਨੁਪਾਤ && decay_ਅਨੁਪਾਤ <= 1.0f);
        ਦਸ਼ਮਲਵ min_lr = max_lr * scheduler->final_learning_ਦਰ_frac;
        ਵਾਪਸ min_lr + (1.0f - sqrtf(decay_ਅਨੁਪਾਤ)) * (max_lr - min_lr);
    }
    ਵਾਪਸ lr;
}

// return the learning rate at a given step
ਦਸ਼ਮਲਵ get_learning_ਦਰ(Learning_ਦਰ_Scheduler *scheduler, ਪੂਰਨ ਕਦਮ) {
    ਦਸ਼ਮਲਵ ਕਦਮ_learning_ਦਰ;
    ਜੇ (strcmp(scheduler->type, "cosine") == 0) {
        ਕਦਮ_learning_ਦਰ = get_learning_ਦਰ_cosine(scheduler, ਕਦਮ);
    } ਨਹੀਂਤਾਂ ਜੇ (strcmp(scheduler->type, "linear") == 0) {
        ਕਦਮ_learning_ਦਰ = get_learning_ਦਰ_ਰੇਖਿਕ(scheduler, ਕਦਮ);
    } ਨਹੀਂਤਾਂ ਜੇ (strcmp(scheduler->type, "constant") == 0) {
        ਕਦਮ_learning_ਦਰ = get_learning_ਦਰ_constant(scheduler, ਕਦਮ);
    } ਨਹੀਂਤਾਂ ਜੇ (strcmp(scheduler->type, "wsd") == 0) {
        ਕਦਮ_learning_ਦਰ = get_learning_ਦਰ_wsd(scheduler, ਕਦਮ);
    } ਨਹੀਂਤਾਂ {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    ਵਾਪਸ ਕਦਮ_learning_ਦਰ;
}

#ਜੇ_ਅੰਤ // SCHEDULERS_H
