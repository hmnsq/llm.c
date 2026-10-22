/*
Implements various learning rate schedulers.
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#अन्तर्भाव <assert.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

प्रकारव्याख्या संरचना {
    स्थिरांक अक्षर* type;
    दशांश learning_rate;
    पूर्णांक warmup_iterations;
    पूर्णांक train_अंक_batches;
    दशांश final_learning_rate_frac;
} LearningRateScheduler;

शून्य lr_scheduler_init(LearningRateScheduler *scheduler, स्थिरांक अक्षर* scheduler_type, दशांश learning_rate, पूर्णांक warmup_iterations, पूर्णांक train_अंक_batches, दशांश final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_अंक_batches = train_अंक_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

// cosine: warmup linearly to max LR, then cosine decay to LR * final_learning_rate_frac
दशांश get_learning_rate_cosine(LearningRateScheduler *scheduler, पूर्णांक step) {
    दशांश lr = scheduler->learning_rate;
    जर (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((दशांश)(step + 1)) / scheduler->warmup_iterations;
    } अन्यथा {
        दशांश decay_ratio = ((दशांश)(step - scheduler->warmup_iterations)) / (scheduler->train_अंक_batches - scheduler->warmup_iterations);
        प्रतिज्ञा(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        दशांश coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        प्रतिज्ञा(0.0f <= coeff && coeff <= 1.0f);
        दशांश min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    परत lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
दशांश get_learning_rate_linear(LearningRateScheduler *scheduler, पूर्णांक step) {
    दशांश lr = scheduler->learning_rate;
    जर (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((दशांश)(step + 1)) / scheduler->warmup_iterations;
    } अन्यथा {
        दशांश decay_ratio = ((दशांश)(step - scheduler->warmup_iterations)) / (scheduler->train_अंक_batches - scheduler->warmup_iterations);
        प्रतिज्ञा(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        दशांश min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    परत lr;
}

// constant
दशांश get_learning_rate_constant(LearningRateScheduler *scheduler, पूर्णांक step) {
    परत scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
दशांश get_learning_rate_wsd(LearningRateScheduler *scheduler, पूर्णांक step) {
    पूर्णांक decay_point = (पूर्णांक)(0.8f * scheduler->train_अंक_batches);
    दशांश max_lr = scheduler->learning_rate;
    दशांश lr = max_lr;
    जर (step < scheduler->warmup_iterations) {
        दशांश decay_ratio = ((दशांश)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } अन्यथा जर (step < decay_point) {
        // noop, keep lr constant
    } अन्यथा {
        दशांश decay_ratio = ((दशांश)(step - decay_point)) / (scheduler->train_अंक_batches - decay_point);
        प्रतिज्ञा(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        दशांश min_lr = max_lr * scheduler->final_learning_rate_frac;
        परत min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    परत lr;
}

// return the learning rate at a given step
दशांश get_learning_rate(LearningRateScheduler *scheduler, पूर्णांक step) {
    दशांश step_learning_rate;
    जर (म_भेद(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } अन्यथा जर (म_भेद(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } अन्यथा जर (म_भेद(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } अन्यथा जर (म_भेद(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        निकास(निकास_त्रुटि);
    }
    परत step_learning_rate;
}

#endif // SCHEDULERS_H
