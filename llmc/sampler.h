/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#నిర్వచితం_కాకపోతే SAMPLER_H
#నిర్వచించు SAMPLER_H

#అన్తర్భావ <math.h>

// Simple xorshift RNG
గుర్తులేని పూర్ణాంకం random_u32(గుర్తులేని దీర్ఘ దీర్ఘ *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    తిరిగి (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

దశాంశం random_f32(గుర్తులేని దీర్ఘ దీర్ఘ *state) { // random float32 in [0,1)
    తిరిగి (random_u32(state) >> 8) / 16777216.0f;
}

పూర్ణాంకం sample_softmax(స్థిరరాశి దశాంశం* logits, పూర్ణాంకం n, దశాంశం coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    ద్విగుణ norm = 0;
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    దశాంశం cdf = 0.0f;
    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        ఒకవేళ (coin < cdf) {
            తిరిగి i;
        }
    }
    తిరిగి n - 1; // in case of rounding errors
}

#అయితే_ముగింపు