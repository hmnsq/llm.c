/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#அந்தர்பாவ <math.h>

// Simple xorshift RNG
unsigned முழு random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    திரும்பு (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

தசம random_f32(unsigned long long *state) { // random float32 in [0,1)
    திரும்பு (random_u32(state) >> 8) / 16777216.0f;
}

முழு sample_softmax(மாறிலி தசம* logits, முழு n, தசம coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    இரட்டை norm = 0;
    க்கு (முழு i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    தசம cdf = 0.0f;
    க்கு (முழு i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        எனில் (coin < cdf) {
            திரும்பு i;
        }
    }
    திரும்பு n - 1; // in case of rounding errors
}

#endif