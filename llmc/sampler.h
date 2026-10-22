/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <math.h>

// Simple xorshift RNG
unsigned số_nguyên random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    trả_về (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    trả_về (random_u32(state) >> 8) / 16777216.0f;
}

số_nguyên sample_softmax(const float* logits, số_nguyên n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    cho (số_nguyên i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    cho (số_nguyên i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        nếu (coin < cdf) {
            trả_về i;
        }
    }
    trả_về n - 1; // in case of rounding errors
}

#endif