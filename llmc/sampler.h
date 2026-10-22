/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <math.h>

// Simple xorshift RNG
unsigned ཧྲིལ random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ལོག (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    ལོག (random_u32(state) >> 8) / 16777216.0f;
}

ཧྲིལ sample_softmax(const float* logits, ཧྲིལ n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    dviguu_na norm = 0;
    དོན་ལུ (ཧྲིལ i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    དོན་ལུ (ཧྲིལ i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        གལ་སྲིད (coin < cdf) {
            ལོག i;
        }
    }
    ལོག n - 1; // in case of rounding errors
}

#endif