/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <math.h>

// Simple xorshift RNG
unsigned ຈຳນວນເຕັມ random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ສົ່ງຄືນ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    ສົ່ງຄືນ (random_u32(state) >> 8) / 16777216.0f;
}

ຈຳນວນເຕັມ sample_softmax(const float* logits, ຈຳນວນເຕັມ n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    dviguu_na norm = 0;
    ສຳລັບ (ຈຳນວນເຕັມ i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    ສຳລັບ (ຈຳນວນເຕັມ i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        ຖ້າ (coin < cdf) {
            ສົ່ງຄືນ i;
        }
    }
    ສົ່ງຄືນ n - 1; // in case of rounding errors
}

#endif