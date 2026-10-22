/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#অন্তর্ভাব <math.h>

// Simple xorshift RNG
unsigned পূর্ণসংখ্যা random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ফেরত (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

দশমিক random_f32(unsigned long long *state) { // random float32 in [0,1)
    ফেরত (random_u32(state) >> 8) / 16777216.0f;
}

পূর্ণসংখ্যা sample_softmax(ধ্রুবক দশমিক* logits, পূর্ণসংখ্যা n, দশমিক coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    দ্বিগুণ norm = 0;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    দশমিক cdf = 0.0f;
    জন্য (পূর্ণসংখ্যা i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        যদি (coin < cdf) {
            ফেরত i;
        }
    }
    ফেরত n - 1; // in case of rounding errors
}

#endif