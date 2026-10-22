/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#ଅନ୍ତର୍ଭାଵ <math.h>

// Simple xorshift RNG
unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ଫେରନ୍ତୁ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

ଦଶମିକ random_f32(unsigned long long *state) { // random float32 in [0,1)
    ଫେରନ୍ତୁ (random_u32(state) >> 8) / 16777216.0f;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା sample_softmax(ଅଚଳ ଦଶମିକ* logits, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ଦଶମିକ coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    ଦ୍ୱିଗୁଣ norm = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    ଦଶମିକ cdf = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        ଯଦି (coin < cdf) {
            ଫେରନ୍ତୁ i;
        }
    }
    ଫେରନ୍ତୁ n - 1; // in case of rounding errors
}

#endif