/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#شامل <math.h>

// Simple xorshift RNG
unsigned صحیح random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    بازگشت (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

اعشاری random_f32(unsigned long long *state) { // random float32 in [0,1)
    بازگشت (random_u32(state) >> 8) / 16777216.0f;
}

صحیح sample_softmax(ثابت اعشاری* logits, صحیح n, اعشاری coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    دوگانه norm = 0;
    برای (صحیح i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    اعشاری cdf = 0.0f;
    برای (صحیح i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        اگر (coin < cdf) {
            بازگشت i;
        }
    }
    بازگشت n - 1; // in case of rounding errors
}

#endif