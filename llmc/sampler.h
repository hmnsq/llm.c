/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#অন্তর্ভাৱ <math.h>

// Simple xorshift RNG
unsigned পূর্ণাঙ্ক random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ঘূৰাই (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

ভিন্ন random_f32(unsigned long long *state) { // random float32 in [0,1)
    ঘূৰাই (random_u32(state) >> 8) / 16777216.0f;
}

পূর্ণাঙ্ক sample_softmax(ধ্রুৱ ভিন্ন* logits, পূর্ণাঙ্ক n, ভিন্ন coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    দ্ৱিগুণ norm = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    ভিন্ন cdf = 0.0f;
    বাবে (পূর্ণাঙ্ক i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        যদি (coin < cdf) {
            ঘূৰাই i;
        }
    }
    ঘূৰাই n - 1; // in case of rounding errors
}

#endif