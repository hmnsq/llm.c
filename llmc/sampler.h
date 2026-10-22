/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#ಅನ್ತರ್ಭಾವ <math.h>

// Simple xorshift RNG
unsigned ಪೂರ್ಣಾಙ್ಕ random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ಪ್ರತ್ಯಾಗಮ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

ಭಿನ್ನ random_f32(unsigned long long *state) { // random float32 in [0,1)
    ಪ್ರತ್ಯಾಗಮ (random_u32(state) >> 8) / 16777216.0f;
}

ಪೂರ್ಣಾಙ್ಕ sample_softmax(ಧ್ರುವ ಭಿನ್ನ* logits, ಪೂರ್ಣಾಙ್ಕ n, ಭಿನ್ನ coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    ದ್ವಿಗುಣ norm = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    ಭಿನ್ನ cdf = 0.0f;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        ಯದಿ (coin < cdf) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
    }
    ಪ್ರತ್ಯಾಗಮ n - 1; // in case of rounding errors
}

#endif