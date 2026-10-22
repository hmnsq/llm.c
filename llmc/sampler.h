/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#અન્તર્ભાવ <math.h>

// Simple xorshift RNG
unsigned પૂર્ણાંક random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    પાછા (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

દશાંશ random_f32(unsigned long long *state) { // random float32 in [0,1)
    પાછા (random_u32(state) >> 8) / 16777216.0f;
}

પૂર્ણાંક sample_softmax(અચલ દશાંશ* logits, પૂર્ણાંક n, દશાંશ coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    દ્વિગુણ norm = 0;
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    દશાંશ cdf = 0.0f;
    માટે (પૂર્ણાંક i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        જો (coin < cdf) {
            પાછા i;
        }
    }
    પાછા n - 1; // in case of rounding errors
}

#endif