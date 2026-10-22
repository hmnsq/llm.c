/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#അന്തര്ഭാവ <math.h>

// Simple xorshift RNG
unsigned പൂർണ്ണസംഖ്യ random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    തിരികെ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

ദശാംശം random_f32(unsigned long long *state) { // random float32 in [0,1)
    തിരികെ (random_u32(state) >> 8) / 16777216.0f;
}

പൂർണ്ണസംഖ്യ sample_softmax(സ്ഥിരം ദശാംശം* logits, പൂർണ്ണസംഖ്യ n, ദശാംശം coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    ഇരട്ട norm = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    ദശാംശം cdf = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        എങ്കിൽ (coin < cdf) {
            തിരികെ i;
        }
    }
    തിരികെ n - 1; // in case of rounding errors
}

#endif