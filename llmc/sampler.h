/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#अन्तर्भाव <math.h>

// Simple xorshift RNG
अचिन्हित पूर्णाङ्क random_u32(अचिन्हित दीर्घ दीर्घ *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    प्रत्यागम (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

भिन्न random_f32(अचिन्हित दीर्घ दीर्घ *state) { // random float32 in [0,1)
    प्रत्यागम (random_u32(state) >> 8) / 16777216.0f;
}

पूर्णाङ्क sample_softmax(ध्रुव भिन्न* logits, पूर्णाङ्क n, भिन्न coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    द्विगुण norm = 0;
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    भिन्न cdf = 0.0f;
    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        यदि (coin < cdf) {
            प्रत्यागम i;
        }
    }
    प्रत्यागम n - 1; // in case of rounding errors
}

#endif