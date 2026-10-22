/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#अगर_अघोषित SAMPLER_H
#घोषणा SAMPLER_H

#समावेश <math.h>

// Simple xorshift RNG
अचिन्हित पूर्णांक random_u32(अचिन्हित दीर्घ दीर्घ *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    वापस (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

भग्न random_f32(अचिन्हित दीर्घ दीर्घ *state) { // random float32 in [0,1)
    वापस (random_u32(state) >> 8) / 16777216.0f;
}

पूर्णांक sample_softmax(स्थिर भग्न* logits, पूर्णांक n, भग्न coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    द्विगुन norm = 0;
    क्रम (पूर्णांक i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    भग्न cdf = 0.0f;
    क्रम (पूर्णांक i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        अगर (coin < cdf) {
            वापस i;
        }
    }
    वापस n - 1; // in case of rounding errors
}

#पूर्ण_अगर