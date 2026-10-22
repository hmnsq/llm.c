/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ SAMPLER_H
#ವ್ಯಾಖ್ಯಾನಿಸು SAMPLER_H

#ಅನ್ತರ್ಭಾವ <math.h>

// Simple xorshift RNG
ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ random_u32(ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ಹಿಂತಿರುಗಿ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

ದಶಮಾಂಶ random_f32(ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ *state) { // random float32 in [0,1)
    ಹಿಂತಿರುಗಿ (random_u32(state) >> 8) / 16777216.0f;
}

ಪೂರ್ಣಾಂಕ sample_softmax(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* logits, ಪೂರ್ಣಾಂಕ n, ದಶಮಾಂಶ coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    ದ್ವಿಗುಣ norm = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    ದಶಮಾಂಶ cdf = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        ಆದರೆ (coin < cdf) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    ಹಿಂತಿರುಗಿ n - 1; // in case of rounding errors
}

#ಆದರೆ_ಅಂತ್ಯ