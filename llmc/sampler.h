/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ SAMPLER_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SAMPLER_H

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>

// Simple xorshift RNG
ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ random_u32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ਵਾਪਸ (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

ਦਸ਼ਮਲਵ random_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ *state) { // random float32 in [0,1)
    ਵਾਪਸ (random_u32(state) >> 8) / 16777216.0f;
}

ਪੂਰਨ sample_softmax(ਅਚਲ ਦਸ਼ਮਲਵ* logits, ਪੂਰਨ n, ਦਸ਼ਮਲਵ coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    ਦੁੱਗਣਾ norm = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    ਦਸ਼ਮਲਵ cdf = 0.0f;
    ਲਈ (ਪੂਰਨ i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        ਜੇ (coin < cdf) {
            ਵਾਪਸ i;
        }
    }
    ਵਾਪਸ n - 1; // in case of rounding errors
}

#ਜੇ_ਅੰਤ