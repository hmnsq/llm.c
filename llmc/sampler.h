/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#جے_بنا_تعریف SAMPLER_H
#تعریف_کرو SAMPLER_H

#ضمن <math.h>

// Simple xorshift RNG
بنانشانی صحيح random_u32(بنانشانی لما لما *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    أرجع (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

عائم random_f32(بنانشانی لما لما *state) { // random float32 in [0,1)
    أرجع (random_u32(state) >> 8) / 16777216.0f;
}

صحيح sample_softmax(ثابت عائم* logits, صحيح n, عائم coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    مزدوج norm = 0;
    لكل (صحيح i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    عائم cdf = 0.0f;
    لكل (صحيح i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        إذا (coin < cdf) {
            أرجع i;
        }
    }
    أرجع n - 1; // in case of rounding errors
}

#جے_انت