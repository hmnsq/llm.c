/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#اگر_غیر_تعریف_شدہ SAMPLER_H
#تعریف_کرو SAMPLER_H

#شامل <math.h>

// Simple xorshift RNG
غیرعلامتی صحیح random_u32(غیرعلامتی طویل طویل *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    واپس (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

اعشاری random_f32(غیرعلامتی طویل طویل *state) { // random float32 in [0,1)
    واپس (random_u32(state) >> 8) / 16777216.0f;
}

صحیح sample_softmax(ثابت اعشاری* logits, صحیح n, اعشاری coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    دوہرا norm = 0;
    برائے (صحیح i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    اعشاری cdf = 0.0f;
    برائے (صحیح i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        اگر (coin < cdf) {
            واپس i;
        }
    }
    واپس n - 1; // in case of rounding errors
}

#اگر_ختم