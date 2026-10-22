/*
Mersenne Twisters implementation, numerically identical to torch.

Example usage:

    mt19937_state state;
    manual_seed(&state, 137);
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));
    printf("%u\n", randint32(&state));

    float t8[8];
    normal_(t8, 8, 0, 1, &state);
    for (int i = 0; i < 8; i++) {
        printf("%f\n", t8[i]);
    }
    printf("%u\n", randint32(&state));

    float t16[16];
    normal_(t16, 16, 0, 1, &state);
    for (int i = 0; i < 16; i++) {
        printf("%f\n", t16[i]);
    }
    printf("%u\n", randint32(&state));

PyTorch reference (producing identical results):

    import torch
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    t = torch.zeros(8);
    t.normal_()
    for i in range(len(t)) :
        print(t[i].item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    t = torch.zeros(16);
    t.normal_()
    for i in range(len(t)) :
        print(t[i].item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())

Both output:

    4053805790
    2173880614
    380293709
    1237255315
    2986595568
    0.7947664260864258
    1.4369317293167114
    - 0.2292192131280899
    0.47556325793266296
    - 0.6334410905838013
    - 0.5791953802108765
    - 0.0925704762339592
    - 0.8659197092056274
    2186503452
    - 1.2813878059387207
    - 2.646395683288574
    - 0.06569503247737885
    0.2180829495191574
    - 0.46536165475845337
    - 0.33108410239219666
    2.5485482215881348
    0.10425379872322083
    0.8460659980773926
    0.9462448358535767
    - 0.2913765013217926
    0.34313806891441345
    - 1.1186704635620117
    - 0.18305328488349915
    - 2.3153159618377686
    0.3961987793445587
    2756748748
*/

#ifndef RAND_H
#define RAND_H

#অন্তর্ভাব <math.h>

#define MERSENNE_STATE_M 397u
#define MERSENNE_STATE_N 624u

#define LMASK 0x7ffffffful
#define UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

প্রকারলেখ কাঠামো {
    unsigned long long seed_;
    পূর্ণসংখ্যা বাম;
    unsigned পূর্ণসংখ্যা পরবর্তী;
    unsigned পূর্ণসংখ্যা state_[MERSENNE_STATE_N];
    unsigned পূর্ণসংখ্যা MATRIX_A[2];
} mt19937_state;

শূন্য manual_seed(mt19937_state* state, unsigned পূর্ণসংখ্যা seed) {
    state->MATRIX_A[0] = 0x0u;
    state->MATRIX_A[1] = 0x9908b0df;
    state->state_[0] = seed & 0xffffffff;
    জন্য (unsigned পূর্ণসংখ্যা j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->বাম = 1;
    state->পরবর্তী = 0;
}

শূন্য পরবর্তী_state(mt19937_state* state) {
    state->বাম = MERSENNE_STATE_N;
    state->পরবর্তী = 0;
    unsigned পূর্ণসংখ্যা y, j;
    জন্য (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    জন্য (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
}

unsigned পূর্ণসংখ্যা randint32(mt19937_state* state) {
    যদি (!state) ফেরত 0;
    যদি (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df) manual_seed(state, 5489); // auto-initialize
    যদি (--state->বাম <= 0) {
        পরবর্তী_state(state);
    }
    unsigned পূর্ণসংখ্যা y = state->state_[state->পরবর্তী++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    ফেরত y;
}

অন্তঃরেখ unsigned long long randint64(mt19937_state* state) {
    ফেরত (((unsigned long long)(randint32(state)) << 32) | randint32(state));
}

অন্তঃরেখ দশমিক randfloat32(mt19937_state* state) {
    ফেরত (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

অন্তঃরেখ দ্বিগুণ randfloat64(mt19937_state* state) {
    ফেরত (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

শূন্য uniform_(দশমিক* তথ্য, unsigned পূর্ণসংখ্যা numel, দশমিক from, দশমিক to, mt19937_state* state) {
    জন্য (unsigned পূর্ণসংখ্যা t = 0; t < numel; t++) {
        তথ্য[t] = randfloat32(state) * (to - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
শূন্য normal_fill_16(দশমিক* তথ্য, দশমিক mean, দশমিক std) {
    #define EPSILONE 1e-12f
    জন্য (unsigned পূর্ণসংখ্যা t = 0; t < 8; t++) {
        দশমিক u1 = 1 - তথ্য[t];
        দশমিক u2 = তথ্য[t + 8];
        দশমিক radius = sqrtf(-2 * logf(u1 + EPSILONE));
        দশমিক theta = (দশমিক) (2.0 * M_PI * u2);
        তথ্য[t] = (radius * cosf(theta) * std + mean);
        তথ্য[t + 8] = (radius * sinf(theta) * std + mean);
    }
}

শূন্য normal_fill(দশমিক* তথ্য, unsigned পূর্ণসংখ্যা numel, দশমিক mean, দশমিক std, mt19937_state* state) {
    জন্য (unsigned পূর্ণসংখ্যা t = 0; t < numel; t++) {
        তথ্য[t] = randfloat32(state);
    }
    জন্য (unsigned পূর্ণসংখ্যা i = 0; i < numel - 15; i += 16) {
        normal_fill_16(তথ্য + i, mean, std);
    }
    যদি (numel % 16 != 0) {
        // recompute the last 16 values
        তথ্য = তথ্য + numel - 16;
        জন্য (unsigned পূর্ণসংখ্যা i = 0; i < 16; i++) {
            তথ্য[i] = randfloat32(state);
        }
        normal_fill_16(তথ্য, mean, std);
    }
}

শূন্য normal_(দশমিক* তথ্য, unsigned পূর্ণসংখ্যা numel, দশমিক mean, দশমিক std, mt19937_state* state) {
    #define EPSILONE 1e-12f
    যদি (numel >= 16) {
        normal_fill(তথ্য, numel, mean, std, state);
    }
    অন্যথা {
        দ্বিগুণ পরবর্তী_double_normal_sample = 0.0; // make compiler warning happy, won't be used
        পূর্ণসংখ্যা has_পরবর্তী_double_normal_sample = 0;
        জন্য (unsigned পূর্ণসংখ্যা  t = 0; t < numel; t++) {
            যদি (has_পরবর্তী_double_normal_sample) {
                তথ্য[t] = (দশমিক)(পরবর্তী_double_normal_sample * std + mean);
                has_পরবর্তী_double_normal_sample = 0;
                চালু;
            }
            // for numel < 16 we draw a double (float64)
            দশমিক u1 = (দশমিক) randfloat64(state);
            দশমিক u2 = (দশমিক) randfloat64(state);
            দশমিক radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            দশমিক theta = (দশমিক) (2.0 * M_PI * u1);
            পরবর্তী_double_normal_sample = radius * sinf(theta);
            has_পরবর্তী_double_normal_sample = 1;
            তথ্য[t] = (radius * cosf(theta) * std + mean);
        }
    }
}

শূন্য init_identity_permutation(পূর্ণসংখ্যা *তথ্য, পূর্ণসংখ্যা numel) {
    জন্য (পূর্ণসংখ্যা i = 0; i < numel; i++) {
        তথ্য[i] = i;
    }
}

শূন্য random_permutation(পূর্ণসংখ্যা* তথ্য, পূর্ণসংখ্যা numel, mt19937_state* state) {
    জন্য (পূর্ণসংখ্যা i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        পূর্ণসংখ্যা j = randint32(state) % (i + 1);
        // swap i <-> j
        পূর্ণসংখ্যা tmp = তথ্য[i];
        তথ্য[i] = তথ্য[j];
        তথ্য[j] = tmp;
    }
}

#endif