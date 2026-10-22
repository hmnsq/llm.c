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

#ضمن <math.h>

#define MERSENNE_STATE_M 397u
#define MERSENNE_STATE_N 624u

#define LMASK 0x7ffffffful
#define UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

تعريفنوع بنية {
    unsigned long long seed_;
    صحيح بایاں;
    unsigned صحيح next_;
    unsigned صحيح state_[MERSENNE_STATE_N];
    unsigned صحيح MATRIX_A[2];
} mt19937_state;

فراغ manual_seed(mt19937_state* state, unsigned صحيح seed) {
    state->MATRIX_A[0] = 0x0u;
    state->MATRIX_A[1] = 0x9908b0df;
    state->state_[0] = seed & 0xffffffff;
    لكل (unsigned صحيح j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->بایاں = 1;
    state->next_ = 0;
}

فراغ next_state(mt19937_state* state) {
    state->بایاں = MERSENNE_STATE_N;
    state->next_ = 0;
    unsigned صحيح y, j;
    لكل (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    لكل (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
}

unsigned صحيح randint32(mt19937_state* state) {
    إذا (!state) أرجع 0;
    إذا (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df) manual_seed(state, 5489); // auto-initialize
    إذا (--state->بایاں <= 0) {
        next_state(state);
    }
    unsigned صحيح y = state->state_[state->next_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    أرجع y;
}

مضمن unsigned long long randint64(mt19937_state* state) {
    أرجع (((unsigned long long)(randint32(state)) << 32) | randint32(state));
}

مضمن عائم randfloat32(mt19937_state* state) {
    أرجع (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

مضمن مزدوج randfloat64(mt19937_state* state) {
    أرجع (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

فراغ uniform_(عائم* data, unsigned صحيح numel, عائم from, عائم to, mt19937_state* state) {
    لكل (unsigned صحيح t = 0; t < numel; t++) {
        data[t] = randfloat32(state) * (to - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
فراغ normal_fill_16(عائم* data, عائم mean, عائم std) {
    #define EPSILONE 1e-12f
    لكل (unsigned صحيح t = 0; t < 8; t++) {
        عائم u1 = 1 - data[t];
        عائم u2 = data[t + 8];
        عائم radius = sqrtf(-2 * logf(u1 + EPSILONE));
        عائم theta = (عائم) (2.0 * M_PI * u2);
        data[t] = (radius * cosf(theta) * std + mean);
        data[t + 8] = (radius * sinf(theta) * std + mean);
    }
}

فراغ normal_fill(عائم* data, unsigned صحيح numel, عائم mean, عائم std, mt19937_state* state) {
    لكل (unsigned صحيح t = 0; t < numel; t++) {
        data[t] = randfloat32(state);
    }
    لكل (unsigned صحيح i = 0; i < numel - 15; i += 16) {
        normal_fill_16(data + i, mean, std);
    }
    إذا (numel % 16 != 0) {
        // recompute the last 16 values
        data = data + numel - 16;
        لكل (unsigned صحيح i = 0; i < 16; i++) {
            data[i] = randfloat32(state);
        }
        normal_fill_16(data, mean, std);
    }
}

فراغ normal_(عائم* data, unsigned صحيح numel, عائم mean, عائم std, mt19937_state* state) {
    #define EPSILONE 1e-12f
    إذا (numel >= 16) {
        normal_fill(data, numel, mean, std, state);
    }
    وإلا {
        مزدوج next_double_normal_sample = 0.0; // make compiler warning happy, won't be used
        صحيح has_next_double_normal_sample = 0;
        لكل (unsigned صحيح  t = 0; t < numel; t++) {
            إذا (has_next_double_normal_sample) {
                data[t] = (عائم)(next_double_normal_sample * std + mean);
                has_next_double_normal_sample = 0;
                استمر;
            }
            // for numel < 16 we draw a double (float64)
            عائم u1 = (عائم) randfloat64(state);
            عائم u2 = (عائم) randfloat64(state);
            عائم radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            عائم theta = (عائم) (2.0 * M_PI * u1);
            next_double_normal_sample = radius * sinf(theta);
            has_next_double_normal_sample = 1;
            data[t] = (radius * cosf(theta) * std + mean);
        }
    }
}

فراغ init_identity_permutation(صحيح *data, صحيح numel) {
    لكل (صحيح i = 0; i < numel; i++) {
        data[i] = i;
    }
}

فراغ random_permutation(صحيح* data, صحيح numel, mt19937_state* state) {
    لكل (صحيح i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        صحيح j = randint32(state) % (i + 1);
        // swap i <-> j
        صحيح tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

#endif