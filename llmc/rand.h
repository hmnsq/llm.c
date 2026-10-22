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

#جيڪڏهن_وصف_نه_ٿيل RAND_H
#وصف_ڪريو RAND_H

#ضمن <math.h>

#وصف_ڪريو MERSENNE_STATE_M 397u
#وصف_ڪريو MERSENNE_STATE_N 624u

#وصف_ڪريو LMASK 0x7ffffffful
#وصف_ڪريو UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

تعريفنوع بنية {
    بنانشان ڊگهو ڊگهو ٻج;
    صحيح کاٻو;
    بنانشان صحيح ايندڙ;
    بنانشان صحيح state_[MERSENNE_STATE_N];
    بنانشان صحيح ميٽرڪس_A[2];
} mt19937_state;

فراغ manual_ٻج(mt19937_state* state, بنانشان صحيح ٻج_2) {
    state->ميٽرڪس_A[0] = 0x0u;
    state->ميٽرڪس_A[1] = 0x9908b0df;
    state->state_[0] = ٻج_2 & 0xffffffff;
    لكل (بنانشان صحيح j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->کاٻو = 1;
    state->ايندڙ = 0;
}

فراغ ايندڙ_state(mt19937_state* state) {
    state->کاٻو = MERSENNE_STATE_N;
    state->ايندڙ = 0;
    بنانشان صحيح y, j;
    لكل (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->ميٽرڪس_A[y & 0x1];
    }
    لكل (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->ميٽرڪس_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->ميٽرڪس_A[y & 0x1];
}

بنانشان صحيح randint32(mt19937_state* state) {
    إذا (!state) أرجع 0;
    إذا (state->ميٽرڪس_A[0] != 0 || state->ميٽرڪس_A[1] != 0x9908b0df) manual_ٻج(state, 5489); // auto-initialize
    إذا (--state->کاٻو <= 0) {
        ايندڙ_state(state);
    }
    بنانشان صحيح y = state->state_[state->ايندڙ++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    أرجع y;
}

مضمن بنانشان ڊگهو ڊگهو randint64(mt19937_state* state) {
    أرجع (((بنانشان ڊگهو ڊگهو)(randint32(state)) << 32) | randint32(state));
}

مضمن عائم randfloat32(mt19937_state* state) {
    أرجع (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

مضمن مزدوج randfloat64(mt19937_state* state) {
    أرجع (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

فراغ هڪجهڙو(عائم* معلومات, بنانشان صحيح numel, عائم from, عائم منزل, mt19937_state* state) {
    لكل (بنانشان صحيح t = 0; t < numel; t++) {
        معلومات[t] = randfloat32(state) * (منزل - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
فراغ عام_fill_16(عائم* معلومات, عائم سراسري, عائم std) {
    #وصف_ڪريو EPSILONE 1e-12f
    لكل (بنانشان صحيح t = 0; t < 8; t++) {
        عائم u1 = 1 - معلومات[t];
        عائم u2 = معلومات[t + 8];
        عائم radius = sqrtf(-2 * logf(u1 + EPSILONE));
        عائم theta = (عائم) (2.0 * M_PI * u2);
        معلومات[t] = (radius * cosf(theta) * std + سراسري);
        معلومات[t + 8] = (radius * sinf(theta) * std + سراسري);
    }
}

فراغ عام_fill(عائم* معلومات, بنانشان صحيح numel, عائم سراسري, عائم std, mt19937_state* state) {
    لكل (بنانشان صحيح t = 0; t < numel; t++) {
        معلومات[t] = randfloat32(state);
    }
    لكل (بنانشان صحيح i = 0; i < numel - 15; i += 16) {
        عام_fill_16(معلومات + i, سراسري, std);
    }
    إذا (numel % 16 != 0) {
        // recompute the last 16 values
        معلومات = معلومات + numel - 16;
        لكل (بنانشان صحيح i = 0; i < 16; i++) {
            معلومات[i] = randfloat32(state);
        }
        عام_fill_16(معلومات, سراسري, std);
    }
}

فراغ عام_2(عائم* معلومات, بنانشان صحيح numel, عائم سراسري, عائم std, mt19937_state* state) {
    #وصف_ڪريو EPSILONE 1e-12f
    إذا (numel >= 16) {
        عام_fill(معلومات, numel, سراسري, std, state);
    }
    وإلا {
        مزدوج ايندڙ_double_عام_sample = 0.0; // make compiler warning happy, won't be used
        صحيح has_ايندڙ_double_عام_sample = 0;
        لكل (بنانشان صحيح  t = 0; t < numel; t++) {
            إذا (has_ايندڙ_double_عام_sample) {
                معلومات[t] = (عائم)(ايندڙ_double_عام_sample * std + سراسري);
                has_ايندڙ_double_عام_sample = 0;
                استمر;
            }
            // for numel < 16 we draw a double (float64)
            عائم u1 = (عائم) randfloat64(state);
            عائم u2 = (عائم) randfloat64(state);
            عائم radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            عائم theta = (عائم) (2.0 * M_PI * u1);
            ايندڙ_double_عام_sample = radius * sinf(theta);
            has_ايندڙ_double_عام_sample = 1;
            معلومات[t] = (radius * cosf(theta) * std + سراسري);
        }
    }
}

فراغ init_سڃاڻپ_عنصر_permutation(صحيح *معلومات, صحيح numel) {
    لكل (صحيح i = 0; i < numel; i++) {
        معلومات[i] = i;
    }
}

فراغ random_permutation(صحيح* معلومات, صحيح numel, mt19937_state* state) {
    لكل (صحيح i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        صحيح j = randint32(state) % (i + 1);
        // swap i <-> j
        صحيح tmp = معلومات[i];
        معلومات[i] = معلومات[j];
        معلومات[j] = tmp;
    }
}

#جيڪڏهن_پورو