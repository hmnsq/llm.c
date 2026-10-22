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

#که_نه_تعریف_شوی RAND_H
#تعریف_کړه RAND_H

#ضمن <math.h>

#تعریف_کړه MERSENNE_STATE_M 397u
#تعریف_کړه MERSENNE_STATE_N 624u

#تعریف_کړه LMASK 0x7ffffffful
#تعریف_کړه UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

تعريفنوع بنية {
    بینښه اوږد اوږد تخم;
    صحيح کیڼ_2;
    بینښه صحيح بل;
    بینښه صحيح state_[MERSENNE_STATE_N];
    بینښه صحيح ماتریکس_A[2];
} mt19937_state;

فراغ manual_تخم(mt19937_state* state, بینښه صحيح تخم_2) {
    state->ماتریکس_A[0] = 0x0u;
    state->ماتریکس_A[1] = 0x9908b0df;
    state->state_[0] = تخم_2 & 0xffffffff;
    لكل (بینښه صحيح j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->کیڼ_2 = 1;
    state->بل = 0;
}

فراغ بل_state(mt19937_state* state) {
    state->کیڼ_2 = MERSENNE_STATE_N;
    state->بل = 0;
    بینښه صحيح y, j;
    لكل (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->ماتریکس_A[y & 0x1];
    }
    لكل (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->ماتریکس_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->ماتریکس_A[y & 0x1];
}

بینښه صحيح randint32(mt19937_state* state) {
    إذا (!state) أرجع 0;
    إذا (state->ماتریکس_A[0] != 0 || state->ماتریکس_A[1] != 0x9908b0df) manual_تخم(state, 5489); // auto-initialize
    إذا (--state->کیڼ_2 <= 0) {
        بل_state(state);
    }
    بینښه صحيح y = state->state_[state->بل++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    أرجع y;
}

مضمن بینښه اوږد اوږد randint64(mt19937_state* state) {
    أرجع (((بینښه اوږد اوږد)(randint32(state)) << 32) | randint32(state));
}

مضمن عائم randfloat32(mt19937_state* state) {
    أرجع (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

مضمن مزدوج randfloat64(mt19937_state* state) {
    أرجع (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

فراغ یوشان(عائم* معلومات, بینښه صحيح numel, عائم from, عائم منزل, mt19937_state* state) {
    لكل (بینښه صحيح t = 0; t < numel; t++) {
        معلومات[t] = randfloat32(state) * (منزل - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
فراغ نورمال_fill_16(عائم* معلومات, عائم منځنۍ, عائم std) {
    #تعریف_کړه EPSILONE 1e-12f
    لكل (بینښه صحيح t = 0; t < 8; t++) {
        عائم u1 = 1 - معلومات[t];
        عائم u2 = معلومات[t + 8];
        عائم radius = sqrtf(-2 * logf(u1 + EPSILONE));
        عائم theta = (عائم) (2.0 * M_PI * u2);
        معلومات[t] = (radius * cosf(theta) * std + منځنۍ);
        معلومات[t + 8] = (radius * sinf(theta) * std + منځنۍ);
    }
}

فراغ نورمال_fill(عائم* معلومات, بینښه صحيح numel, عائم منځنۍ, عائم std, mt19937_state* state) {
    لكل (بینښه صحيح t = 0; t < numel; t++) {
        معلومات[t] = randfloat32(state);
    }
    لكل (بینښه صحيح i = 0; i < numel - 15; i += 16) {
        نورمال_fill_16(معلومات + i, منځنۍ, std);
    }
    إذا (numel % 16 != 0) {
        // recompute the last 16 values
        معلومات = معلومات + numel - 16;
        لكل (بینښه صحيح i = 0; i < 16; i++) {
            معلومات[i] = randfloat32(state);
        }
        نورمال_fill_16(معلومات, منځنۍ, std);
    }
}

فراغ نورمال(عائم* معلومات, بینښه صحيح numel, عائم منځنۍ, عائم std, mt19937_state* state) {
    #تعریف_کړه EPSILONE 1e-12f
    إذا (numel >= 16) {
        نورمال_fill(معلومات, numel, منځنۍ, std, state);
    }
    وإلا {
        مزدوج بل_double_نورمال_sample = 0.0; // make compiler warning happy, won't be used
        صحيح has_بل_double_نورمال_sample = 0;
        لكل (بینښه صحيح  t = 0; t < numel; t++) {
            إذا (has_بل_double_نورمال_sample) {
                معلومات[t] = (عائم)(بل_double_نورمال_sample * std + منځنۍ);
                has_بل_double_نورمال_sample = 0;
                استمر;
            }
            // for numel < 16 we draw a double (float64)
            عائم u1 = (عائم) randfloat64(state);
            عائم u2 = (عائم) randfloat64(state);
            عائم radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            عائم theta = (عائم) (2.0 * M_PI * u1);
            بل_double_نورمال_sample = radius * sinf(theta);
            has_بل_double_نورمال_sample = 1;
            معلومات[t] = (radius * cosf(theta) * std + منځنۍ);
        }
    }
}

فراغ init_بې_اغېزه_عنصر_permutation(صحيح *معلومات, صحيح numel) {
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

#د_که_پای