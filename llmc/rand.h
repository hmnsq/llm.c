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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ RAND_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RAND_H

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MERSENNE_STATE_M 397u
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MERSENNE_STATE_N 624u

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LMASK 0x7ffffffful
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ ਬੀਜ;
    ਪੂਰਨ ਖੱਬਾ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਅਗਲਾ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ state_[MERSENNE_STATE_N];
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਮੈਟ੍ਰਿਕਸ_A[2];
} mt19937_state;

ਖਾਲੀ manual_ਬੀਜ(mt19937_state* state, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਬੀਜ_2) {
    state->ਮੈਟ੍ਰਿਕਸ_A[0] = 0x0u;
    state->ਮੈਟ੍ਰਿਕਸ_A[1] = 0x9908b0df;
    state->state_[0] = ਬੀਜ_2 & 0xffffffff;
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->ਖੱਬਾ = 1;
    state->ਅਗਲਾ = 0;
}

ਖਾਲੀ ਅਗਲਾ_state(mt19937_state* state) {
    state->ਖੱਬਾ = MERSENNE_STATE_N;
    state->ਅਗਲਾ = 0;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ y, j;
    ਲਈ (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->ਮੈਟ੍ਰਿਕਸ_A[y & 0x1];
    }
    ਲਈ (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->ਮੈਟ੍ਰਿਕਸ_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->ਮੈਟ੍ਰਿਕਸ_A[y & 0x1];
}

ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ randint32(mt19937_state* state) {
    ਜੇ (!state) ਵਾਪਸ 0;
    ਜੇ (state->ਮੈਟ੍ਰਿਕਸ_A[0] != 0 || state->ਮੈਟ੍ਰਿਕਸ_A[1] != 0x9908b0df) manual_ਬੀਜ(state, 5489); // auto-initialize
    ਜੇ (--state->ਖੱਬਾ <= 0) {
        ਅਗਲਾ_state(state);
    }
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ y = state->state_[state->ਅਗਲਾ++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    ਵਾਪਸ y;
}

ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ randint64(mt19937_state* state) {
    ਵਾਪਸ (((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ)(randint32(state)) << 32) | randint32(state));
}

ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ randfloat32(mt19937_state* state) {
    ਵਾਪਸ (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ randfloat64(mt19937_state* state) {
    ਵਾਪਸ (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

ਖਾਲੀ ਇਕਸਾਰ(ਦਸ਼ਮਲਵ* ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ numel, ਦਸ਼ਮਲਵ from, ਦਸ਼ਮਲਵ ਮੰਜ਼ਿਲ, mt19937_state* state) {
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ t = 0; t < numel; t++) {
        ਅੰਕੜੇ[t] = randfloat32(state) * (ਮੰਜ਼ਿਲ - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
ਖਾਲੀ ਸਾਧਾਰਨ_fill_16(ਦਸ਼ਮਲਵ* ਅੰਕੜੇ, ਦਸ਼ਮਲਵ ਔਸਤ, ਦਸ਼ਮਲਵ std) {
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ EPSILONE 1e-12f
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ t = 0; t < 8; t++) {
        ਦਸ਼ਮਲਵ u1 = 1 - ਅੰਕੜੇ[t];
        ਦਸ਼ਮਲਵ u2 = ਅੰਕੜੇ[t + 8];
        ਦਸ਼ਮਲਵ radius = sqrtf(-2 * logf(u1 + EPSILONE));
        ਦਸ਼ਮਲਵ theta = (ਦਸ਼ਮਲਵ) (2.0 * M_PI * u2);
        ਅੰਕੜੇ[t] = (radius * cosf(theta) * std + ਔਸਤ);
        ਅੰਕੜੇ[t + 8] = (radius * sinf(theta) * std + ਔਸਤ);
    }
}

ਖਾਲੀ ਸਾਧਾਰਨ_fill(ਦਸ਼ਮਲਵ* ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ numel, ਦਸ਼ਮਲਵ ਔਸਤ, ਦਸ਼ਮਲਵ std, mt19937_state* state) {
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ t = 0; t < numel; t++) {
        ਅੰਕੜੇ[t] = randfloat32(state);
    }
    ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < numel - 15; i += 16) {
        ਸਾਧਾਰਨ_fill_16(ਅੰਕੜੇ + i, ਔਸਤ, std);
    }
    ਜੇ (numel % 16 != 0) {
        // recompute the last 16 values
        ਅੰਕੜੇ = ਅੰਕੜੇ + numel - 16;
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < 16; i++) {
            ਅੰਕੜੇ[i] = randfloat32(state);
        }
        ਸਾਧਾਰਨ_fill_16(ਅੰਕੜੇ, ਔਸਤ, std);
    }
}

ਖਾਲੀ ਸਾਧਾਰਨ(ਦਸ਼ਮਲਵ* ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ numel, ਦਸ਼ਮਲਵ ਔਸਤ, ਦਸ਼ਮਲਵ std, mt19937_state* state) {
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ EPSILONE 1e-12f
    ਜੇ (numel >= 16) {
        ਸਾਧਾਰਨ_fill(ਅੰਕੜੇ, numel, ਔਸਤ, std, state);
    }
    ਨਹੀਂਤਾਂ {
        ਦੁੱਗਣਾ ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample = 0.0; // make compiler warning happy, won't be used
        ਪੂਰਨ has_ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample = 0;
        ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ  t = 0; t < numel; t++) {
            ਜੇ (has_ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample) {
                ਅੰਕੜੇ[t] = (ਦਸ਼ਮਲਵ)(ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample * std + ਔਸਤ);
                has_ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample = 0;
                ਜਾਰੀ;
            }
            // for numel < 16 we draw a double (float64)
            ਦਸ਼ਮਲਵ u1 = (ਦਸ਼ਮਲਵ) randfloat64(state);
            ਦਸ਼ਮਲਵ u2 = (ਦਸ਼ਮਲਵ) randfloat64(state);
            ਦਸ਼ਮਲਵ radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            ਦਸ਼ਮਲਵ theta = (ਦਸ਼ਮਲਵ) (2.0 * M_PI * u1);
            ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample = radius * sinf(theta);
            has_ਅਗਲਾ_double_ਸਾਧਾਰਨ_sample = 1;
            ਅੰਕੜੇ[t] = (radius * cosf(theta) * std + ਔਸਤ);
        }
    }
}

ਖਾਲੀ init_ਪਛਾਣ_ਤੱਤ_permutation(ਪੂਰਨ *ਅੰਕੜੇ, ਪੂਰਨ numel) {
    ਲਈ (ਪੂਰਨ i = 0; i < numel; i++) {
        ਅੰਕੜੇ[i] = i;
    }
}

ਖਾਲੀ random_permutation(ਪੂਰਨ* ਅੰਕੜੇ, ਪੂਰਨ numel, mt19937_state* state) {
    ਲਈ (ਪੂਰਨ i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        ਪੂਰਨ j = randint32(state) % (i + 1);
        // swap i <-> j
        ਪੂਰਨ tmp = ਅੰਕੜੇ[i];
        ਅੰਕੜੇ[i] = ਅੰਕੜੇ[j];
        ਅੰਕੜੇ[j] = tmp;
    }
}

#ਜੇ_ਅੰਤ