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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ RAND_H
#ವ್ಯಾಖ್ಯಾನಿಸು RAND_H

#ಅನ್ತರ್ಭಾವ <math.h>

#ವ್ಯಾಖ್ಯಾನಿಸು MERSENNE_STATE_M 397u
#ವ್ಯಾಖ್ಯಾನಿಸು MERSENNE_STATE_N 624u

#ವ್ಯಾಖ್ಯಾನಿಸು LMASK 0x7ffffffful
#ವ್ಯಾಖ್ಯಾನಿಸು UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ ಬೀಜ;
    ಪೂರ್ಣಾಂಕ ಎಡ_2;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಮುಂದಿನ;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ state_[MERSENNE_STATE_N];
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಮಾತೃಕೆ_A[2];
} mt19937_state;

ಶೂನ್ಯ manual_ಬೀಜ(mt19937_state* state, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಬೀಜ_2) {
    state->ಮಾತೃಕೆ_A[0] = 0x0u;
    state->ಮಾತೃಕೆ_A[1] = 0x9908b0df;
    state->state_[0] = ಬೀಜ_2 & 0xffffffff;
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->ಎಡ_2 = 1;
    state->ಮುಂದಿನ = 0;
}

ಶೂನ್ಯ ಮುಂದಿನ_state(mt19937_state* state) {
    state->ಎಡ_2 = MERSENNE_STATE_N;
    state->ಮುಂದಿನ = 0;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ y, j;
    ಗಾಗಿ (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->ಮಾತೃಕೆ_A[y & 0x1];
    }
    ಗಾಗಿ (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->ಮಾತೃಕೆ_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->ಮಾತೃಕೆ_A[y & 0x1];
}

ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ randint32(mt19937_state* state) {
    ಆದರೆ (!state) ಹಿಂತಿರುಗಿ 0;
    ಆದರೆ (state->ಮಾತೃಕೆ_A[0] != 0 || state->ಮಾತೃಕೆ_A[1] != 0x9908b0df) manual_ಬೀಜ(state, 5489); // auto-initialize
    ಆದರೆ (--state->ಎಡ_2 <= 0) {
        ಮುಂದಿನ_state(state);
    }
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ y = state->state_[state->ಮುಂದಿನ++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    ಹಿಂತಿರುಗಿ y;
}

ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ randint64(mt19937_state* state) {
    ಹಿಂತಿರುಗಿ (((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ)(randint32(state)) << 32) | randint32(state));
}

ಸಾಲೊಳಗೆ ದಶಮಾಂಶ randfloat32(mt19937_state* state) {
    ಹಿಂತಿರುಗಿ (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ randfloat64(mt19937_state* state) {
    ಹಿಂತಿರುಗಿ (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

ಶೂನ್ಯ ಏಕರೂಪ(ದಶಮಾಂಶ* ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ numel, ದಶಮಾಂಶ from, ದಶಮಾಂಶ ಗಮ್ಯ, mt19937_state* state) {
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ t = 0; t < numel; t++) {
        ದತ್ತಾಂಶ[t] = randfloat32(state) * (ಗಮ್ಯ - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
ಶೂನ್ಯ ಸಾಮಾನ್ಯ_fill_16(ದಶಮಾಂಶ* ದತ್ತಾಂಶ, ದಶಮಾಂಶ ಸರಾಸರಿ, ದಶಮಾಂಶ std) {
    #ವ್ಯಾಖ್ಯಾನಿಸು EPSILONE 1e-12f
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ t = 0; t < 8; t++) {
        ದಶಮಾಂಶ u1 = 1 - ದತ್ತಾಂಶ[t];
        ದಶಮಾಂಶ u2 = ದತ್ತಾಂಶ[t + 8];
        ದಶಮಾಂಶ radius = sqrtf(-2 * logf(u1 + EPSILONE));
        ದಶಮಾಂಶ theta = (ದಶಮಾಂಶ) (2.0 * M_PI * u2);
        ದತ್ತಾಂಶ[t] = (radius * cosf(theta) * std + ಸರಾಸರಿ);
        ದತ್ತಾಂಶ[t + 8] = (radius * sinf(theta) * std + ಸರಾಸರಿ);
    }
}

ಶೂನ್ಯ ಸಾಮಾನ್ಯ_fill(ದಶಮಾಂಶ* ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ numel, ದಶಮಾಂಶ ಸರಾಸರಿ, ದಶಮಾಂಶ std, mt19937_state* state) {
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ t = 0; t < numel; t++) {
        ದತ್ತಾಂಶ[t] = randfloat32(state);
    }
    ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < numel - 15; i += 16) {
        ಸಾಮಾನ್ಯ_fill_16(ದತ್ತಾಂಶ + i, ಸರಾಸರಿ, std);
    }
    ಆದರೆ (numel % 16 != 0) {
        // recompute the last 16 values
        ದತ್ತಾಂಶ = ದತ್ತಾಂಶ + numel - 16;
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
            ದತ್ತಾಂಶ[i] = randfloat32(state);
        }
        ಸಾಮಾನ್ಯ_fill_16(ದತ್ತಾಂಶ, ಸರಾಸರಿ, std);
    }
}

ಶೂನ್ಯ ಸಾಮಾನ್ಯ(ದಶಮಾಂಶ* ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ numel, ದಶಮಾಂಶ ಸರಾಸರಿ, ದಶಮಾಂಶ std, mt19937_state* state) {
    #ವ್ಯಾಖ್ಯಾನಿಸು EPSILONE 1e-12f
    ಆದರೆ (numel >= 16) {
        ಸಾಮಾನ್ಯ_fill(ದತ್ತಾಂಶ, numel, ಸರಾಸರಿ, std, state);
    }
    ಇಲ್ಲದಿದ್ದರೆ {
        ದ್ವಿಗುಣ ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample = 0.0; // make compiler warning happy, won't be used
        ಪೂರ್ಣಾಂಕ has_ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample = 0;
        ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ  t = 0; t < numel; t++) {
            ಆದರೆ (has_ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample) {
                ದತ್ತಾಂಶ[t] = (ದಶಮಾಂಶ)(ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample * std + ಸರಾಸರಿ);
                has_ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample = 0;
                ಮುಂದುವರಿ;
            }
            // for numel < 16 we draw a double (float64)
            ದಶಮಾಂಶ u1 = (ದಶಮಾಂಶ) randfloat64(state);
            ದಶಮಾಂಶ u2 = (ದಶಮಾಂಶ) randfloat64(state);
            ದಶಮಾಂಶ radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            ದಶಮಾಂಶ theta = (ದಶಮಾಂಶ) (2.0 * M_PI * u1);
            ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample = radius * sinf(theta);
            has_ಮುಂದಿನ_double_ಸಾಮಾನ್ಯ_sample = 1;
            ದತ್ತಾಂಶ[t] = (radius * cosf(theta) * std + ಸರಾಸರಿ);
        }
    }
}

ಶೂನ್ಯ init_ತಟಸ್ಥ_permutation(ಪೂರ್ಣಾಂಕ *ದತ್ತಾಂಶ, ಪೂರ್ಣಾಂಕ numel) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < numel; i++) {
        ದತ್ತಾಂಶ[i] = i;
    }
}

ಶೂನ್ಯ random_permutation(ಪೂರ್ಣಾಂಕ* ದತ್ತಾಂಶ, ಪೂರ್ಣಾಂಕ numel, mt19937_state* state) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        ಪೂರ್ಣಾಂಕ j = randint32(state) % (i + 1);
        // swap i <-> j
        ಪೂರ್ಣಾಂಕ tmp = ದತ್ತಾಂಶ[i];
        ದತ್ತಾಂಶ[i] = ದತ್ತಾಂಶ[j];
        ದತ್ತಾಂಶ[j] = tmp;
    }
}

#ಆದರೆ_ಅಂತ್ಯ