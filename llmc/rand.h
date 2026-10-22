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

#अगर_अघोषित RAND_H
#घोषणा RAND_H

#समावेश <math.h>

#घोषणा MERSENNE_STATE_M 397u
#घोषणा MERSENNE_STATE_N 624u

#घोषणा LMASK 0x7ffffffful
#घोषणा UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

प्रकार काष्ठा {
    अचिन्हित दीर्घ दीर्घ seed_;
    पूर्णांक बायाँ;
    अचिन्हित पूर्णांक अग्र;
    अचिन्हित पूर्णांक state_[MERSENNE_STATE_N];
    अचिन्हित पूर्णांक MATRIX_A[2];
} mt19937_state;

व्योम manual_seed(mt19937_state* state, अचिन्हित पूर्णांक seed) {
    state->MATRIX_A[0] = 0x0u;
    state->MATRIX_A[1] = 0x9908b0df;
    state->state_[0] = seed & 0xffffffff;
    क्रम (अचिन्हित पूर्णांक j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->बायाँ = 1;
    state->अग्र = 0;
}

व्योम अग्र_state(mt19937_state* state) {
    state->बायाँ = MERSENNE_STATE_N;
    state->अग्र = 0;
    अचिन्हित पूर्णांक y, j;
    क्रम (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    क्रम (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
}

अचिन्हित पूर्णांक randint32(mt19937_state* state) {
    अगर (!state) वापस 0;
    अगर (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df) manual_seed(state, 5489); // auto-initialize
    अगर (--state->बायाँ <= 0) {
        अग्र_state(state);
    }
    अचिन्हित पूर्णांक y = state->state_[state->अग्र++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    वापस y;
}

अंतरभूत अचिन्हित दीर्घ दीर्घ randint64(mt19937_state* state) {
    वापस (((अचिन्हित दीर्घ दीर्घ)(randint32(state)) << 32) | randint32(state));
}

अंतरभूत भग्न randfloat32(mt19937_state* state) {
    वापस (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

अंतरभूत द्विगुन randfloat64(mt19937_state* state) {
    वापस (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

व्योम uniform_(भग्न* तथ्य, अचिन्हित पूर्णांक numel, भग्न from, भग्न to, mt19937_state* state) {
    क्रम (अचिन्हित पूर्णांक t = 0; t < numel; t++) {
        तथ्य[t] = randfloat32(state) * (to - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
व्योम normal_fill_16(भग्न* तथ्य, भग्न mean, भग्न std) {
    #घोषणा EPSILONE 1e-12f
    क्रम (अचिन्हित पूर्णांक t = 0; t < 8; t++) {
        भग्न u1 = 1 - तथ्य[t];
        भग्न u2 = तथ्य[t + 8];
        भग्न radius = sqrtf(-2 * logf(u1 + EPSILONE));
        भग्न theta = (भग्न) (2.0 * M_PI * u2);
        तथ्य[t] = (radius * cosf(theta) * std + mean);
        तथ्य[t + 8] = (radius * sinf(theta) * std + mean);
    }
}

व्योम normal_fill(भग्न* तथ्य, अचिन्हित पूर्णांक numel, भग्न mean, भग्न std, mt19937_state* state) {
    क्रम (अचिन्हित पूर्णांक t = 0; t < numel; t++) {
        तथ्य[t] = randfloat32(state);
    }
    क्रम (अचिन्हित पूर्णांक i = 0; i < numel - 15; i += 16) {
        normal_fill_16(तथ्य + i, mean, std);
    }
    अगर (numel % 16 != 0) {
        // recompute the last 16 values
        तथ्य = तथ्य + numel - 16;
        क्रम (अचिन्हित पूर्णांक i = 0; i < 16; i++) {
            तथ्य[i] = randfloat32(state);
        }
        normal_fill_16(तथ्य, mean, std);
    }
}

व्योम normal_(भग्न* तथ्य, अचिन्हित पूर्णांक numel, भग्न mean, भग्न std, mt19937_state* state) {
    #घोषणा EPSILONE 1e-12f
    अगर (numel >= 16) {
        normal_fill(तथ्य, numel, mean, std, state);
    }
    अन्यथा {
        द्विगुन अग्र_double_normal_sample = 0.0; // make compiler warning happy, won't be used
        पूर्णांक has_अग्र_double_normal_sample = 0;
        क्रम (अचिन्हित पूर्णांक  t = 0; t < numel; t++) {
            अगर (has_अग्र_double_normal_sample) {
                तथ्य[t] = (भग्न)(अग्र_double_normal_sample * std + mean);
                has_अग्र_double_normal_sample = 0;
                जारी;
            }
            // for numel < 16 we draw a double (float64)
            भग्न u1 = (भग्न) randfloat64(state);
            भग्न u2 = (भग्न) randfloat64(state);
            भग्न radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            भग्न theta = (भग्न) (2.0 * M_PI * u1);
            अग्र_double_normal_sample = radius * sinf(theta);
            has_अग्र_double_normal_sample = 1;
            तथ्य[t] = (radius * cosf(theta) * std + mean);
        }
    }
}

व्योम init_identity_permutation(पूर्णांक *तथ्य, पूर्णांक numel) {
    क्रम (पूर्णांक i = 0; i < numel; i++) {
        तथ्य[i] = i;
    }
}

व्योम random_permutation(पूर्णांक* तथ्य, पूर्णांक numel, mt19937_state* state) {
    क्रम (पूर्णांक i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        पूर्णांक j = randint32(state) % (i + 1);
        // swap i <-> j
        पूर्णांक tmp = तथ्य[i];
        तथ्य[i] = तथ्य[j];
        तथ्य[j] = tmp;
    }
}

#पूर्ण_अगर