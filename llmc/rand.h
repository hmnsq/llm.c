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

#اگر_غیر_تعریف_شدہ RAND_H
#تعریف_کرو RAND_H

#شامل <math.h>

#تعریف_کرو MERSENNE_STATE_M 397u
#تعریف_کرو MERSENNE_STATE_N 624u

#تعریف_کرو LMASK 0x7ffffffful
#تعریف_کرو UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

قسمتعریف ساخت {
    غیرعلامتی طویل طویل بیج;
    صحیح بایاں;
    غیرعلامتی صحیح اگلا;
    غیرعلامتی صحیح state_[MERSENNE_STATE_N];
    غیرعلامتی صحیح میٹرکس_A[2];
} mt19937_state;

خالی manual_بیج(mt19937_state* state, غیرعلامتی صحیح بیج_2) {
    state->میٹرکس_A[0] = 0x0u;
    state->میٹرکس_A[1] = 0x9908b0df;
    state->state_[0] = بیج_2 & 0xffffffff;
    برائے (غیرعلامتی صحیح j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->بایاں = 1;
    state->اگلا = 0;
}

خالی اگلا_state(mt19937_state* state) {
    state->بایاں = MERSENNE_STATE_N;
    state->اگلا = 0;
    غیرعلامتی صحیح y, j;
    برائے (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->میٹرکس_A[y & 0x1];
    }
    برائے (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->میٹرکس_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->میٹرکس_A[y & 0x1];
}

غیرعلامتی صحیح randint32(mt19937_state* state) {
    اگر (!state) واپس 0;
    اگر (state->میٹرکس_A[0] != 0 || state->میٹرکس_A[1] != 0x9908b0df) manual_بیج(state, 5489); // auto-initialize
    اگر (--state->بایاں <= 0) {
        اگلا_state(state);
    }
    غیرعلامتی صحیح y = state->state_[state->اگلا++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    واپس y;
}

اندرونی غیرعلامتی طویل طویل randint64(mt19937_state* state) {
    واپس (((غیرعلامتی طویل طویل)(randint32(state)) << 32) | randint32(state));
}

اندرونی اعشاری randfloat32(mt19937_state* state) {
    واپس (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

اندرونی دوہرا randfloat64(mt19937_state* state) {
    واپس (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

خالی یکساں(اعشاری* ڈیٹا, غیرعلامتی صحیح numel, اعشاری from, اعشاری مقصد, mt19937_state* state) {
    برائے (غیرعلامتی صحیح t = 0; t < numel; t++) {
        ڈیٹا[t] = randfloat32(state) * (مقصد - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
خالی معمول_fill_16(اعشاری* ڈیٹا, اعشاری اوسط, اعشاری std) {
    #تعریف_کرو EPSILONE 1e-12f
    برائے (غیرعلامتی صحیح t = 0; t < 8; t++) {
        اعشاری u1 = 1 - ڈیٹا[t];
        اعشاری u2 = ڈیٹا[t + 8];
        اعشاری radius = sqrtf(-2 * logf(u1 + EPSILONE));
        اعشاری theta = (اعشاری) (2.0 * M_PI * u2);
        ڈیٹا[t] = (radius * cosf(theta) * std + اوسط);
        ڈیٹا[t + 8] = (radius * sinf(theta) * std + اوسط);
    }
}

خالی معمول_fill(اعشاری* ڈیٹا, غیرعلامتی صحیح numel, اعشاری اوسط, اعشاری std, mt19937_state* state) {
    برائے (غیرعلامتی صحیح t = 0; t < numel; t++) {
        ڈیٹا[t] = randfloat32(state);
    }
    برائے (غیرعلامتی صحیح i = 0; i < numel - 15; i += 16) {
        معمول_fill_16(ڈیٹا + i, اوسط, std);
    }
    اگر (numel % 16 != 0) {
        // recompute the last 16 values
        ڈیٹا = ڈیٹا + numel - 16;
        برائے (غیرعلامتی صحیح i = 0; i < 16; i++) {
            ڈیٹا[i] = randfloat32(state);
        }
        معمول_fill_16(ڈیٹا, اوسط, std);
    }
}

خالی معمول(اعشاری* ڈیٹا, غیرعلامتی صحیح numel, اعشاری اوسط, اعشاری std, mt19937_state* state) {
    #تعریف_کرو EPSILONE 1e-12f
    اگر (numel >= 16) {
        معمول_fill(ڈیٹا, numel, اوسط, std, state);
    }
    ورنہ {
        دوہرا اگلا_double_معمول_sample = 0.0; // make compiler warning happy, won't be used
        صحیح has_اگلا_double_معمول_sample = 0;
        برائے (غیرعلامتی صحیح  t = 0; t < numel; t++) {
            اگر (has_اگلا_double_معمول_sample) {
                ڈیٹا[t] = (اعشاری)(اگلا_double_معمول_sample * std + اوسط);
                has_اگلا_double_معمول_sample = 0;
                جاری;
            }
            // for numel < 16 we draw a double (float64)
            اعشاری u1 = (اعشاری) randfloat64(state);
            اعشاری u2 = (اعشاری) randfloat64(state);
            اعشاری radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            اعشاری theta = (اعشاری) (2.0 * M_PI * u1);
            اگلا_double_معمول_sample = radius * sinf(theta);
            has_اگلا_double_معمول_sample = 1;
            ڈیٹا[t] = (radius * cosf(theta) * std + اوسط);
        }
    }
}

خالی init_شناختی_عنصر_permutation(صحیح *ڈیٹا, صحیح numel) {
    برائے (صحیح i = 0; i < numel; i++) {
        ڈیٹا[i] = i;
    }
}

خالی random_permutation(صحیح* ڈیٹا, صحیح numel, mt19937_state* state) {
    برائے (صحیح i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        صحیح j = randint32(state) % (i + 1);
        // swap i <-> j
        صحیح tmp = ڈیٹا[i];
        ڈیٹا[i] = ڈیٹا[j];
        ڈیٹا[j] = tmp;
    }
}

#اگر_ختم