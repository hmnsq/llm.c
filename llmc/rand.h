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

#నిర్వచితం_కాకపోతే RAND_H
#నిర్వచించు RAND_H

#అన్తర్భావ <math.h>

#నిర్వచించు MERSENNE_STATE_M 397u
#నిర్వచించు MERSENNE_STATE_N 624u

#నిర్వచించు LMASK 0x7ffffffful
#నిర్వచించు UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

రకనిర్వచనం నిర్మాణం {
    గుర్తులేని దీర్ఘ దీర్ఘ విత్తనం;
    పూర్ణాంకం ఎడమ_2;
    గుర్తులేని పూర్ణాంకం తదుపరి;
    గుర్తులేని పూర్ణాంకం state_[MERSENNE_STATE_N];
    గుర్తులేని పూర్ణాంకం మాతృక_A[2];
} mt19937_state;

శూన్య manual_విత్తనం(mt19937_state* state, గుర్తులేని పూర్ణాంకం విత్తనం_2) {
    state->మాతృక_A[0] = 0x0u;
    state->మాతృక_A[1] = 0x9908b0df;
    state->state_[0] = విత్తనం_2 & 0xffffffff;
    కొరకు (గుర్తులేని పూర్ణాంకం j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->ఎడమ_2 = 1;
    state->తదుపరి = 0;
}

శూన్య తదుపరి_state(mt19937_state* state) {
    state->ఎడమ_2 = MERSENNE_STATE_N;
    state->తదుపరి = 0;
    గుర్తులేని పూర్ణాంకం y, j;
    కొరకు (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->మాతృక_A[y & 0x1];
    }
    కొరకు (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->మాతృక_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->మాతృక_A[y & 0x1];
}

గుర్తులేని పూర్ణాంకం randint32(mt19937_state* state) {
    ఒకవేళ (!state) తిరిగి 0;
    ఒకవేళ (state->మాతృక_A[0] != 0 || state->మాతృక_A[1] != 0x9908b0df) manual_విత్తనం(state, 5489); // auto-initialize
    ఒకవేళ (--state->ఎడమ_2 <= 0) {
        తదుపరి_state(state);
    }
    గుర్తులేని పూర్ణాంకం y = state->state_[state->తదుపరి++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    తిరిగి y;
}

అంతఃవరుస గుర్తులేని దీర్ఘ దీర్ఘ randint64(mt19937_state* state) {
    తిరిగి (((గుర్తులేని దీర్ఘ దీర్ఘ)(randint32(state)) << 32) | randint32(state));
}

అంతఃవరుస దశాంశం randfloat32(mt19937_state* state) {
    తిరిగి (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

అంతఃవరుస ద్విగుణ randfloat64(mt19937_state* state) {
    తిరిగి (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

శూన్య ఏకరీతి(దశాంశం* దత్తాంశం, గుర్తులేని పూర్ణాంకం numel, దశాంశం from, దశాంశం గమ్యం, mt19937_state* state) {
    కొరకు (గుర్తులేని పూర్ణాంకం t = 0; t < numel; t++) {
        దత్తాంశం[t] = randfloat32(state) * (గమ్యం - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
శూన్య సాధారణ_fill_16(దశాంశం* దత్తాంశం, దశాంశం సగటు, దశాంశం std) {
    #నిర్వచించు EPSILONE 1e-12f
    కొరకు (గుర్తులేని పూర్ణాంకం t = 0; t < 8; t++) {
        దశాంశం u1 = 1 - దత్తాంశం[t];
        దశాంశం u2 = దత్తాంశం[t + 8];
        దశాంశం radius = sqrtf(-2 * logf(u1 + EPSILONE));
        దశాంశం theta = (దశాంశం) (2.0 * M_PI * u2);
        దత్తాంశం[t] = (radius * cosf(theta) * std + సగటు);
        దత్తాంశం[t + 8] = (radius * sinf(theta) * std + సగటు);
    }
}

శూన్య సాధారణ_fill(దశాంశం* దత్తాంశం, గుర్తులేని పూర్ణాంకం numel, దశాంశం సగటు, దశాంశం std, mt19937_state* state) {
    కొరకు (గుర్తులేని పూర్ణాంకం t = 0; t < numel; t++) {
        దత్తాంశం[t] = randfloat32(state);
    }
    కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < numel - 15; i += 16) {
        సాధారణ_fill_16(దత్తాంశం + i, సగటు, std);
    }
    ఒకవేళ (numel % 16 != 0) {
        // recompute the last 16 values
        దత్తాంశం = దత్తాంశం + numel - 16;
        కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < 16; i++) {
            దత్తాంశం[i] = randfloat32(state);
        }
        సాధారణ_fill_16(దత్తాంశం, సగటు, std);
    }
}

శూన్య సాధారణ(దశాంశం* దత్తాంశం, గుర్తులేని పూర్ణాంకం numel, దశాంశం సగటు, దశాంశం std, mt19937_state* state) {
    #నిర్వచించు EPSILONE 1e-12f
    ఒకవేళ (numel >= 16) {
        సాధారణ_fill(దత్తాంశం, numel, సగటు, std, state);
    }
    లేకపోతే {
        ద్విగుణ తదుపరి_double_సాధారణ_sample = 0.0; // make compiler warning happy, won't be used
        పూర్ణాంకం has_తదుపరి_double_సాధారణ_sample = 0;
        కొరకు (గుర్తులేని పూర్ణాంకం  t = 0; t < numel; t++) {
            ఒకవేళ (has_తదుపరి_double_సాధారణ_sample) {
                దత్తాంశం[t] = (దశాంశం)(తదుపరి_double_సాధారణ_sample * std + సగటు);
                has_తదుపరి_double_సాధారణ_sample = 0;
                కొనసాగు;
            }
            // for numel < 16 we draw a double (float64)
            దశాంశం u1 = (దశాంశం) randfloat64(state);
            దశాంశం u2 = (దశాంశం) randfloat64(state);
            దశాంశం radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            దశాంశం theta = (దశాంశం) (2.0 * M_PI * u1);
            తదుపరి_double_సాధారణ_sample = radius * sinf(theta);
            has_తదుపరి_double_సాధారణ_sample = 1;
            దత్తాంశం[t] = (radius * cosf(theta) * std + సగటు);
        }
    }
}

శూన్య init_తటస్థం_permutation(పూర్ణాంకం *దత్తాంశం, పూర్ణాంకం numel) {
    కొరకు (పూర్ణాంకం i = 0; i < numel; i++) {
        దత్తాంశం[i] = i;
    }
}

శూన్య random_permutation(పూర్ణాంకం* దత్తాంశం, పూర్ణాంకం numel, mt19937_state* state) {
    కొరకు (పూర్ణాంకం i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        పూర్ణాంకం j = randint32(state) % (i + 1);
        // swap i <-> j
        పూర్ణాంకం tmp = దత్తాంశం[i];
        దత్తాంశం[i] = దత్తాంశం[j];
        దత్తాంశం[j] = tmp;
    }
}

#అయితే_ముగింపు