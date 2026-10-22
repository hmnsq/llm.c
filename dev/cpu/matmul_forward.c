/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <time.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

ਖਾਲੀ matmul_forward_cpu(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ,
                    ਅਚਲ ਦਸ਼ਮਲਵ* inp, ਅਚਲ ਦਸ਼ਮਲਵ* ਭਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ* bias,
                    ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    ਲਈ (ਪੂਰਨ b = 0; b < B; b++) {
        ਲਈ (ਪੂਰਨ t = 0; t < T; t++) {
            ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ_bt = ਬਾਹਰਲਾ + b * T * OC + t * OC;
            ਅਚਲ ਦਸ਼ਮਲਵ* inp_bt = inp + b * T * C + t * C;
            ਲਈ (ਪੂਰਨ o = 0; o < OC; o++) {
                ਦਸ਼ਮਲਵ val = (bias != ਨਿਰਾ) ? bias[o] : 0.0f;
                ਅਚਲ ਦਸ਼ਮਲਵ* wrow = ਭਾਰ + o*C;
                ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                ਬਾਹਰਲਾ_bt[o] = val;
            }
        }
    }
}

ਖਾਲੀ matmul_forward_ngc92(ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ,
    ਅਚਲ ਦਸ਼ਮਲਵ* inp, ਅਚਲ ਦਸ਼ਮਲਵ* ਭਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ* bias,
    ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOOP_UNROLL 8

    ਜੇ (B * T % LOOP_UNROLL != 0) {
        ਲਿਖੋ("MUST BE A MULTIPLE OF 8"); // FIXME
        ਵਾਪਸ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    ਲਈ (ਪੂਰਨ obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        ਲਈ (ਪੂਰਨ o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            ਦਸ਼ਮਲਵ ਨਤੀਜਾ[LOOP_UNROLL];
            ਲਈ (ਪੂਰਨ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ਨਤੀਜਾ[ibt] = (bias != ਨਿਰਾ) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            ਲਈ (ਪੂਰਨ i = 0; i < C; i++) {
                ਦਸ਼ਮਲਵ w = ਭਾਰ[i + o * C];
                ਲਈ (ਪੂਰਨ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    ਪੂਰਨ bt = obt + ibt;
                    ਨਤੀਜਾ[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            ਲਈ (ਪੂਰਨ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ਪੂਰਨ bt = obt + ibt;
                ਬਾਹਰਲਾ[bt * OC + o] = ਨਤੀਜਾ[ibt];
            }
        }
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_KERNELS 2

ਖਾਲੀ matmul_forward(ਪੂਰਨ kernel_ਅੰਕ,
    ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ,
    ਅਚਲ ਦਸ਼ਮਲਵ* inp, ਅਚਲ ਦਸ਼ਮਲਵ* ਭਾਰ, ਅਚਲ ਦਸ਼ਮਲਵ* bias,
    ਪੂਰਨ B, ਪੂਰਨ T, ਪੂਰਨ C, ਪੂਰਨ OC) {

    ਚੋਣ (kernel_ਅੰਕ) {
        ਹਾਲਤ 0:
            matmul_forward_cpu(ਬਾਹਰਲਾ, inp, ਭਾਰ, bias, B, T, C, OC);
            ਰੋਕੋ;
        ਹਾਲਤ 1:
            matmul_forward_ngc92(ਬਾਹਰਲਾ, inp, ਭਾਰ, bias, B, T, C, OC);
            ਰੋਕੋ;
        ਮੂਲ:
            ਲਿਖੋ("Invalid kernel number\n");
            ਬਾਹਰ_ਨਿਕਲੋ(1);
    }
}


ਖਾਲੀ validate_results_cpu(ਅਚਲ ਦਸ਼ਮਲਵ* device_ਨਤੀਜਾ, ਅਚਲ ਦਸ਼ਮਲਵ* cpu_reference, ਅਚਲ ਅੱਖਰ* name, ਪੂਰਨ ਅੰਕ_elements, ਦਸ਼ਮਲਵ tolerance);
ਦਸ਼ਮਲਵ* make_random_float(size_t N);

ਪੂਰਨ ਮੁੱਖ(ਪੂਰਨ argc, ਅੱਖਰ **argv) {
    srand(0);

    ਪੂਰਨ B = 8;
    ਪੂਰਨ T = 1024;
    ਪੂਰਨ C = 768;
    ਪੂਰਨ OC = 768 * 4; // expansion of 4, e.g. in the MLP
    ਪੂਰਨ RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    ਦਸ਼ਮਲਵ* ਬਾਹਰਲਾ = make_random_float(B * T * OC);
    ਦਸ਼ਮਲਵ* inp = make_random_float(B * T * C);
    ਦਸ਼ਮਲਵ* ਭਾਰ = make_random_float(OC * C);
    ਦਸ਼ਮਲਵ* bias = make_random_float(OC);

    ਦਸ਼ਮਲਵ* grad_ਬਾਹਰਲਾ = make_random_float(B * T * OC);
    ਦਸ਼ਮਲਵ* grad_inp = make_random_float(B * T * C);
    ਦਸ਼ਮਲਵ* grad_ਭਾਰ = make_random_float(OC * C);
    ਦਸ਼ਮਲਵ* grad_bias = make_random_float(OC);

    ਲਿਖੋ("> Calculating reference\n");
    matmul_forward_cpu(ਬਾਹਰਲਾ, inp, ਭਾਰ, bias, B, T, C, OC);

    ਲਈ (ਪੂਰਨ kernel_ਅੰਕ = 0; kernel_ਅੰਕ < ਅੰਕ_KERNELS; kernel_ਅੰਕ++) {
        ਲਿਖੋ("> Verifying kernel #%d\n", kernel_ਅੰਕ);

        srand(137);

        ਦਸ਼ਮਲਵ* kernel_ਬਾਹਰਲਾ = make_random_float(B * T * OC);
        ਦਸ਼ਮਲਵ* kernel_inp = make_random_float(B * T * C);
        ਦਸ਼ਮਲਵ* kernel_ਭਾਰ = make_random_float(OC * C);
        ਦਸ਼ਮਲਵ* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_ਅੰਕ, kernel_ਬਾਹਰਲਾ, kernel_inp, kernel_ਭਾਰ, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_ਬਾਹਰਲਾ, ਬਾਹਰਲਾ, "out", B * T * OC, 1e-5);

        ਮੁਕਤ(kernel_ਬਾਹਰਲਾ);
        ਮੁਕਤ(kernel_inp);
        ਮੁਕਤ(kernel_ਭਾਰ);
        ਮੁਕਤ(kernel_bias);
    }

    ਲਿਖੋ("All kernels passed! Starting benchmarks.\n\n");

    ਲਈ (ਪੂਰਨ kernel_ਅੰਕ = 0; kernel_ਅੰਕ < ਅੰਕ_KERNELS; kernel_ਅੰਕ++) {
        ਲਿਖੋ("> Running kernel #%d\n", kernel_ਅੰਕ);
        ਬਣਤਰ timespec ਸ਼ੁਰੂ, ਸਿਰਾ;
        clock_gettime(CLOCK_MONOTONIC, &ਸ਼ੁਰੂ);

        ਲਈ (ਪੂਰਨ i = 0; i < RUNS; i++) {
            matmul_forward(kernel_ਅੰਕ, ਬਾਹਰਲਾ, inp, ਭਾਰ, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &ਸਿਰਾ);
        ਦੁੱਗਣਾ ਸਮਾਂ_elapsed_s = (ਸਿਰਾ.tv_sec - ਸ਼ੁਰੂ.tv_sec) + (ਸਿਰਾ.tv_nsec - ਸ਼ੁਰੂ.tv_nsec) / 1e9;
        ਲਿਖੋ("> Kernel #%d, (took %f ms)\n", kernel_ਅੰਕ, ਸਮਾਂ_elapsed_s * 1000);
    }

    // free memory
    ਮੁਕਤ(ਬਾਹਰਲਾ);
    ਮੁਕਤ(inp);
    ਮੁਕਤ(ਭਾਰ);
    ਮੁਕਤ(bias);

    ਮੁਕਤ(grad_ਬਾਹਰਲਾ);
    ਮੁਕਤ(grad_inp);
    ਮੁਕਤ(grad_ਭਾਰ);
    ਮੁਕਤ(grad_bias);

    ਵਾਪਸ 0;
}

ਦਸ਼ਮਲਵ* make_random_float(size_t N) {
    ਦਸ਼ਮਲਵ* arr = (ਦਸ਼ਮਲਵ*)ਸਮ੍ਰਿਤੀ_ਦਿਓ(N * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਲਈ (size_t i = 0; i < N; i++) {
        arr[i] = ((ਦਸ਼ਮਲਵ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ਵਾਪਸ arr;
}

ਖਾਲੀ validate_results_cpu(ਅਚਲ ਦਸ਼ਮਲਵ* kernel_ਨਤੀਜਾ, ਅਚਲ ਦਸ਼ਮਲਵ* cpu_reference, ਅਚਲ ਅੱਖਰ* name, ਪੂਰਨ ਅੰਕ_elements, ਦਸ਼ਮਲਵ tolerance) {
    ਪੂਰਨ nfaults = 0;
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_elements; i++) {
        // print the first few comparisons
        ਜੇ (i < 5) {
            ਲਿਖੋ("%f %f\n", cpu_reference[i], kernel_ਨਤੀਜਾ[i]);
        }
        ਦਸ਼ਮਲਵ t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        ਜੇ (fabs(cpu_reference[i] - kernel_ਨਤੀਜਾ[i]) > t_eff) {
            ਲਿਖੋ("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_ਨਤੀਜਾ[i]);
            nfaults++;
            ਜੇ (nfaults >= 10) {
                ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
            }
        }
    }
    ਜੇ (nfaults > 0) {
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
    ਲਿਖੋ("OK\n");
}