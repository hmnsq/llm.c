/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <time.h>
#ಅನ್ತರ್ಭಾವ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

ಶೂನ್ಯ matmul_forward_cpu(ದಶಮಾಂಶ* ಹೊರಹೋಗು,
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ತೂಕ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* bias,
                    ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < B; b++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < T; t++) {
            ದಶಮಾಂಶ* ಹೊರಹೋಗು_bt = ಹೊರಹೋಗು + b * T * OC + t * OC;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp_bt = inp + b * T * C + t * C;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < OC; o++) {
                ದಶಮಾಂಶ val = (bias != ಖಾಲಿ) ? bias[o] : 0.0f;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* wrow = ತೂಕ + o*C;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                ಹೊರಹೋಗು_bt[o] = val;
            }
        }
    }
}

ಶೂನ್ಯ matmul_forward_ngc92(ದಶಮಾಂಶ* ಹೊರಹೋಗು,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ತೂಕ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* bias,
    ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #ವ್ಯಾಖ್ಯಾನಿಸು LOOP_UNROLL 8

    ಆದರೆ (B * T % LOOP_UNROLL != 0) {
        ಬರೆ("MUST BE A MULTIPLE OF 8"); // FIXME
        ಹಿಂತಿರುಗಿ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            ದಶಮಾಂಶ ಫಲಿತಾಂಶ[LOOP_UNROLL];
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ಫಲಿತಾಂಶ[ibt] = (bias != ಖಾಲಿ) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < C; i++) {
                ದಶಮಾಂಶ w = ತೂಕ[i + o * C];
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    ಪೂರ್ಣಾಂಕ bt = obt + ibt;
                    ಫಲಿತಾಂಶ[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ಪೂರ್ಣಾಂಕ bt = obt + ibt;
                ಹೊರಹೋಗು[bt * OC + o] = ಫಲಿತಾಂಶ[ibt];
            }
        }
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_KERNELS 2

ಶೂನ್ಯ matmul_forward(ಪೂರ್ಣಾಂಕ kernel_ಅಂಕೆ,
    ದಶಮಾಂಶ* ಹೊರಹೋಗು,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* inp, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* ತೂಕ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* bias,
    ಪೂರ್ಣಾಂಕ B, ಪೂರ್ಣಾಂಕ T, ಪೂರ್ಣಾಂಕ C, ಪೂರ್ಣಾಂಕ OC) {

    ಆಯ್ಕೆ (kernel_ಅಂಕೆ) {
        ಸ್ಥಿತಿ 0:
            matmul_forward_cpu(ಹೊರಹೋಗು, inp, ತೂಕ, bias, B, T, C, OC);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ 1:
            matmul_forward_ngc92(ಹೊರಹೋಗು, inp, ತೂಕ, bias, B, T, C, OC);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಬರೆ("Invalid kernel number\n");
            ನಿರ್ಗಮಿಸು(1);
    }
}


ಶೂನ್ಯ validate_results_cpu(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* device_ಫಲಿತಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* cpu_reference, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* name, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elements, ದಶಮಾಂಶ tolerance);
ದಶಮಾಂಶ* make_random_float(size_t N);

ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ(ಪೂರ್ಣಾಂಕ argc, ಅಕ್ಷರ **argv) {
    srand(0);

    ಪೂರ್ಣಾಂಕ B = 8;
    ಪೂರ್ಣಾಂಕ T = 1024;
    ಪೂರ್ಣಾಂಕ C = 768;
    ಪೂರ್ಣಾಂಕ OC = 768 * 4; // expansion of 4, e.g. in the MLP
    ಪೂರ್ಣಾಂಕ RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    ದಶಮಾಂಶ* ಹೊರಹೋಗು = make_random_float(B * T * OC);
    ದಶಮಾಂಶ* inp = make_random_float(B * T * C);
    ದಶಮಾಂಶ* ತೂಕ = make_random_float(OC * C);
    ದಶಮಾಂಶ* bias = make_random_float(OC);

    ದಶಮಾಂಶ* grad_ಹೊರಹೋಗು = make_random_float(B * T * OC);
    ದಶಮಾಂಶ* grad_inp = make_random_float(B * T * C);
    ದಶಮಾಂಶ* grad_ತೂಕ = make_random_float(OC * C);
    ದಶಮಾಂಶ* grad_bias = make_random_float(OC);

    ಬರೆ("> Calculating reference\n");
    matmul_forward_cpu(ಹೊರಹೋಗು, inp, ತೂಕ, bias, B, T, C, OC);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kernel_ಅಂಕೆ = 0; kernel_ಅಂಕೆ < ಅಂಕೆ_KERNELS; kernel_ಅಂಕೆ++) {
        ಬರೆ("> Verifying kernel #%d\n", kernel_ಅಂಕೆ);

        srand(137);

        ದಶಮಾಂಶ* kernel_ಹೊರಹೋಗು = make_random_float(B * T * OC);
        ದಶಮಾಂಶ* kernel_inp = make_random_float(B * T * C);
        ದಶಮಾಂಶ* kernel_ತೂಕ = make_random_float(OC * C);
        ದಶಮಾಂಶ* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_ಅಂಕೆ, kernel_ಹೊರಹೋಗು, kernel_inp, kernel_ತೂಕ, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_ಹೊರಹೋಗು, ಹೊರಹೋಗು, "out", B * T * OC, 1e-5);

        ಬಿಡುಗಡೆ(kernel_ಹೊರಹೋಗು);
        ಬಿಡುಗಡೆ(kernel_inp);
        ಬಿಡುಗಡೆ(kernel_ತೂಕ);
        ಬಿಡುಗಡೆ(kernel_bias);
    }

    ಬರೆ("All kernels passed! Starting benchmarks.\n\n");

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kernel_ಅಂಕೆ = 0; kernel_ಅಂಕೆ < ಅಂಕೆ_KERNELS; kernel_ಅಂಕೆ++) {
        ಬರೆ("> Running kernel #%d\n", kernel_ಅಂಕೆ);
        ರಚನೆ timespec ಆರಂಭ, ಕೊನೆ;
        clock_gettime(CLOCK_MONOTONIC, &ಆರಂಭ);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < RUNS; i++) {
            matmul_forward(kernel_ಅಂಕೆ, ಹೊರಹೋಗು, inp, ತೂಕ, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &ಕೊನೆ);
        ದ್ವಿಗುಣ ಸಮಯ_elapsed_s = (ಕೊನೆ.tv_sec - ಆರಂಭ.tv_sec) + (ಕೊನೆ.tv_nsec - ಆರಂಭ.tv_nsec) / 1e9;
        ಬರೆ("> Kernel #%d, (took %f ms)\n", kernel_ಅಂಕೆ, ಸಮಯ_elapsed_s * 1000);
    }

    // free memory
    ಬಿಡುಗಡೆ(ಹೊರಹೋಗು);
    ಬಿಡುಗಡೆ(inp);
    ಬಿಡುಗಡೆ(ತೂಕ);
    ಬಿಡುಗಡೆ(bias);

    ಬಿಡುಗಡೆ(grad_ಹೊರಹೋಗು);
    ಬಿಡುಗಡೆ(grad_inp);
    ಬಿಡುಗಡೆ(grad_ತೂಕ);
    ಬಿಡುಗಡೆ(grad_bias);

    ಹಿಂತಿರುಗಿ 0;
}

ದಶಮಾಂಶ* make_random_float(size_t N) {
    ದಶಮಾಂಶ* arr = (ದಶಮಾಂಶ*)ಸ್ಮರಣೆ_ಕೊಡು(N * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಗಾಗಿ (size_t i = 0; i < N; i++) {
        arr[i] = ((ದಶಮಾಂಶ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ಹಿಂತಿರುಗಿ arr;
}

ಶೂನ್ಯ validate_results_cpu(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* kernel_ಫಲಿತಾಂಶ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* cpu_reference, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* name, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elements, ದಶಮಾಂಶ tolerance) {
    ಪೂರ್ಣಾಂಕ nfaults = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_elements; i++) {
        // print the first few comparisons
        ಆದರೆ (i < 5) {
            ಬರೆ("%f %f\n", cpu_reference[i], kernel_ಫಲಿತಾಂಶ[i]);
        }
        ದಶಮಾಂಶ t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        ಆದರೆ (fabs(cpu_reference[i] - kernel_ಫಲಿತಾಂಶ[i]) > t_eff) {
            ಬರೆ("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_ಫಲಿತಾಂಶ[i]);
            nfaults++;
            ಆದರೆ (nfaults >= 10) {
                ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
            }
        }
    }
    ಆದರೆ (nfaults > 0) {
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
    ಬರೆ("OK\n");
}