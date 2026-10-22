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

ಶೂನ್ಯ matmul_forward_cpu(ಭಿನ್ನ* out,
                    ಧ್ರುವ ಭಿನ್ನ* inp, ಧ್ರುವ ಭಿನ್ನ* weight, ಧ್ರುವ ಭಿನ್ನ* bias,
                    ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ C, ಪೂರ್ಣಾಙ್ಕ OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ b = 0; b < B; b++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < T; t++) {
            ಭಿನ್ನ* out_bt = out + b * T * OC + t * OC;
            ಧ್ರುವ ಭಿನ್ನ* inp_bt = inp + b * T * C + t * C;
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ o = 0; o < OC; o++) {
                ಭಿನ್ನ val = (bias != ರಿಕ್ತ) ? bias[o] : 0.0f;
                ಧ್ರುವ ಭಿನ್ನ* wrow = weight + o*C;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

ಶೂನ್ಯ matmul_forward_ngc92(ಭಿನ್ನ* out,
    ಧ್ರುವ ಭಿನ್ನ* inp, ಧ್ರುವ ಭಿನ್ನ* weight, ಧ್ರುವ ಭಿನ್ನ* bias,
    ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ C, ಪೂರ್ಣಾಙ್ಕ OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    ಯದಿ (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        ಪ್ರತ್ಯಾಗಮ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            ಭಿನ್ನ result[LOOP_UNROLL];
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != ರಿಕ್ತ) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < C; i++) {
                ಭಿನ್ನ w = weight[i + o * C];
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    ಪೂರ್ಣಾಙ್ಕ bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ಪೂರ್ಣಾಙ್ಕ bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define ಅಂಕೆ_KERNELS 2

ಶೂನ್ಯ matmul_forward(ಪೂರ್ಣಾಙ್ಕ kernel_ಅಂಕೆ,
    ಭಿನ್ನ* out,
    ಧ್ರುವ ಭಿನ್ನ* inp, ಧ್ರುವ ಭಿನ್ನ* weight, ಧ್ರುವ ಭಿನ್ನ* bias,
    ಪೂರ್ಣಾಙ್ಕ B, ಪೂರ್ಣಾಙ್ಕ T, ಪೂರ್ಣಾಙ್ಕ C, ಪೂರ್ಣಾಙ್ಕ OC) {

    ವಿಕಲ್ಪ (kernel_ಅಂಕೆ) {
        ಪಕ್ಷ 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            ಭಙ್ಗ;
        ಪಕ್ಷ 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


ಶೂನ್ಯ validate_results_cpu(ಧ್ರುವ ಭಿನ್ನ* device_result, ಧ್ರುವ ಭಿನ್ನ* cpu_reference, ಧ್ರುವ ವರ್ಣ* name, ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elements, ಭಿನ್ನ tolerance);
ಭಿನ್ನ* make_random_float(size_t N);

ಪೂರ್ಣಾಙ್ಕ main(ಪೂರ್ಣಾಙ್ಕ argc, ವರ್ಣ **argv) {
    srand(0);

    ಪೂರ್ಣಾಙ್ಕ B = 8;
    ಪೂರ್ಣಾಙ್ಕ T = 1024;
    ಪೂರ್ಣಾಙ್ಕ C = 768;
    ಪೂರ್ಣಾಙ್ಕ OC = 768 * 4; // expansion of 4, e.g. in the MLP
    ಪೂರ್ಣಾಙ್ಕ RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    ಭಿನ್ನ* out = make_random_float(B * T * OC);
    ಭಿನ್ನ* inp = make_random_float(B * T * C);
    ಭಿನ್ನ* weight = make_random_float(OC * C);
    ಭಿನ್ನ* bias = make_random_float(OC);

    ಭಿನ್ನ* grad_out = make_random_float(B * T * OC);
    ಭಿನ್ನ* grad_inp = make_random_float(B * T * C);
    ಭಿನ್ನ* grad_weight = make_random_float(OC * C);
    ಭಿನ್ನ* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ kernel_ಅಂಕೆ = 0; kernel_ಅಂಕೆ < ಅಂಕೆ_KERNELS; kernel_ಅಂಕೆ++) {
        printf("> Verifying kernel #%d\n", kernel_ಅಂಕೆ);

        srand(137);

        ಭಿನ್ನ* kernel_out = make_random_float(B * T * OC);
        ಭಿನ್ನ* kernel_inp = make_random_float(B * T * C);
        ಭಿನ್ನ* kernel_weight = make_random_float(OC * C);
        ಭಿನ್ನ* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_ಅಂಕೆ, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        ಮುಕ್ತ(kernel_out);
        ಮುಕ್ತ(kernel_inp);
        ಮುಕ್ತ(kernel_weight);
        ಮುಕ್ತ(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ kernel_ಅಂಕೆ = 0; kernel_ಅಂಕೆ < ಅಂಕೆ_KERNELS; kernel_ಅಂಕೆ++) {
        printf("> Running kernel #%d\n", kernel_ಅಂಕೆ);
        ಸಂರಚನಾ timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < RUNS; i++) {
            matmul_forward(kernel_ಅಂಕೆ, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        ದ್ವಿಗುಣ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_ಅಂಕೆ, time_elapsed_s * 1000);
    }

    // free memory
    ಮುಕ್ತ(out);
    ಮುಕ್ತ(inp);
    ಮುಕ್ತ(weight);
    ಮುಕ್ತ(bias);

    ಮುಕ್ತ(grad_out);
    ಮುಕ್ತ(grad_inp);
    ಮುಕ್ತ(grad_weight);
    ಮುಕ್ತ(grad_bias);

    ಪ್ರತ್ಯಾಗಮ 0;
}

ಭಿನ್ನ* make_random_float(size_t N) {
    ಭಿನ್ನ* arr = (ಭಿನ್ನ*)malloc(N * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತಿ (size_t i = 0; i < N; i++) {
        arr[i] = ((ಭಿನ್ನ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ಪ್ರತ್ಯಾಗಮ arr;
}

ಶೂನ್ಯ validate_results_cpu(ಧ್ರುವ ಭಿನ್ನ* kernel_result, ಧ್ರುವ ಭಿನ್ನ* cpu_reference, ಧ್ರುವ ವರ್ಣ* name, ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_elements, ಭಿನ್ನ tolerance) {
    ಪೂರ್ಣಾಙ್ಕ nfaults = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_elements; i++) {
        // print the first few comparisons
        ಯದಿ (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        ಭಿನ್ನ t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        ಯದಿ (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            ಯದಿ (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    ಯದಿ (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}