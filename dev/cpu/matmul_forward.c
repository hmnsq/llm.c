/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <time.h>
#അന്തര്ഭാവ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

ശൂന്യം matmul_forward_cpu(ദശാംശം* out,
                    സ്ഥിരം ദശാംശം* inp, സ്ഥിരം ദശാംശം* weight, സ്ഥിരം ദശാംശം* bias,
                    പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ C, പൂർണ്ണസംഖ്യ OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    വേണ്ടി (പൂർണ്ണസംഖ്യ b = 0; b < B; b++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < T; t++) {
            ദശാംശം* out_bt = out + b * T * OC + t * OC;
            സ്ഥിരം ദശാംശം* inp_bt = inp + b * T * C + t * C;
            വേണ്ടി (പൂർണ്ണസംഖ്യ o = 0; o < OC; o++) {
                ദശാംശം val = (bias != ശൂന്യ) ? bias[o] : 0.0f;
                സ്ഥിരം ദശാംശം* wrow = weight + o*C;
                വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

ശൂന്യം matmul_forward_ngc92(ദശാംശം* out,
    സ്ഥിരം ദശാംശം* inp, സ്ഥിരം ദശാംശം* weight, സ്ഥിരം ദശാംശം* bias,
    പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ C, പൂർണ്ണസംഖ്യ OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    എങ്കിൽ (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        തിരികെ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    വേണ്ടി (പൂർണ്ണസംഖ്യ obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            ദശാംശം result[LOOP_UNROLL];
            വേണ്ടി (പൂർണ്ണസംഖ്യ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != ശൂന്യ) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < C; i++) {
                ദശാംശം w = weight[i + o * C];
                വേണ്ടി (പൂർണ്ണസംഖ്യ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    പൂർണ്ണസംഖ്യ bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            വേണ്ടി (പൂർണ്ണസംഖ്യ ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                പൂർണ്ണസംഖ്യ bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define അക്കം_KERNELS 2

ശൂന്യം matmul_forward(പൂർണ്ണസംഖ്യ kernel_അക്കം,
    ദശാംശം* out,
    സ്ഥിരം ദശാംശം* inp, സ്ഥിരം ദശാംശം* weight, സ്ഥിരം ദശാംശം* bias,
    പൂർണ്ണസംഖ്യ B, പൂർണ്ണസംഖ്യ T, പൂർണ്ണസംഖ്യ C, പൂർണ്ണസംഖ്യ OC) {

    തിരഞ്ഞെടുപ്പ് (kernel_അക്കം) {
        അവസ്ഥ 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            നിർത്തുക;
        അവസ്ഥ 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


ശൂന്യം validate_results_cpu(സ്ഥിരം ദശാംശം* device_result, സ്ഥിരം ദശാംശം* cpu_reference, സ്ഥിരം അക്ഷരം* name, പൂർണ്ണസംഖ്യ അക്കം_elements, ദശാംശം tolerance);
ദശാംശം* make_random_float(size_t N);

പൂർണ്ണസംഖ്യ പ്രധാന(പൂർണ്ണസംഖ്യ argc, അക്ഷരം **argv) {
    srand(0);

    പൂർണ്ണസംഖ്യ B = 8;
    പൂർണ്ണസംഖ്യ T = 1024;
    പൂർണ്ണസംഖ്യ C = 768;
    പൂർണ്ണസംഖ്യ OC = 768 * 4; // expansion of 4, e.g. in the MLP
    പൂർണ്ണസംഖ്യ RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    ദശാംശം* out = make_random_float(B * T * OC);
    ദശാംശം* inp = make_random_float(B * T * C);
    ദശാംശം* weight = make_random_float(OC * C);
    ദശാംശം* bias = make_random_float(OC);

    ദശാംശം* grad_out = make_random_float(B * T * OC);
    ദശാംശം* grad_inp = make_random_float(B * T * C);
    ദശാംശം* grad_weight = make_random_float(OC * C);
    ദശാംശം* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    വേണ്ടി (പൂർണ്ണസംഖ്യ kernel_അക്കം = 0; kernel_അക്കം < അക്കം_KERNELS; kernel_അക്കം++) {
        printf("> Verifying kernel #%d\n", kernel_അക്കം);

        srand(137);

        ദശാംശം* kernel_out = make_random_float(B * T * OC);
        ദശാംശം* kernel_inp = make_random_float(B * T * C);
        ദശാംശം* kernel_weight = make_random_float(OC * C);
        ദശാംശം* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_അക്കം, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        സ്വതന്ത്രം(kernel_out);
        സ്വതന്ത്രം(kernel_inp);
        സ്വതന്ത്രം(kernel_weight);
        സ്വതന്ത്രം(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    വേണ്ടി (പൂർണ്ണസംഖ്യ kernel_അക്കം = 0; kernel_അക്കം < അക്കം_KERNELS; kernel_അക്കം++) {
        printf("> Running kernel #%d\n", kernel_അക്കം);
        ഘടന timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < RUNS; i++) {
            matmul_forward(kernel_അക്കം, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        ഇരട്ട time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_അക്കം, time_elapsed_s * 1000);
    }

    // free memory
    സ്വതന്ത്രം(out);
    സ്വതന്ത്രം(inp);
    സ്വതന്ത്രം(weight);
    സ്വതന്ത്രം(bias);

    സ്വതന്ത്രം(grad_out);
    സ്വതന്ത്രം(grad_inp);
    സ്വതന്ത്രം(grad_weight);
    സ്വതന്ത്രം(grad_bias);

    തിരികെ 0;
}

ദശാംശം* make_random_float(size_t N) {
    ദശാംശം* arr = (ദശാംശം*)malloc(N * വലുപ്പം(ദശാംശം));
    വേണ്ടി (size_t i = 0; i < N; i++) {
        arr[i] = ((ദശാംശം)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    തിരികെ arr;
}

ശൂന്യം validate_results_cpu(സ്ഥിരം ദശാംശം* kernel_result, സ്ഥിരം ദശാംശം* cpu_reference, സ്ഥിരം അക്ഷരം* name, പൂർണ്ണസംഖ്യ അക്കം_elements, ദശാംശം tolerance) {
    പൂർണ്ണസംഖ്യ nfaults = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_elements; i++) {
        // print the first few comparisons
        എങ്കിൽ (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        ദശാംശം t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        എങ്കിൽ (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            എങ്കിൽ (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    എങ്കിൽ (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}