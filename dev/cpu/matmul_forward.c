/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

शून्य matmul_forward_cpu(भिन्न* out,
                    ध्रुव भिन्न* inp, ध्रुव भिन्न* weight, ध्रुव भिन्न* bias,
                    पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C, पूर्णाङ्क OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    प्रति (पूर्णाङ्क b = 0; b < B; b++) {
        प्रति (पूर्णाङ्क t = 0; t < T; t++) {
            भिन्न* out_bt = out + b * T * OC + t * OC;
            ध्रुव भिन्न* inp_bt = inp + b * T * C + t * C;
            प्रति (पूर्णाङ्क o = 0; o < OC; o++) {
                भिन्न val = (bias != रिक्त) ? bias[o] : 0.0f;
                ध्रुव भिन्न* wrow = weight + o*C;
                प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

शून्य matmul_forward_ngc92(भिन्न* out,
    ध्रुव भिन्न* inp, ध्रुव भिन्न* weight, ध्रुव भिन्न* bias,
    पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C, पूर्णाङ्क OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    यदि (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        प्रत्यागम;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    प्रति (पूर्णाङ्क obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        प्रति (पूर्णाङ्क o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            भिन्न result[LOOP_UNROLL];
            प्रति (पूर्णाङ्क ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != रिक्त) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            प्रति (पूर्णाङ्क i = 0; i < C; i++) {
                भिन्न w = weight[i + o * C];
                प्रति (पूर्णाङ्क ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    पूर्णाङ्क bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            प्रति (पूर्णाङ्क ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                पूर्णाङ्क bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define अंक_KERNELS 2

शून्य matmul_forward(पूर्णाङ्क kernel_अंक,
    भिन्न* out,
    ध्रुव भिन्न* inp, ध्रुव भिन्न* weight, ध्रुव भिन्न* bias,
    पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C, पूर्णाङ्क OC) {

    विकल्प (kernel_अंक) {
        पक्ष 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            भङ्ग;
        पक्ष 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            भङ्ग;
        स्वभाव:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


शून्य validate_results_cpu(ध्रुव भिन्न* device_result, ध्रुव भिन्न* cpu_reference, ध्रुव वर्ण* name, पूर्णाङ्क अंक_elements, भिन्न tolerance);
भिन्न* make_random_float(size_t N);

पूर्णाङ्क main(पूर्णाङ्क argc, वर्ण **argv) {
    srand(0);

    पूर्णाङ्क B = 8;
    पूर्णाङ्क T = 1024;
    पूर्णाङ्क C = 768;
    पूर्णाङ्क OC = 768 * 4; // expansion of 4, e.g. in the MLP
    पूर्णाङ्क RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    भिन्न* out = make_random_float(B * T * OC);
    भिन्न* inp = make_random_float(B * T * C);
    भिन्न* weight = make_random_float(OC * C);
    भिन्न* bias = make_random_float(OC);

    भिन्न* grad_out = make_random_float(B * T * OC);
    भिन्न* grad_inp = make_random_float(B * T * C);
    भिन्न* grad_weight = make_random_float(OC * C);
    भिन्न* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    प्रति (पूर्णाङ्क kernel_अंक = 0; kernel_अंक < अंक_KERNELS; kernel_अंक++) {
        printf("> Verifying kernel #%d\n", kernel_अंक);

        srand(137);

        भिन्न* kernel_out = make_random_float(B * T * OC);
        भिन्न* kernel_inp = make_random_float(B * T * C);
        भिन्न* kernel_weight = make_random_float(OC * C);
        भिन्न* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_अंक, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        मुक्त(kernel_out);
        मुक्त(kernel_inp);
        मुक्त(kernel_weight);
        मुक्त(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    प्रति (पूर्णाङ्क kernel_अंक = 0; kernel_अंक < अंक_KERNELS; kernel_अंक++) {
        printf("> Running kernel #%d\n", kernel_अंक);
        संरचना timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        प्रति (पूर्णाङ्क i = 0; i < RUNS; i++) {
            matmul_forward(kernel_अंक, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुण time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_अंक, time_elapsed_s * 1000);
    }

    // free memory
    मुक्त(out);
    मुक्त(inp);
    मुक्त(weight);
    मुक्त(bias);

    मुक्त(grad_out);
    मुक्त(grad_inp);
    मुक्त(grad_weight);
    मुक्त(grad_bias);

    प्रत्यागम 0;
}

भिन्न* make_random_float(size_t N) {
    भिन्न* arr = (भिन्न*)malloc(N * परिमाण(भिन्न));
    प्रति (size_t i = 0; i < N; i++) {
        arr[i] = ((भिन्न)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    प्रत्यागम arr;
}

शून्य validate_results_cpu(ध्रुव भिन्न* kernel_result, ध्रुव भिन्न* cpu_reference, ध्रुव वर्ण* name, पूर्णाङ्क अंक_elements, भिन्न tolerance) {
    पूर्णाङ्क nfaults = 0;
    प्रति (पूर्णाङ्क i = 0; i < अंक_elements; i++) {
        // print the first few comparisons
        यदि (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        भिन्न t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        यदि (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            यदि (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    यदि (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}