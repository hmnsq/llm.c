/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <time.h>
#અન્તર્ભાવ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

શૂન્ય matmul_forward_cpu(દશાંશ* out,
                    અચલ દશાંશ* inp, અચલ દશાંશ* weight, અચલ દશાંશ* bias,
                    પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક C, પૂર્ણાંક OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    માટે (પૂર્ણાંક b = 0; b < B; b++) {
        માટે (પૂર્ણાંક t = 0; t < T; t++) {
            દશાંશ* out_bt = out + b * T * OC + t * OC;
            અચલ દશાંશ* inp_bt = inp + b * T * C + t * C;
            માટે (પૂર્ણાંક o = 0; o < OC; o++) {
                દશાંશ val = (bias != ખાલી) ? bias[o] : 0.0f;
                અચલ દશાંશ* wrow = weight + o*C;
                માટે (પૂર્ણાંક i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

શૂન્ય matmul_forward_ngc92(દશાંશ* out,
    અચલ દશાંશ* inp, અચલ દશાંશ* weight, અચલ દશાંશ* bias,
    પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક C, પૂર્ણાંક OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    જો (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        પાછા;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    માટે (પૂર્ણાંક obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        માટે (પૂર્ણાંક o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            દશાંશ result[LOOP_UNROLL];
            માટે (પૂર્ણાંક ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != ખાલી) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            માટે (પૂર્ણાંક i = 0; i < C; i++) {
                દશાંશ w = weight[i + o * C];
                માટે (પૂર્ણાંક ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    પૂર્ણાંક bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            માટે (પૂર્ણાંક ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                પૂર્ણાંક bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define અંક_KERNELS 2

શૂન્ય matmul_forward(પૂર્ણાંક kernel_અંક,
    દશાંશ* out,
    અચલ દશાંશ* inp, અચલ દશાંશ* weight, અચલ દશાંશ* bias,
    પૂર્ણાંક B, પૂર્ણાંક T, પૂર્ણાંક C, પૂર્ણાંક OC) {

    પસંદગી (kernel_અંક) {
        સ્થિતિ 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            તોડો;
        સ્થિતિ 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            તોડો;
        મૂળભૂત:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


શૂન્ય validate_results_cpu(અચલ દશાંશ* device_result, અચલ દશાંશ* cpu_reference, અચલ અક્ષર* name, પૂર્ણાંક અંક_elements, દશાંશ tolerance);
દશાંશ* make_random_float(size_t N);

પૂર્ણાંક મુખ્ય(પૂર્ણાંક argc, અક્ષર **argv) {
    srand(0);

    પૂર્ણાંક B = 8;
    પૂર્ણાંક T = 1024;
    પૂર્ણાંક C = 768;
    પૂર્ણાંક OC = 768 * 4; // expansion of 4, e.g. in the MLP
    પૂર્ણાંક RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    દશાંશ* out = make_random_float(B * T * OC);
    દશાંશ* inp = make_random_float(B * T * C);
    દશાંશ* weight = make_random_float(OC * C);
    દશાંશ* bias = make_random_float(OC);

    દશાંશ* grad_out = make_random_float(B * T * OC);
    દશાંશ* grad_inp = make_random_float(B * T * C);
    દશાંશ* grad_weight = make_random_float(OC * C);
    દશાંશ* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    માટે (પૂર્ણાંક kernel_અંક = 0; kernel_અંક < અંક_KERNELS; kernel_અંક++) {
        printf("> Verifying kernel #%d\n", kernel_અંક);

        srand(137);

        દશાંશ* kernel_out = make_random_float(B * T * OC);
        દશાંશ* kernel_inp = make_random_float(B * T * C);
        દશાંશ* kernel_weight = make_random_float(OC * C);
        દશાંશ* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_અંક, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        મુક્ત(kernel_out);
        મુક્ત(kernel_inp);
        મુક્ત(kernel_weight);
        મુક્ત(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    માટે (પૂર્ણાંક kernel_અંક = 0; kernel_અંક < અંક_KERNELS; kernel_અંક++) {
        printf("> Running kernel #%d\n", kernel_અંક);
        સંરચના timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        માટે (પૂર્ણાંક i = 0; i < RUNS; i++) {
            matmul_forward(kernel_અંક, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        દ્વિગુણ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_અંક, time_elapsed_s * 1000);
    }

    // free memory
    મુક્ત(out);
    મુક્ત(inp);
    મુક્ત(weight);
    મુક્ત(bias);

    મુક્ત(grad_out);
    મુક્ત(grad_inp);
    મુક્ત(grad_weight);
    મુક્ત(grad_bias);

    પાછા 0;
}

દશાંશ* make_random_float(size_t N) {
    દશાંશ* arr = (દશાંશ*)malloc(N * કદ(દશાંશ));
    માટે (size_t i = 0; i < N; i++) {
        arr[i] = ((દશાંશ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    પાછા arr;
}

શૂન્ય validate_results_cpu(અચલ દશાંશ* kernel_result, અચલ દશાંશ* cpu_reference, અચલ અક્ષર* name, પૂર્ણાંક અંક_elements, દશાંશ tolerance) {
    પૂર્ણાંક nfaults = 0;
    માટે (પૂર્ણાંક i = 0; i < અંક_elements; i++) {
        // print the first few comparisons
        જો (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        દશાંશ t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        જો (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            જો (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    જો (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}