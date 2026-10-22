/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <time.h>
#ଅନ୍ତର୍ଭାଵ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

ଶୂନ୍ୟ matmul_forward_cpu(ଦଶମିକ* out,
                    ଅଚଳ ଦଶମିକ* inp, ଅଚଳ ଦଶମିକ* weight, ଅଚଳ ଦଶମିକ* bias,
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, ପୂର୍ଣ୍ଣସଂଖ୍ୟା OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = 0; b < B; b++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < T; t++) {
            ଦଶମିକ* out_bt = out + b * T * OC + t * OC;
            ଅଚଳ ଦଶମିକ* inp_bt = inp + b * T * C + t * C;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା o = 0; o < OC; o++) {
                ଦଶମିକ val = (bias != ଖାଲି) ? bias[o] : 0.0f;
                ଅଚଳ ଦଶମିକ* wrow = weight + o*C;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

ଶୂନ୍ୟ matmul_forward_ngc92(ଦଶମିକ* out,
    ଅଚଳ ଦଶମିକ* inp, ଅଚଳ ଦଶମିକ* weight, ଅଚଳ ଦଶମିକ* bias,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, ପୂର୍ଣ୍ଣସଂଖ୍ୟା OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    ଯଦି (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        ଫେରନ୍ତୁ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            ଦଶମିକ result[LOOP_UNROLL];
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != ଖାଲି) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < C; i++) {
                ଦଶମିକ w = weight[i + o * C];
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define ଅଙ୍କ_KERNELS 2

ଶୂନ୍ୟ matmul_forward(ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_ଅଙ୍କ,
    ଦଶମିକ* out,
    ଅଚଳ ଦଶମିକ* inp, ଅଚଳ ଦଶମିକ* weight, ଅଚଳ ଦଶମିକ* bias,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B, ପୂର୍ଣ୍ଣସଂଖ୍ୟା T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା C, ପୂର୍ଣ୍ଣସଂଖ୍ୟା OC) {

    ବାଛ (kernel_ଅଙ୍କ) {
        ସ୍ଥିତି 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


ଶୂନ୍ୟ validate_results_cpu(ଅଚଳ ଦଶମିକ* device_result, ଅଚଳ ଦଶମିକ* cpu_reference, ଅଚଳ ଅକ୍ଷର* name, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elements, ଦଶମିକ tolerance);
ଦଶମିକ* make_random_float(size_t N);

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୁଖ୍ୟ(ପୂର୍ଣ୍ଣସଂଖ୍ୟା argc, ଅକ୍ଷର **argv) {
    srand(0);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା B = 8;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା T = 1024;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା C = 768;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା OC = 768 * 4; // expansion of 4, e.g. in the MLP
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    ଦଶମିକ* out = make_random_float(B * T * OC);
    ଦଶମିକ* inp = make_random_float(B * T * C);
    ଦଶମିକ* weight = make_random_float(OC * C);
    ଦଶମିକ* bias = make_random_float(OC);

    ଦଶମିକ* grad_out = make_random_float(B * T * OC);
    ଦଶମିକ* grad_inp = make_random_float(B * T * C);
    ଦଶମିକ* grad_weight = make_random_float(OC * C);
    ଦଶମିକ* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_ଅଙ୍କ = 0; kernel_ଅଙ୍କ < ଅଙ୍କ_KERNELS; kernel_ଅଙ୍କ++) {
        printf("> Verifying kernel #%d\n", kernel_ଅଙ୍କ);

        srand(137);

        ଦଶମିକ* kernel_out = make_random_float(B * T * OC);
        ଦଶମିକ* kernel_inp = make_random_float(B * T * C);
        ଦଶମିକ* kernel_weight = make_random_float(OC * C);
        ଦଶମିକ* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_ଅଙ୍କ, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        ମୁକ୍ତ(kernel_out);
        ମୁକ୍ତ(kernel_inp);
        ମୁକ୍ତ(kernel_weight);
        ମୁକ୍ତ(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_ଅଙ୍କ = 0; kernel_ଅଙ୍କ < ଅଙ୍କ_KERNELS; kernel_ଅଙ୍କ++) {
        printf("> Running kernel #%d\n", kernel_ଅଙ୍କ);
        ସଂରଚନା timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < RUNS; i++) {
            matmul_forward(kernel_ଅଙ୍କ, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        ଦ୍ୱିଗୁଣ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_ଅଙ୍କ, time_elapsed_s * 1000);
    }

    // free memory
    ମୁକ୍ତ(out);
    ମୁକ୍ତ(inp);
    ମୁକ୍ତ(weight);
    ମୁକ୍ତ(bias);

    ମୁକ୍ତ(grad_out);
    ମୁକ୍ତ(grad_inp);
    ମୁକ୍ତ(grad_weight);
    ମୁକ୍ତ(grad_bias);

    ଫେରନ୍ତୁ 0;
}

ଦଶମିକ* make_random_float(size_t N) {
    ଦଶମିକ* arr = (ଦଶମିକ*)malloc(N * ଆକାର(ଦଶମିକ));
    ପାଇଁ (size_t i = 0; i < N; i++) {
        arr[i] = ((ଦଶମିକ)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ଫେରନ୍ତୁ arr;
}

ଶୂନ୍ୟ validate_results_cpu(ଅଚଳ ଦଶମିକ* kernel_result, ଅଚଳ ଦଶମିକ* cpu_reference, ଅଚଳ ଅକ୍ଷର* name, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_elements, ଦଶମିକ tolerance) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nfaults = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_elements; i++) {
        // print the first few comparisons
        ଯଦି (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        ଦଶମିକ t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        ଯଦି (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            ଯଦି (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    ଯଦି (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}