/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#অন্তর্ভাৱ <stdio.h>
#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <time.h>
#অন্তর্ভাৱ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

শূন্য matmul_forward_cpu(ভিন্ন* out,
                    ধ্রুৱ ভিন্ন* inp, ধ্রুৱ ভিন্ন* weight, ধ্রুৱ ভিন্ন* bias,
                    পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক C, পূর্ণাঙ্ক OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    বাবে (পূর্ণাঙ্ক b = 0; b < B; b++) {
        বাবে (পূর্ণাঙ্ক t = 0; t < T; t++) {
            ভিন্ন* out_bt = out + b * T * OC + t * OC;
            ধ্রুৱ ভিন্ন* inp_bt = inp + b * T * C + t * C;
            বাবে (পূর্ণাঙ্ক o = 0; o < OC; o++) {
                ভিন্ন val = (bias != রিক্ত) ? bias[o] : 0.0f;
                ধ্রুৱ ভিন্ন* wrow = weight + o*C;
                বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

শূন্য matmul_forward_ngc92(ভিন্ন* out,
    ধ্রুৱ ভিন্ন* inp, ধ্রুৱ ভিন্ন* weight, ধ্রুৱ ভিন্ন* bias,
    পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক C, পূর্ণাঙ্ক OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    যদি (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        ঘূৰাই;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    বাবে (পূর্ণাঙ্ক obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        বাবে (পূর্ণাঙ্ক o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            ভিন্ন result[LOOP_UNROLL];
            বাবে (পূর্ণাঙ্ক ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != রিক্ত) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            বাবে (পূর্ণাঙ্ক i = 0; i < C; i++) {
                ভিন্ন w = weight[i + o * C];
                বাবে (পূর্ণাঙ্ক ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    পূর্ণাঙ্ক bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            বাবে (পূর্ণাঙ্ক ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                পূর্ণাঙ্ক bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define অঙ্ক_KERNELS 2

শূন্য matmul_forward(পূর্ণাঙ্ক kernel_অঙ্ক,
    ভিন্ন* out,
    ধ্রুৱ ভিন্ন* inp, ধ্রুৱ ভিন্ন* weight, ধ্রুৱ ভিন্ন* bias,
    পূর্ণাঙ্ক B, পূর্ণাঙ্ক T, পূর্ণাঙ্ক C, পূর্ণাঙ্ক OC) {

    ৱিকল্প (kernel_অঙ্ক) {
        পক্ষ 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            ভঙ্গ;
        পক্ষ 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            ভঙ্গ;
        স্ৱভাৱ:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


শূন্য validate_results_cpu(ধ্রুৱ ভিন্ন* device_result, ধ্রুৱ ভিন্ন* cpu_reference, ধ্রুৱ ৱর্ণ* name, পূর্ণাঙ্ক অঙ্ক_elements, ভিন্ন tolerance);
ভিন্ন* make_random_float(size_t N);

পূর্ণাঙ্ক main(পূর্ণাঙ্ক argc, ৱর্ণ **argv) {
    srand(0);

    পূর্ণাঙ্ক B = 8;
    পূর্ণাঙ্ক T = 1024;
    পূর্ণাঙ্ক C = 768;
    পূর্ণাঙ্ক OC = 768 * 4; // expansion of 4, e.g. in the MLP
    পূর্ণাঙ্ক RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    ভিন্ন* out = make_random_float(B * T * OC);
    ভিন্ন* inp = make_random_float(B * T * C);
    ভিন্ন* weight = make_random_float(OC * C);
    ভিন্ন* bias = make_random_float(OC);

    ভিন্ন* grad_out = make_random_float(B * T * OC);
    ভিন্ন* grad_inp = make_random_float(B * T * C);
    ভিন্ন* grad_weight = make_random_float(OC * C);
    ভিন্ন* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    বাবে (পূর্ণাঙ্ক kernel_অঙ্ক = 0; kernel_অঙ্ক < অঙ্ক_KERNELS; kernel_অঙ্ক++) {
        printf("> Verifying kernel #%d\n", kernel_অঙ্ক);

        srand(137);

        ভিন্ন* kernel_out = make_random_float(B * T * OC);
        ভিন্ন* kernel_inp = make_random_float(B * T * C);
        ভিন্ন* kernel_weight = make_random_float(OC * C);
        ভিন্ন* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_অঙ্ক, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        মুক্ত(kernel_out);
        মুক্ত(kernel_inp);
        মুক্ত(kernel_weight);
        মুক্ত(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    বাবে (পূর্ণাঙ্ক kernel_অঙ্ক = 0; kernel_অঙ্ক < অঙ্ক_KERNELS; kernel_অঙ্ক++) {
        printf("> Running kernel #%d\n", kernel_অঙ্ক);
        সংরচনা timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        বাবে (পূর্ণাঙ্ক i = 0; i < RUNS; i++) {
            matmul_forward(kernel_অঙ্ক, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        দ্ৱিগুণ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_অঙ্ক, time_elapsed_s * 1000);
    }

    // free memory
    মুক্ত(out);
    মুক্ত(inp);
    মুক্ত(weight);
    মুক্ত(bias);

    মুক্ত(grad_out);
    মুক্ত(grad_inp);
    মুক্ত(grad_weight);
    মুক্ত(grad_bias);

    ঘূৰাই 0;
}

ভিন্ন* make_random_float(size_t N) {
    ভিন্ন* arr = (ভিন্ন*)malloc(N * পরিমাণ(ভিন্ন));
    বাবে (size_t i = 0; i < N; i++) {
        arr[i] = ((ভিন্ন)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ঘূৰাই arr;
}

শূন্য validate_results_cpu(ধ্রুৱ ভিন্ন* kernel_result, ধ্রুৱ ভিন্ন* cpu_reference, ধ্রুৱ ৱর্ণ* name, পূর্ণাঙ্ক অঙ্ক_elements, ভিন্ন tolerance) {
    পূর্ণাঙ্ক nfaults = 0;
    বাবে (পূর্ণাঙ্ক i = 0; i < অঙ্ক_elements; i++) {
        // print the first few comparisons
        যদি (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        ভিন্ন t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        যদি (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            যদি (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    যদি (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}