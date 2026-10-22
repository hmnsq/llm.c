/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <math.h>
#ضمن <time.h>
#ضمن <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

فراغ matmul_forward_cpu(عائم* out,
                    ثابت عائم* inp, ثابت عائم* weight, ثابت عائم* bias,
                    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* out_bt = out + b * T * OC + t * OC;
            ثابت عائم* inp_bt = inp + b * T * C + t * C;
            لكل (صحيح o = 0; o < OC; o++) {
                عائم val = (bias != عدم) ? bias[o] : 0.0f;
                ثابت عائم* wrow = weight + o*C;
                لكل (صحيح i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

فراغ matmul_forward_ngc92(عائم* out,
    ثابت عائم* inp, ثابت عائم* weight, ثابت عائم* bias,
    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    ܐܢ (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        ܦܢܐ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    لكل (صحيح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        لكل (صحيح o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            عائم result[LOOP_UNROLL];
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != عدم) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            لكل (صحيح i = 0; i < C; i++) {
                عائم w = weight[i + o * C];
                لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    صحيح bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                صحيح bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define NUM_KERNELS 2

فراغ matmul_forward(صحيح kernel_num,
    عائم* out,
    ثابت عائم* inp, ثابت عائم* weight, ثابت عائم* bias,
    صحيح B, صحيح T, صحيح C, صحيح OC) {

    بدل (kernel_num) {
        حالة 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            اكسر;
        حالة 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            اكسر;
        افتراضي:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


فراغ validate_results_cpu(ثابت عائم* device_result, ثابت عائم* cpu_reference, ثابت حرف* name, صحيح num_elements, عائم tolerance);
عائم* make_random_float(size_t N);

صحيح main(صحيح argc, حرف **argv) {
    srand(0);

    صحيح B = 8;
    صحيح T = 1024;
    صحيح C = 768;
    صحيح OC = 768 * 4; // expansion of 4, e.g. in the MLP
    صحيح RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    عائم* out = make_random_float(B * T * OC);
    عائم* inp = make_random_float(B * T * C);
    عائم* weight = make_random_float(OC * C);
    عائم* bias = make_random_float(OC);

    عائم* grad_out = make_random_float(B * T * OC);
    عائم* grad_inp = make_random_float(B * T * C);
    عائم* grad_weight = make_random_float(OC * C);
    عائم* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    لكل (صحيح kernel_num = 0; kernel_num < NUM_KERNELS; kernel_num++) {
        printf("> Verifying kernel #%d\n", kernel_num);

        srand(137);

        عائم* kernel_out = make_random_float(B * T * OC);
        عائم* kernel_inp = make_random_float(B * T * C);
        عائم* kernel_weight = make_random_float(OC * C);
        عائم* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_num, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        حرر(kernel_out);
        حرر(kernel_inp);
        حرر(kernel_weight);
        حرر(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    لكل (صحيح kernel_num = 0; kernel_num < NUM_KERNELS; kernel_num++) {
        printf("> Running kernel #%d\n", kernel_num);
        بنية timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        لكل (صحيح i = 0; i < RUNS; i++) {
            matmul_forward(kernel_num, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        مزدوج time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    // free memory
    حرر(out);
    حرر(inp);
    حرر(weight);
    حرر(bias);

    حرر(grad_out);
    حرر(grad_inp);
    حرر(grad_weight);
    حرر(grad_bias);

    ܦܢܐ 0;
}

عائم* make_random_float(size_t N) {
    عائم* arr = (عائم*)malloc(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ܦܢܐ arr;
}

فراغ validate_results_cpu(ثابت عائم* kernel_result, ثابت عائم* cpu_reference, ثابت حرف* name, صحيح num_elements, عائم tolerance) {
    صحيح nfaults = 0;
    لكل (صحيح i = 0; i < num_elements; i++) {
        // print the first few comparisons
        ܐܢ (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        عائم t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        ܐܢ (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            ܐܢ (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    ܐܢ (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}