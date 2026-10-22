/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <time.h>
#অন্তর্ভাব <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

শূন্য matmul_forward_cpu(দশমিক* out,
                    ধ্রুবক দশমিক* inp, ধ্রুবক দশমিক* weight, ধ্রুবক দশমিক* bias,
                    পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    জন্য (পূর্ণসংখ্যা b = 0; b < B; b++) {
        জন্য (পূর্ণসংখ্যা t = 0; t < T; t++) {
            দশমিক* out_bt = out + b * T * OC + t * OC;
            ধ্রুবক দশমিক* inp_bt = inp + b * T * C + t * C;
            জন্য (পূর্ণসংখ্যা o = 0; o < OC; o++) {
                দশমিক val = (bias != ফাঁকা) ? bias[o] : 0.0f;
                ধ্রুবক দশমিক* wrow = weight + o*C;
                জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

শূন্য matmul_forward_ngc92(দশমিক* out,
    ধ্রুবক দশমিক* inp, ধ্রুবক দশমিক* weight, ধ্রুবক দশমিক* bias,
    পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    যদি (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        ফেরত;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    জন্য (পূর্ণসংখ্যা obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        জন্য (পূর্ণসংখ্যা o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            দশমিক ফলাফল[LOOP_UNROLL];
            জন্য (পূর্ণসংখ্যা ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ফলাফল[ibt] = (bias != ফাঁকা) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            জন্য (পূর্ণসংখ্যা i = 0; i < C; i++) {
                দশমিক w = weight[i + o * C];
                জন্য (পূর্ণসংখ্যা ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    পূর্ণসংখ্যা bt = obt + ibt;
                    ফলাফল[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            জন্য (পূর্ণসংখ্যা ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                পূর্ণসংখ্যা bt = obt + ibt;
                out[bt * OC + o] = ফলাফল[ibt];
            }
        }
    }
}

#define অঙ্ক_KERNELS 2

শূন্য matmul_forward(পূর্ণসংখ্যা kernel_অঙ্ক,
    দশমিক* out,
    ধ্রুবক দশমিক* inp, ধ্রুবক দশমিক* weight, ধ্রুবক দশমিক* bias,
    পূর্ণসংখ্যা B, পূর্ণসংখ্যা T, পূর্ণসংখ্যা C, পূর্ণসংখ্যা OC) {

    নির্বাচন (kernel_অঙ্ক) {
        ক্ষেত্র 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            বিরতি;
        ক্ষেত্র 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            বিরতি;
        পূর্বনির্ধারিত:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


শূন্য validate_results_cpu(ধ্রুবক দশমিক* device_ফলাফল, ধ্রুবক দশমিক* cpu_reference, ধ্রুবক অক্ষর* name, পূর্ণসংখ্যা অঙ্ক_elements, দশমিক tolerance);
দশমিক* make_random_float(size_t N);

পূর্ণসংখ্যা প্রধান(পূর্ণসংখ্যা argc, অক্ষর **argv) {
    srand(0);

    পূর্ণসংখ্যা B = 8;
    পূর্ণসংখ্যা T = 1024;
    পূর্ণসংখ্যা C = 768;
    পূর্ণসংখ্যা OC = 768 * 4; // expansion of 4, e.g. in the MLP
    পূর্ণসংখ্যা RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    দশমিক* out = make_random_float(B * T * OC);
    দশমিক* inp = make_random_float(B * T * C);
    দশমিক* weight = make_random_float(OC * C);
    দশমিক* bias = make_random_float(OC);

    দশমিক* grad_out = make_random_float(B * T * OC);
    দশমিক* grad_inp = make_random_float(B * T * C);
    দশমিক* grad_weight = make_random_float(OC * C);
    দশমিক* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    জন্য (পূর্ণসংখ্যা kernel_অঙ্ক = 0; kernel_অঙ্ক < অঙ্ক_KERNELS; kernel_অঙ্ক++) {
        printf("> Verifying kernel #%d\n", kernel_অঙ্ক);

        srand(137);

        দশমিক* kernel_out = make_random_float(B * T * OC);
        দশমিক* kernel_inp = make_random_float(B * T * C);
        দশমিক* kernel_weight = make_random_float(OC * C);
        দশমিক* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_অঙ্ক, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        মুক্ত(kernel_out);
        মুক্ত(kernel_inp);
        মুক্ত(kernel_weight);
        মুক্ত(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    জন্য (পূর্ণসংখ্যা kernel_অঙ্ক = 0; kernel_অঙ্ক < অঙ্ক_KERNELS; kernel_অঙ্ক++) {
        printf("> Running kernel #%d\n", kernel_অঙ্ক);
        কাঠামো timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        জন্য (পূর্ণসংখ্যা i = 0; i < RUNS; i++) {
            matmul_forward(kernel_অঙ্ক, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        দ্বিগুণ time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
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

    ফেরত 0;
}

দশমিক* make_random_float(size_t N) {
    দশমিক* arr = (দশমিক*)malloc(N * আকার(দশমিক));
    জন্য (size_t i = 0; i < N; i++) {
        arr[i] = ((দশমিক)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    ফেরত arr;
}

শূন্য validate_results_cpu(ধ্রুবক দশমিক* kernel_ফলাফল, ধ্রুবক দশমিক* cpu_reference, ধ্রুবক অক্ষর* name, পূর্ণসংখ্যা অঙ্ক_elements, দশমিক tolerance) {
    পূর্ণসংখ্যা nfaults = 0;
    জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_elements; i++) {
        // print the first few comparisons
        যদি (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_ফলাফল[i]);
        }
        দশমিক t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        যদি (fabs(cpu_reference[i] - kernel_ফলাফল[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_ফলাফল[i]);
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