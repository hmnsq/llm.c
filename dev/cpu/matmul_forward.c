/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <math.h>
#شامل <time.h>
#شامل <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

تهی matmul_forward_cpu(اعشاری* out,
                    ثابت اعشاری* inp, ثابت اعشاری* weight, ثابت اعشاری* bias,
                    صحیح B, صحیح T, صحیح C, صحیح OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    برای (صحیح b = 0; b < B; b++) {
        برای (صحیح t = 0; t < T; t++) {
            اعشاری* out_bt = out + b * T * OC + t * OC;
            ثابت اعشاری* inp_bt = inp + b * T * C + t * C;
            برای (صحیح o = 0; o < OC; o++) {
                اعشاری val = (bias != پوچ) ? bias[o] : 0.0f;
                ثابت اعشاری* wrow = weight + o*C;
                برای (صحیح i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

تهی matmul_forward_ngc92(اعشاری* out,
    ثابت اعشاری* inp, ثابت اعشاری* weight, ثابت اعشاری* bias,
    صحیح B, صحیح T, صحیح C, صحیح OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    اگر (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        بازگشت;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    برای (صحیح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        برای (صحیح o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            اعشاری result[LOOP_UNROLL];
            برای (صحیح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != پوچ) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            برای (صحیح i = 0; i < C; i++) {
                اعشاری w = weight[i + o * C];
                برای (صحیح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    صحیح bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            برای (صحیح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                صحیح bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define رقم_KERNELS 2

تهی matmul_forward(صحیح kernel_رقم,
    اعشاری* out,
    ثابت اعشاری* inp, ثابت اعشاری* weight, ثابت اعشاری* bias,
    صحیح B, صحیح T, صحیح C, صحیح OC) {

    انتخاب (kernel_رقم) {
        حالت 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            بشکن;
        حالت 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            بشکن;
        پیشفرض:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


تهی validate_results_cpu(ثابت اعشاری* device_result, ثابت اعشاری* cpu_reference, ثابت نویسه* name, صحیح رقم_elements, اعشاری tolerance);
اعشاری* make_random_float(size_t N);

صحیح اصلی(صحیح argc, نویسه **argv) {
    srand(0);

    صحیح B = 8;
    صحیح T = 1024;
    صحیح C = 768;
    صحیح OC = 768 * 4; // expansion of 4, e.g. in the MLP
    صحیح RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    اعشاری* out = make_random_float(B * T * OC);
    اعشاری* inp = make_random_float(B * T * C);
    اعشاری* weight = make_random_float(OC * C);
    اعشاری* bias = make_random_float(OC);

    اعشاری* grad_out = make_random_float(B * T * OC);
    اعشاری* grad_inp = make_random_float(B * T * C);
    اعشاری* grad_weight = make_random_float(OC * C);
    اعشاری* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    برای (صحیح kernel_رقم = 0; kernel_رقم < رقم_KERNELS; kernel_رقم++) {
        printf("> Verifying kernel #%d\n", kernel_رقم);

        srand(137);

        اعشاری* kernel_out = make_random_float(B * T * OC);
        اعشاری* kernel_inp = make_random_float(B * T * C);
        اعشاری* kernel_weight = make_random_float(OC * C);
        اعشاری* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_رقم, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        آزاد(kernel_out);
        آزاد(kernel_inp);
        آزاد(kernel_weight);
        آزاد(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    برای (صحیح kernel_رقم = 0; kernel_رقم < رقم_KERNELS; kernel_رقم++) {
        printf("> Running kernel #%d\n", kernel_رقم);
        ساختار timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        برای (صحیح i = 0; i < RUNS; i++) {
            matmul_forward(kernel_رقم, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        دوگانه time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_رقم, time_elapsed_s * 1000);
    }

    // free memory
    آزاد(out);
    آزاد(inp);
    آزاد(weight);
    آزاد(bias);

    آزاد(grad_out);
    آزاد(grad_inp);
    آزاد(grad_weight);
    آزاد(grad_bias);

    بازگشت 0;
}

اعشاری* make_random_float(size_t N) {
    اعشاری* arr = (اعشاری*)malloc(N * اندازه(اعشاری));
    برای (size_t i = 0; i < N; i++) {
        arr[i] = ((اعشاری)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    بازگشت arr;
}

تهی validate_results_cpu(ثابت اعشاری* kernel_result, ثابت اعشاری* cpu_reference, ثابت نویسه* name, صحیح رقم_elements, اعشاری tolerance) {
    صحیح nfaults = 0;
    برای (صحیح i = 0; i < رقم_elements; i++) {
        // print the first few comparisons
        اگر (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        اعشاری t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        اگر (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            اگر (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    اگر (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}