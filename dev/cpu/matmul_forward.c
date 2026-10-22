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

فراغ matmul_forward_cpu(عائم* بیرون,
                    ثابت عائم* inp, ثابت عائم* وزن, ثابت عائم* bias,
                    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* بیرون_bt = بیرون + b * T * OC + t * OC;
            ثابت عائم* inp_bt = inp + b * T * C + t * C;
            لكل (صحيح o = 0; o < OC; o++) {
                عائم val = (bias != عدم) ? bias[o] : 0.0f;
                ثابت عائم* wrow = وزن + o*C;
                لكل (صحيح i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                بیرون_bt[o] = val;
            }
        }
    }
}

فراغ matmul_forward_ngc92(عائم* بیرون,
    ثابت عائم* inp, ثابت عائم* وزن, ثابت عائم* bias,
    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #تعریف_کن LOOP_UNROLL 8

    إذا (B * T % LOOP_UNROLL != 0) {
        چاپ_کن("MUST BE A MULTIPLE OF 8"); // FIXME
        أرجع;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    لكل (صحيح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        لكل (صحيح o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            عائم نتیجه[LOOP_UNROLL];
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                نتیجه[ibt] = (bias != عدم) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            لكل (صحيح i = 0; i < C; i++) {
                عائم w = وزن[i + o * C];
                لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    صحيح bt = obt + ibt;
                    نتیجه[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                صحيح bt = obt + ibt;
                بیرون[bt * OC + o] = نتیجه[ibt];
            }
        }
    }
}

#تعریف_کن رقم_KERNELS 2

فراغ matmul_forward(صحيح kernel_رقم,
    عائم* بیرون,
    ثابت عائم* inp, ثابت عائم* وزن, ثابت عائم* bias,
    صحيح B, صحيح T, صحيح C, صحيح OC) {

    بدل (kernel_رقم) {
        حالة 0:
            matmul_forward_cpu(بیرون, inp, وزن, bias, B, T, C, OC);
            اكسر;
        حالة 1:
            matmul_forward_ngc92(بیرون, inp, وزن, bias, B, T, C, OC);
            اكسر;
        افتراضي:
            چاپ_کن("Invalid kernel number\n");
            خروج(1);
    }
}


فراغ validate_results_cpu(ثابت عائم* device_نتیجه, ثابت عائم* cpu_reference, ثابت حرف* name, صحيح رقم_elements, عائم tolerance);
عائم* make_random_float(size_t N);

صحيح اصلی(صحيح argc, حرف **argv) {
    srand(0);

    صحيح B = 8;
    صحيح T = 1024;
    صحيح C = 768;
    صحيح OC = 768 * 4; // expansion of 4, e.g. in the MLP
    صحيح RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    عائم* بیرون = make_random_float(B * T * OC);
    عائم* inp = make_random_float(B * T * C);
    عائم* وزن = make_random_float(OC * C);
    عائم* bias = make_random_float(OC);

    عائم* grad_بیرون = make_random_float(B * T * OC);
    عائم* grad_inp = make_random_float(B * T * C);
    عائم* grad_وزن = make_random_float(OC * C);
    عائم* grad_bias = make_random_float(OC);

    چاپ_کن("> Calculating reference\n");
    matmul_forward_cpu(بیرون, inp, وزن, bias, B, T, C, OC);

    لكل (صحيح kernel_رقم = 0; kernel_رقم < رقم_KERNELS; kernel_رقم++) {
        چاپ_کن("> Verifying kernel #%d\n", kernel_رقم);

        srand(137);

        عائم* kernel_بیرون = make_random_float(B * T * OC);
        عائم* kernel_inp = make_random_float(B * T * C);
        عائم* kernel_وزن = make_random_float(OC * C);
        عائم* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_رقم, kernel_بیرون, kernel_inp, kernel_وزن, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_بیرون, بیرون, "out", B * T * OC, 1e-5);

        حرر(kernel_بیرون);
        حرر(kernel_inp);
        حرر(kernel_وزن);
        حرر(kernel_bias);
    }

    چاپ_کن("All kernels passed! Starting benchmarks.\n\n");

    لكل (صحيح kernel_رقم = 0; kernel_رقم < رقم_KERNELS; kernel_رقم++) {
        چاپ_کن("> Running kernel #%d\n", kernel_رقم);
        بنية timespec شروع, انتها;
        clock_gettime(CLOCK_MONOTONIC, &شروع);

        لكل (صحيح i = 0; i < RUNS; i++) {
            matmul_forward(kernel_رقم, بیرون, inp, وزن, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &انتها);
        مزدوج زمان_elapsed_s = (انتها.tv_sec - شروع.tv_sec) + (انتها.tv_nsec - شروع.tv_nsec) / 1e9;
        چاپ_کن("> Kernel #%d, (took %f ms)\n", kernel_رقم, زمان_elapsed_s * 1000);
    }

    // free memory
    حرر(بیرون);
    حرر(inp);
    حرر(وزن);
    حرر(bias);

    حرر(grad_بیرون);
    حرر(grad_inp);
    حرر(grad_وزن);
    حرر(grad_bias);

    أرجع 0;
}

عائم* make_random_float(size_t N) {
    عائم* arr = (عائم*)حافظه_بده(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    أرجع arr;
}

فراغ validate_results_cpu(ثابت عائم* kernel_نتیجه, ثابت عائم* cpu_reference, ثابت حرف* name, صحيح رقم_elements, عائم tolerance) {
    صحيح nfaults = 0;
    لكل (صحيح i = 0; i < رقم_elements; i++) {
        // print the first few comparisons
        إذا (i < 5) {
            چاپ_کن("%f %f\n", cpu_reference[i], kernel_نتیجه[i]);
        }
        عائم t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        إذا (fabs(cpu_reference[i] - kernel_نتیجه[i]) > t_eff) {
            چاپ_کن("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_نتیجه[i]);
            nfaults++;
            إذا (nfaults >= 10) {
                خروج(EXIT_FAILURE);
            }
        }
    }
    إذا (nfaults > 0) {
        خروج(EXIT_FAILURE);
    }
    چاپ_کن("OK\n");
}