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

فراغ matmul_forward_cpu(عائم* بهر,
                    ثابت عائم* inp, ثابت عائم* وزن, ثابت عائم* bias,
                    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    لكل (صحيح b = 0; b < B; b++) {
        لكل (صحيح t = 0; t < T; t++) {
            عائم* بهر_bt = بهر + b * T * OC + t * OC;
            ثابت عائم* inp_bt = inp + b * T * C + t * C;
            لكل (صحيح o = 0; o < OC; o++) {
                عائم val = (bias != عدم) ? bias[o] : 0.0f;
                ثابت عائم* wrow = وزن + o*C;
                لكل (صحيح i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                بهر_bt[o] = val;
            }
        }
    }
}

فراغ matmul_forward_ngc92(عائم* بهر,
    ثابت عائم* inp, ثابت عائم* وزن, ثابت عائم* bias,
    صحيح B, صحيح T, صحيح C, صحيح OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #تعریف_کړه LOOP_UNROLL 8

    إذا (B * T % LOOP_UNROLL != 0) {
        چاپ_کړه("MUST BE A MULTIPLE OF 8"); // FIXME
        أرجع;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    لكل (صحيح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        لكل (صحيح o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            عائم پایله[LOOP_UNROLL];
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                پایله[ibt] = (bias != عدم) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            لكل (صحيح i = 0; i < C; i++) {
                عائم w = وزن[i + o * C];
                لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    صحيح bt = obt + ibt;
                    پایله[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            لكل (صحيح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                صحيح bt = obt + ibt;
                بهر[bt * OC + o] = پایله[ibt];
            }
        }
    }
}

#تعریف_کړه عدد_KERNELS 2

فراغ matmul_forward(صحيح kernel_عدد,
    عائم* بهر,
    ثابت عائم* inp, ثابت عائم* وزن, ثابت عائم* bias,
    صحيح B, صحيح T, صحيح C, صحيح OC) {

    بدل (kernel_عدد) {
        حالة 0:
            matmul_forward_cpu(بهر, inp, وزن, bias, B, T, C, OC);
            اكسر;
        حالة 1:
            matmul_forward_ngc92(بهر, inp, وزن, bias, B, T, C, OC);
            اكسر;
        افتراضي:
            چاپ_کړه("Invalid kernel number\n");
            وتل(1);
    }
}


فراغ validate_results_cpu(ثابت عائم* device_پایله, ثابت عائم* cpu_reference, ثابت حرف* name, صحيح عدد_elements, عائم tolerance);
عائم* make_random_float(size_t N);

صحيح اصلي(صحيح argc, حرف **argv) {
    srand(0);

    صحيح B = 8;
    صحيح T = 1024;
    صحيح C = 768;
    صحيح OC = 768 * 4; // expansion of 4, e.g. in the MLP
    صحيح RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    عائم* بهر = make_random_float(B * T * OC);
    عائم* inp = make_random_float(B * T * C);
    عائم* وزن = make_random_float(OC * C);
    عائم* bias = make_random_float(OC);

    عائم* grad_بهر = make_random_float(B * T * OC);
    عائم* grad_inp = make_random_float(B * T * C);
    عائم* grad_وزن = make_random_float(OC * C);
    عائم* grad_bias = make_random_float(OC);

    چاپ_کړه("> Calculating reference\n");
    matmul_forward_cpu(بهر, inp, وزن, bias, B, T, C, OC);

    لكل (صحيح kernel_عدد = 0; kernel_عدد < عدد_KERNELS; kernel_عدد++) {
        چاپ_کړه("> Verifying kernel #%d\n", kernel_عدد);

        srand(137);

        عائم* kernel_بهر = make_random_float(B * T * OC);
        عائم* kernel_inp = make_random_float(B * T * C);
        عائم* kernel_وزن = make_random_float(OC * C);
        عائم* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_عدد, kernel_بهر, kernel_inp, kernel_وزن, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_بهر, بهر, "out", B * T * OC, 1e-5);

        حرر(kernel_بهر);
        حرر(kernel_inp);
        حرر(kernel_وزن);
        حرر(kernel_bias);
    }

    چاپ_کړه("All kernels passed! Starting benchmarks.\n\n");

    لكل (صحيح kernel_عدد = 0; kernel_عدد < عدد_KERNELS; kernel_عدد++) {
        چاپ_کړه("> Running kernel #%d\n", kernel_عدد);
        بنية timespec پیل, پای_ټکی;
        clock_gettime(CLOCK_MONOTONIC, &پیل);

        لكل (صحيح i = 0; i < RUNS; i++) {
            matmul_forward(kernel_عدد, بهر, inp, وزن, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &پای_ټکی);
        مزدوج وخت_elapsed_s = (پای_ټکی.tv_sec - پیل.tv_sec) + (پای_ټکی.tv_nsec - پیل.tv_nsec) / 1e9;
        چاپ_کړه("> Kernel #%d, (took %f ms)\n", kernel_عدد, وخت_elapsed_s * 1000);
    }

    // free memory
    حرر(بهر);
    حرر(inp);
    حرر(وزن);
    حرر(bias);

    حرر(grad_بهر);
    حرر(grad_inp);
    حرر(grad_وزن);
    حرر(grad_bias);

    أرجع 0;
}

عائم* make_random_float(size_t N) {
    عائم* arr = (عائم*)حافظه_ورکړه(N * حجم(عائم));
    لكل (size_t i = 0; i < N; i++) {
        arr[i] = ((عائم)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    أرجع arr;
}

فراغ validate_results_cpu(ثابت عائم* kernel_پایله, ثابت عائم* cpu_reference, ثابت حرف* name, صحيح عدد_elements, عائم tolerance) {
    صحيح nfaults = 0;
    لكل (صحيح i = 0; i < عدد_elements; i++) {
        // print the first few comparisons
        إذا (i < 5) {
            چاپ_کړه("%f %f\n", cpu_reference[i], kernel_پایله[i]);
        }
        عائم t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        إذا (fabs(cpu_reference[i] - kernel_پایله[i]) > t_eff) {
            چاپ_کړه("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_پایله[i]);
            nfaults++;
            إذا (nfaults >= 10) {
                وتل(EXIT_FAILURE);
            }
        }
    }
    إذا (nfaults > 0) {
        وتل(EXIT_FAILURE);
    }
    چاپ_کړه("OK\n");
}