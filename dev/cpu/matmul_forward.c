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

خالی matmul_forward_cpu(اعشاری* برآمد,
                    ثابت اعشاری* inp, ثابت اعشاری* وزن, ثابت اعشاری* bias,
                    صحیح B, صحیح T, صحیح C, صحیح OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    برائے (صحیح b = 0; b < B; b++) {
        برائے (صحیح t = 0; t < T; t++) {
            اعشاری* برآمد_bt = برآمد + b * T * OC + t * OC;
            ثابت اعشاری* inp_bt = inp + b * T * C + t * C;
            برائے (صحیح o = 0; o < OC; o++) {
                اعشاری val = (bias != تہی) ? bias[o] : 0.0f;
                ثابت اعشاری* wrow = وزن + o*C;
                برائے (صحیح i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                برآمد_bt[o] = val;
            }
        }
    }
}

خالی matmul_forward_ngc92(اعشاری* برآمد,
    ثابت اعشاری* inp, ثابت اعشاری* وزن, ثابت اعشاری* bias,
    صحیح B, صحیح T, صحیح C, صحیح OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #تعریف_کرو LOOP_UNROLL 8

    اگر (B * T % LOOP_UNROLL != 0) {
        لکھو("MUST BE A MULTIPLE OF 8"); // FIXME
        واپس;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    برائے (صحیح obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        برائے (صحیح o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            اعشاری نتیجہ[LOOP_UNROLL];
            برائے (صحیح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                نتیجہ[ibt] = (bias != تہی) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            برائے (صحیح i = 0; i < C; i++) {
                اعشاری w = وزن[i + o * C];
                برائے (صحیح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    صحیح bt = obt + ibt;
                    نتیجہ[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            برائے (صحیح ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                صحیح bt = obt + ibt;
                برآمد[bt * OC + o] = نتیجہ[ibt];
            }
        }
    }
}

#تعریف_کرو عدد_KERNELS 2

خالی matmul_forward(صحیح kernel_عدد,
    اعشاری* برآمد,
    ثابت اعشاری* inp, ثابت اعشاری* وزن, ثابت اعشاری* bias,
    صحیح B, صحیح T, صحیح C, صحیح OC) {

    انتخاب (kernel_عدد) {
        حالت 0:
            matmul_forward_cpu(برآمد, inp, وزن, bias, B, T, C, OC);
            توڑو;
        حالت 1:
            matmul_forward_ngc92(برآمد, inp, وزن, bias, B, T, C, OC);
            توڑو;
        طےشدہ:
            لکھو("Invalid kernel number\n");
            باہر_نکلو(1);
    }
}


خالی validate_results_cpu(ثابت اعشاری* device_نتیجہ, ثابت اعشاری* cpu_reference, ثابت حرف* name, صحیح عدد_elements, اعشاری tolerance);
اعشاری* make_random_float(size_t N);

صحیح مین(صحیح argc, حرف **argv) {
    srand(0);

    صحیح B = 8;
    صحیح T = 1024;
    صحیح C = 768;
    صحیح OC = 768 * 4; // expansion of 4, e.g. in the MLP
    صحیح RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    اعشاری* برآمد = make_random_float(B * T * OC);
    اعشاری* inp = make_random_float(B * T * C);
    اعشاری* وزن = make_random_float(OC * C);
    اعشاری* bias = make_random_float(OC);

    اعشاری* grad_برآمد = make_random_float(B * T * OC);
    اعشاری* grad_inp = make_random_float(B * T * C);
    اعشاری* grad_وزن = make_random_float(OC * C);
    اعشاری* grad_bias = make_random_float(OC);

    لکھو("> Calculating reference\n");
    matmul_forward_cpu(برآمد, inp, وزن, bias, B, T, C, OC);

    برائے (صحیح kernel_عدد = 0; kernel_عدد < عدد_KERNELS; kernel_عدد++) {
        لکھو("> Verifying kernel #%d\n", kernel_عدد);

        srand(137);

        اعشاری* kernel_برآمد = make_random_float(B * T * OC);
        اعشاری* kernel_inp = make_random_float(B * T * C);
        اعشاری* kernel_وزن = make_random_float(OC * C);
        اعشاری* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_عدد, kernel_برآمد, kernel_inp, kernel_وزن, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_برآمد, برآمد, "out", B * T * OC, 1e-5);

        آزاد(kernel_برآمد);
        آزاد(kernel_inp);
        آزاد(kernel_وزن);
        آزاد(kernel_bias);
    }

    لکھو("All kernels passed! Starting benchmarks.\n\n");

    برائے (صحیح kernel_عدد = 0; kernel_عدد < عدد_KERNELS; kernel_عدد++) {
        لکھو("> Running kernel #%d\n", kernel_عدد);
        ساخت timespec آغاز, انتہا;
        clock_gettime(CLOCK_MONOTONIC, &آغاز);

        برائے (صحیح i = 0; i < RUNS; i++) {
            matmul_forward(kernel_عدد, برآمد, inp, وزن, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &انتہا);
        دوہرا وقت_elapsed_s = (انتہا.tv_sec - آغاز.tv_sec) + (انتہا.tv_nsec - آغاز.tv_nsec) / 1e9;
        لکھو("> Kernel #%d, (took %f ms)\n", kernel_عدد, وقت_elapsed_s * 1000);
    }

    // free memory
    آزاد(برآمد);
    آزاد(inp);
    آزاد(وزن);
    آزاد(bias);

    آزاد(grad_برآمد);
    آزاد(grad_inp);
    آزاد(grad_وزن);
    آزاد(grad_bias);

    واپس 0;
}

اعشاری* make_random_float(size_t N) {
    اعشاری* arr = (اعشاری*)حافظہ_دو(N * حجم(اعشاری));
    برائے (size_t i = 0; i < N; i++) {
        arr[i] = ((اعشاری)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    واپس arr;
}

خالی validate_results_cpu(ثابت اعشاری* kernel_نتیجہ, ثابت اعشاری* cpu_reference, ثابت حرف* name, صحیح عدد_elements, اعشاری tolerance) {
    صحیح nfaults = 0;
    برائے (صحیح i = 0; i < عدد_elements; i++) {
        // print the first few comparisons
        اگر (i < 5) {
            لکھو("%f %f\n", cpu_reference[i], kernel_نتیجہ[i]);
        }
        اعشاری t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        اگر (fabs(cpu_reference[i] - kernel_نتیجہ[i]) > t_eff) {
            لکھو("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_نتیجہ[i]);
            nfaults++;
            اگر (nfaults >= 10) {
                باہر_نکلو(EXIT_FAILURE);
            }
        }
    }
    اگر (nfaults > 0) {
        باہر_نکلو(EXIT_FAILURE);
    }
    لکھو("OK\n");
}