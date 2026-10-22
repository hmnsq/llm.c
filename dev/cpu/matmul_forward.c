/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <time.h>
#அந்தர்பாவ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

வெறுமை matmul_forward_cpu(தசம* out,
                    மாறிலி தசம* inp, மாறிலி தசம* weight, மாறிலி தசம* bias,
                    முழு B, முழு T, முழு C, முழு OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    க்கு (முழு b = 0; b < B; b++) {
        க்கு (முழு t = 0; t < T; t++) {
            தசம* out_bt = out + b * T * OC + t * OC;
            மாறிலி தசம* inp_bt = inp + b * T * C + t * C;
            க்கு (முழு o = 0; o < OC; o++) {
                தசம val = (bias != வெற்று) ? bias[o] : 0.0f;
                மாறிலி தசம* wrow = weight + o*C;
                க்கு (முழு i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

வெறுமை matmul_forward_ngc92(தசம* out,
    மாறிலி தசம* inp, மாறிலி தசம* weight, மாறிலி தசம* bias,
    முழு B, முழு T, முழு C, முழு OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    எனில் (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME
        திரும்பு;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    க்கு (முழு obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        க்கு (முழு o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            தசம result[LOOP_UNROLL];
            க்கு (முழு ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != வெற்று) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            க்கு (முழு i = 0; i < C; i++) {
                தசம w = weight[i + o * C];
                க்கு (முழு ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    முழு bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            க்கு (முழு ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                முழு bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define இலக்கம்_KERNELS 2

வெறுமை matmul_forward(முழு kernel_இலக்கம்,
    தசம* out,
    மாறிலி தசம* inp, மாறிலி தசம* weight, மாறிலி தசம* bias,
    முழு B, முழு T, முழு C, முழு OC) {

    தேர்வு (kernel_இலக்கம்) {
        நிலை 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            நிறுத்து;
        நிலை 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            நிறுத்து;
        இயல்பு:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


வெறுமை validate_results_cpu(மாறிலி தசம* device_result, மாறிலி தசம* cpu_reference, மாறிலி எழுத்து* name, முழு இலக்கம்_elements, தசம tolerance);
தசம* make_random_float(size_t N);

முழு முதன்மை(முழு argc, எழுத்து **argv) {
    srand(0);

    முழு B = 8;
    முழு T = 1024;
    முழு C = 768;
    முழு OC = 768 * 4; // expansion of 4, e.g. in the MLP
    முழு RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    தசம* out = make_random_float(B * T * OC);
    தசம* inp = make_random_float(B * T * C);
    தசம* weight = make_random_float(OC * C);
    தசம* bias = make_random_float(OC);

    தசம* grad_out = make_random_float(B * T * OC);
    தசம* grad_inp = make_random_float(B * T * C);
    தசம* grad_weight = make_random_float(OC * C);
    தசம* grad_bias = make_random_float(OC);

    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    க்கு (முழு kernel_இலக்கம் = 0; kernel_இலக்கம் < இலக்கம்_KERNELS; kernel_இலக்கம்++) {
        printf("> Verifying kernel #%d\n", kernel_இலக்கம்);

        srand(137);

        தசம* kernel_out = make_random_float(B * T * OC);
        தசம* kernel_inp = make_random_float(B * T * C);
        தசம* kernel_weight = make_random_float(OC * C);
        தசம* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_இலக்கம், kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        விடுவி(kernel_out);
        விடுவி(kernel_inp);
        விடுவி(kernel_weight);
        விடுவி(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    க்கு (முழு kernel_இலக்கம் = 0; kernel_இலக்கம் < இலக்கம்_KERNELS; kernel_இலக்கம்++) {
        printf("> Running kernel #%d\n", kernel_இலக்கம்);
        கட்டமைப்பு timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        க்கு (முழு i = 0; i < RUNS; i++) {
            matmul_forward(kernel_இலக்கம், out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        இரட்டை time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("> Kernel #%d, (took %f ms)\n", kernel_இலக்கம், time_elapsed_s * 1000);
    }

    // free memory
    விடுவி(out);
    விடுவி(inp);
    விடுவி(weight);
    விடுவி(bias);

    விடுவி(grad_out);
    விடுவி(grad_inp);
    விடுவி(grad_weight);
    விடுவி(grad_bias);

    திரும்பு 0;
}

தசம* make_random_float(size_t N) {
    தசம* arr = (தசம*)malloc(N * அளவு(தசம));
    க்கு (size_t i = 0; i < N; i++) {
        arr[i] = ((தசம)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    திரும்பு arr;
}

வெறுமை validate_results_cpu(மாறிலி தசம* kernel_result, மாறிலி தசம* cpu_reference, மாறிலி எழுத்து* name, முழு இலக்கம்_elements, தசம tolerance) {
    முழு nfaults = 0;
    க்கு (முழு i = 0; i < இலக்கம்_elements; i++) {
        // print the first few comparisons
        எனில் (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        தசம t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        எனில் (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            எனில் (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }
    எனில் (nfaults > 0) {
        exit(EXIT_FAILURE);
    }
    printf("OK\n");
}