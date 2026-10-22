/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <time.h>
#अन्तर्भाव <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

शून्य matmul_forward_cpu(दशांश* out,
                    स्थिरांक दशांश* inp, स्थिरांक दशांश* weight, स्थिरांक दशांश* bias,
                    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    साठी (पूर्णांक b = 0; b < B; b++) {
        साठी (पूर्णांक t = 0; t < T; t++) {
            दशांश* out_bt = out + b * T * OC + t * OC;
            स्थिरांक दशांश* inp_bt = inp + b * T * C + t * C;
            साठी (पूर्णांक o = 0; o < OC; o++) {
                दशांश val = (bias != रिक्त) ? bias[o] : 0.0f;
                स्थिरांक दशांश* wrow = weight + o*C;
                साठी (पूर्णांक i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

शून्य matmul_forward_ngc92(दशांश* out,
    स्थिरांक दशांश* inp, स्थिरांक दशांश* weight, स्थिरांक दशांश* bias,
    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    जर (B * T % LOOP_UNROLL != 0) {
        म_लिखो("MUST BE A MULTIPLE OF 8"); // FIXME
        परत;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    साठी (पूर्णांक obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        साठी (पूर्णांक o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            दशांश result[LOOP_UNROLL];
            साठी (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != रिक्त) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            साठी (पूर्णांक i = 0; i < C; i++) {
                दशांश w = weight[i + o * C];
                साठी (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    पूर्णांक bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            साठी (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                पूर्णांक bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define अंक_KERNELS 2

शून्य matmul_forward(पूर्णांक kernel_अंक,
    दशांश* out,
    स्थिरांक दशांश* inp, स्थिरांक दशांश* weight, स्थिरांक दशांश* bias,
    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {

    निवड (kernel_अंक) {
        स्थिती 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            थांबा;
        स्थिती 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            थांबा;
        पूर्वनिर्धारित:
            म_लिखो("Invalid kernel number\n");
            निकास(1);
    }
}


शून्य validate_results_cpu(स्थिरांक दशांश* device_result, स्थिरांक दशांश* cpu_reference, स्थिरांक अक्षर* name, पूर्णांक अंक_elements, दशांश tolerance);
दशांश* make_random_float(माप_प्रकार N);

पूर्णांक main(पूर्णांक argc, अक्षर **argv) {
    बेक्रम(0);

    पूर्णांक B = 8;
    पूर्णांक T = 1024;
    पूर्णांक C = 768;
    पूर्णांक OC = 768 * 4; // expansion of 4, e.g. in the MLP
    पूर्णांक RUNS = 4; // number of times to run a kernel for benchmarks

    बेक्रम(137);

    दशांश* out = make_random_float(B * T * OC);
    दशांश* inp = make_random_float(B * T * C);
    दशांश* weight = make_random_float(OC * C);
    दशांश* bias = make_random_float(OC);

    दशांश* grad_out = make_random_float(B * T * OC);
    दशांश* grad_inp = make_random_float(B * T * C);
    दशांश* grad_weight = make_random_float(OC * C);
    दशांश* grad_bias = make_random_float(OC);

    म_लिखो("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    साठी (पूर्णांक kernel_अंक = 0; kernel_अंक < अंक_KERNELS; kernel_अंक++) {
        म_लिखो("> Verifying kernel #%d\n", kernel_अंक);

        बेक्रम(137);

        दशांश* kernel_out = make_random_float(B * T * OC);
        दशांश* kernel_inp = make_random_float(B * T * C);
        दशांश* kernel_weight = make_random_float(OC * C);
        दशांश* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_अंक, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        मुक्त(kernel_out);
        मुक्त(kernel_inp);
        मुक्त(kernel_weight);
        मुक्त(kernel_bias);
    }

    म_लिखो("All kernels passed! Starting benchmarks.\n\n");

    साठी (पूर्णांक kernel_अंक = 0; kernel_अंक < अंक_KERNELS; kernel_अंक++) {
        म_लिखो("> Running kernel #%d\n", kernel_अंक);
        संरचना timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        साठी (पूर्णांक i = 0; i < RUNS; i++) {
            matmul_forward(kernel_अंक, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुण time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        म_लिखो("> Kernel #%d, (took %f ms)\n", kernel_अंक, time_elapsed_s * 1000);
    }

    // free memory
    मुक्त(out);
    मुक्त(inp);
    मुक्त(weight);
    मुक्त(bias);

    मुक्त(grad_out);
    मुक्त(grad_inp);
    मुक्त(grad_weight);
    मुक्त(grad_bias);

    परत 0;
}

दशांश* make_random_float(माप_प्रकार N) {
    दशांश* arr = (दशांश*)दो_स्मृति(N * आकार(दशांश));
    साठी (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((दशांश)अक्रम() / अक्रम_उच्च) * 2.0 - 1.0; // range -1..1
    }
    परत arr;
}

शून्य validate_results_cpu(स्थिरांक दशांश* kernel_result, स्थिरांक दशांश* cpu_reference, स्थिरांक अक्षर* name, पूर्णांक अंक_elements, दशांश tolerance) {
    पूर्णांक nfaults = 0;
    साठी (पूर्णांक i = 0; i < अंक_elements; i++) {
        // print the first few comparisons
        जर (i < 5) {
            म_लिखो("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        दशांश t_eff = tolerance + भ_असल(cpu_reference[i]);
        // ensure correctness for all elements.
        जर (भ_असल(cpu_reference[i] - kernel_result[i]) > t_eff) {
            म_लिखो("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            जर (nfaults >= 10) {
                निकास(निकास_त्रुटि);
            }
        }
    }
    जर (nfaults > 0) {
        निकास(निकास_त्रुटि);
    }
    म_लिखो("OK\n");
}