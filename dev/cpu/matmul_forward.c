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

शून्य matmul_forward_cpu(भिन्न* out,
                    अचर भिन्न* inp, अचर भिन्न* weight, अचर भिन्न* bias,
                    पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C, पूर्णाङ्क OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    लागि (पूर्णाङ्क b = 0; b < B; b++) {
        लागि (पूर्णाङ्क t = 0; t < T; t++) {
            भिन्न* out_bt = out + b * T * OC + t * OC;
            अचर भिन्न* inp_bt = inp + b * T * C + t * C;
            लागि (पूर्णाङ्क o = 0; o < OC; o++) {
                भिन्न val = (bias != रिक्त) ? bias[o] : 0.0f;
                अचर भिन्न* wrow = weight + o*C;
                लागि (पूर्णाङ्क i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

शून्य matmul_forward_ngc92(भिन्न* out,
    अचर भिन्न* inp, अचर भिन्न* weight, अचर भिन्न* bias,
    पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C, पूर्णाङ्क OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #define LOOP_UNROLL 8

    यदि (B * T % LOOP_UNROLL != 0) {
        म_लिखो("MUST BE A MULTIPLE OF 8"); // FIXME
        फर्काऊ;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    लागि (पूर्णाङ्क obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        लागि (पूर्णाङ्क o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            भिन्न result[LOOP_UNROLL];
            लागि (पूर्णाङ्क ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != रिक्त) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            लागि (पूर्णाङ्क i = 0; i < C; i++) {
                भिन्न w = weight[i + o * C];
                लागि (पूर्णाङ्क ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    पूर्णाङ्क bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            लागि (पूर्णाङ्क ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                पूर्णाङ्क bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

#define अङ्क_KERNELS 2

शून्य matmul_forward(पूर्णाङ्क kernel_अङ्क,
    भिन्न* out,
    अचर भिन्न* inp, अचर भिन्न* weight, अचर भिन्न* bias,
    पूर्णाङ्क B, पूर्णाङ्क T, पूर्णाङ्क C, पूर्णाङ्क OC) {

    विकल्प (kernel_अङ्क) {
        पक्ष 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            रोक;
        पक्ष 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            रोक;
        स्वभाव:
            म_लिखो("Invalid kernel number\n");
            निकास(1);
    }
}


शून्य validate_results_cpu(अचर भिन्न* device_result, अचर भिन्न* cpu_reference, अचर वर्ण* name, पूर्णाङ्क अङ्क_elements, भिन्न tolerance);
भिन्न* make_random_float(माप_प्रकार N);

पूर्णाङ्क main(पूर्णाङ्क argc, वर्ण **argv) {
    बेक्रम(0);

    पूर्णाङ्क B = 8;
    पूर्णाङ्क T = 1024;
    पूर्णाङ्क C = 768;
    पूर्णाङ्क OC = 768 * 4; // expansion of 4, e.g. in the MLP
    पूर्णाङ्क RUNS = 4; // number of times to run a kernel for benchmarks

    बेक्रम(137);

    भिन्न* out = make_random_float(B * T * OC);
    भिन्न* inp = make_random_float(B * T * C);
    भिन्न* weight = make_random_float(OC * C);
    भिन्न* bias = make_random_float(OC);

    भिन्न* grad_out = make_random_float(B * T * OC);
    भिन्न* grad_inp = make_random_float(B * T * C);
    भिन्न* grad_weight = make_random_float(OC * C);
    भिन्न* grad_bias = make_random_float(OC);

    म_लिखो("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    लागि (पूर्णाङ्क kernel_अङ्क = 0; kernel_अङ्क < अङ्क_KERNELS; kernel_अङ्क++) {
        म_लिखो("> Verifying kernel #%d\n", kernel_अङ्क);

        बेक्रम(137);

        भिन्न* kernel_out = make_random_float(B * T * OC);
        भिन्न* kernel_inp = make_random_float(B * T * C);
        भिन्न* kernel_weight = make_random_float(OC * C);
        भिन्न* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_अङ्क, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        मुक्त(kernel_out);
        मुक्त(kernel_inp);
        मुक्त(kernel_weight);
        मुक्त(kernel_bias);
    }

    म_लिखो("All kernels passed! Starting benchmarks.\n\n");

    लागि (पूर्णाङ्क kernel_अङ्क = 0; kernel_अङ्क < अङ्क_KERNELS; kernel_अङ्क++) {
        म_लिखो("> Running kernel #%d\n", kernel_अङ्क);
        संरचना timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        लागि (पूर्णाङ्क i = 0; i < RUNS; i++) {
            matmul_forward(kernel_अङ्क, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुण time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        म_लिखो("> Kernel #%d, (took %f ms)\n", kernel_अङ्क, time_elapsed_s * 1000);
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

    फर्काऊ 0;
}

भिन्न* make_random_float(माप_प्रकार N) {
    भिन्न* arr = (भिन्न*)दो_स्मृति(N * परिमाण(भिन्न));
    लागि (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((भिन्न)अक्रम() / अक्रम_उच्च) * 2.0 - 1.0; // range -1..1
    }
    फर्काऊ arr;
}

शून्य validate_results_cpu(अचर भिन्न* kernel_result, अचर भिन्न* cpu_reference, अचर वर्ण* name, पूर्णाङ्क अङ्क_elements, भिन्न tolerance) {
    पूर्णाङ्क nfaults = 0;
    लागि (पूर्णाङ्क i = 0; i < अङ्क_elements; i++) {
        // print the first few comparisons
        यदि (i < 5) {
            म_लिखो("%f %f\n", cpu_reference[i], kernel_result[i]);
        }
        भिन्न t_eff = tolerance + भ_असल(cpu_reference[i]);
        // ensure correctness for all elements.
        यदि (भ_असल(cpu_reference[i] - kernel_result[i]) > t_eff) {
            म_लिखो("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;
            यदि (nfaults >= 10) {
                निकास(निकास_त्रुटि);
            }
        }
    }
    यदि (nfaults > 0) {
        निकास(निकास_त्रुटि);
    }
    म_लिखो("OK\n");
}