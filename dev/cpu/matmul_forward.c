/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <math.h>
#समावेश <time.h>
#समावेश <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

व्योम matmul_forward_cpu(भग्न* out,
                    स्थिर भग्न* inp, स्थिर भग्न* weight, स्थिर भग्न* bias,
                    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    क्रम (पूर्णांक b = 0; b < B; b++) {
        क्रम (पूर्णांक t = 0; t < T; t++) {
            भग्न* out_bt = out + b * T * OC + t * OC;
            स्थिर भग्न* inp_bt = inp + b * T * C + t * C;
            क्रम (पूर्णांक o = 0; o < OC; o++) {
                भग्न val = (bias != शून्य) ? bias[o] : 0.0f;
                स्थिर भग्न* wrow = weight + o*C;
                क्रम (पूर्णांक i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

व्योम matmul_forward_ngc92(भग्न* out,
    स्थिर भग्न* inp, स्थिर भग्न* weight, स्थिर भग्न* bias,
    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #घोषणा LOOP_UNROLL 8

    अगर (B * T % LOOP_UNROLL != 0) {
        म_लिखो("MUST BE A MULTIPLE OF 8"); // FIXME
        वापस;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    क्रम (पूर्णांक obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        क्रम (पूर्णांक o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            भग्न परिणाम[LOOP_UNROLL];
            क्रम (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                परिणाम[ibt] = (bias != शून्य) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            क्रम (पूर्णांक i = 0; i < C; i++) {
                भग्न w = weight[i + o * C];
                क्रम (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    पूर्णांक bt = obt + ibt;
                    परिणाम[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            क्रम (पूर्णांक ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                पूर्णांक bt = obt + ibt;
                out[bt * OC + o] = परिणाम[ibt];
            }
        }
    }
}

#घोषणा अंक_KERNELS 2

व्योम matmul_forward(पूर्णांक kernel_अंक,
    भग्न* out,
    स्थिर भग्न* inp, स्थिर भग्न* weight, स्थिर भग्न* bias,
    पूर्णांक B, पूर्णांक T, पूर्णांक C, पूर्णांक OC) {

    चयन (kernel_अंक) {
        हाल 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            अवरोध;
        हाल 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            अवरोध;
        शेष:
            म_लिखो("Invalid kernel number\n");
            निकास(1);
    }
}


व्योम validate_results_cpu(स्थिर भग्न* device_परिणाम, स्थिर भग्न* cpu_reference, स्थिर अक्षर* name, पूर्णांक अंक_elements, भग्न tolerance);
भग्न* make_random_float(माप_प्रकार N);

पूर्णांक मुख्य(पूर्णांक argc, अक्षर **argv) {
    बेक्रम(0);

    पूर्णांक B = 8;
    पूर्णांक T = 1024;
    पूर्णांक C = 768;
    पूर्णांक OC = 768 * 4; // expansion of 4, e.g. in the MLP
    पूर्णांक RUNS = 4; // number of times to run a kernel for benchmarks

    बेक्रम(137);

    भग्न* out = make_random_float(B * T * OC);
    भग्न* inp = make_random_float(B * T * C);
    भग्न* weight = make_random_float(OC * C);
    भग्न* bias = make_random_float(OC);

    भग्न* grad_out = make_random_float(B * T * OC);
    भग्न* grad_inp = make_random_float(B * T * C);
    भग्न* grad_weight = make_random_float(OC * C);
    भग्न* grad_bias = make_random_float(OC);

    म_लिखो("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    क्रम (पूर्णांक kernel_अंक = 0; kernel_अंक < अंक_KERNELS; kernel_अंक++) {
        म_लिखो("> Verifying kernel #%d\n", kernel_अंक);

        बेक्रम(137);

        भग्न* kernel_out = make_random_float(B * T * OC);
        भग्न* kernel_inp = make_random_float(B * T * C);
        भग्न* kernel_weight = make_random_float(OC * C);
        भग्न* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_अंक, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        मुक्त(kernel_out);
        मुक्त(kernel_inp);
        मुक्त(kernel_weight);
        मुक्त(kernel_bias);
    }

    म_लिखो("All kernels passed! Starting benchmarks.\n\n");

    क्रम (पूर्णांक kernel_अंक = 0; kernel_अंक < अंक_KERNELS; kernel_अंक++) {
        म_लिखो("> Running kernel #%d\n", kernel_अंक);
        काष्ठा timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        क्रम (पूर्णांक i = 0; i < RUNS; i++) {
            matmul_forward(kernel_अंक, out, inp, weight, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);
        द्विगुन time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
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

    वापस 0;
}

भग्न* make_random_float(माप_प्रकार N) {
    भग्न* arr = (भग्न*)दो_स्मृति(N * माप(भग्न));
    क्रम (माप_प्रकार i = 0; i < N; i++) {
        arr[i] = ((भग्न)अक्रम() / अक्रम_उच्च) * 2.0 - 1.0; // range -1..1
    }
    वापस arr;
}

व्योम validate_results_cpu(स्थिर भग्न* kernel_परिणाम, स्थिर भग्न* cpu_reference, स्थिर अक्षर* name, पूर्णांक अंक_elements, भग्न tolerance) {
    पूर्णांक nfaults = 0;
    क्रम (पूर्णांक i = 0; i < अंक_elements; i++) {
        // print the first few comparisons
        अगर (i < 5) {
            म_लिखो("%f %f\n", cpu_reference[i], kernel_परिणाम[i]);
        }
        भग्न t_eff = tolerance + भ_असल(cpu_reference[i]);
        // ensure correctness for all elements.
        अगर (भ_असल(cpu_reference[i] - kernel_परिणाम[i]) > t_eff) {
            म_लिखो("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_परिणाम[i]);
            nfaults++;
            अगर (nfaults >= 10) {
                निकास(निकास_त्रुटि);
            }
        }
    }
    अगर (nfaults > 0) {
        निकास(निकास_त्रुटि);
    }
    म_लिखो("OK\n");
}