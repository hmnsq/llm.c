/*
CPU Kernels for matmul forward pass.
*/

// Compile Examples:
//
//      MSVC: cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
//            cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
//

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <time.h>
#అన్తర్భావ <unistd.h>

// ----------------------------------------------------------------------------
// CPU code reference

శూన్య matmul_forward_cpu(దశాంశం* బహిర్గతం,
                    స్థిరరాశి దశాంశం* inp, స్థిరరాశి దశాంశం* బరువు, స్థిరరాశి దశాంశం* bias,
                    పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    కొరకు (పూర్ణాంకం b = 0; b < B; b++) {
        కొరకు (పూర్ణాంకం t = 0; t < T; t++) {
            దశాంశం* బహిర్గతం_bt = బహిర్గతం + b * T * OC + t * OC;
            స్థిరరాశి దశాంశం* inp_bt = inp + b * T * C + t * C;
            కొరకు (పూర్ణాంకం o = 0; o < OC; o++) {
                దశాంశం val = (bias != ఖాళీ) ? bias[o] : 0.0f;
                స్థిరరాశి దశాంశం* wrow = బరువు + o*C;
                కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                బహిర్గతం_bt[o] = val;
            }
        }
    }
}

శూన్య matmul_forward_ngc92(దశాంశం* బహిర్గతం,
    స్థిరరాశి దశాంశం* inp, స్థిరరాశి దశాంశం* బరువు, స్థిరరాశి దశాంశం* bias,
    పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct, otherwise, fallback to slow version
    #నిర్వచించు LOOP_UNROLL 8

    ఒకవేళ (B * T % LOOP_UNROLL != 0) {
        రాయి("MUST BE A MULTIPLE OF 8"); // FIXME
        తిరిగి;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // for significant speed-ups.
    కొరకు (పూర్ణాంకం obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        కొరకు (పూర్ణాంకం o = 0; o < OC; o++) {
            // keep LOOP_UNROLL many results in register, initialized by the bias term.
            దశాంశం ఫలితం[LOOP_UNROLL];
            కొరకు (పూర్ణాంకం ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                ఫలితం[ibt] = (bias != ఖాళీ) ? bias[o] : 0.0f;
            }

            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into a bunch of FMAs
            కొరకు (పూర్ణాంకం i = 0; i < C; i++) {
                దశాంశం w = బరువు[i + o * C];
                కొరకు (పూర్ణాంకం ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    పూర్ణాంకం bt = obt + ibt;
                    ఫలితం[ibt] += inp[bt * C + i] * w;
                }
            }

            // write back results to main memory
            కొరకు (పూర్ణాంకం ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                పూర్ణాంకం bt = obt + ibt;
                బహిర్గతం[bt * OC + o] = ఫలితం[ibt];
            }
        }
    }
}

#నిర్వచించు అంకె_KERNELS 2

శూన్య matmul_forward(పూర్ణాంకం kernel_అంకె,
    దశాంశం* బహిర్గతం,
    స్థిరరాశి దశాంశం* inp, స్థిరరాశి దశాంశం* బరువు, స్థిరరాశి దశాంశం* bias,
    పూర్ణాంకం B, పూర్ణాంకం T, పూర్ణాంకం C, పూర్ణాంకం OC) {

    ఎంపిక (kernel_అంకె) {
        స్థితి 0:
            matmul_forward_cpu(బహిర్గతం, inp, బరువు, bias, B, T, C, OC);
            ఆపు;
        స్థితి 1:
            matmul_forward_ngc92(బహిర్గతం, inp, బరువు, bias, B, T, C, OC);
            ఆపు;
        అప్రమేయ:
            రాయి("Invalid kernel number\n");
            నిష్క్రమించు(1);
    }
}


శూన్య validate_results_cpu(స్థిరరాశి దశాంశం* device_ఫలితం, స్థిరరాశి దశాంశం* cpu_reference, స్థిరరాశి అక్షరం* name, పూర్ణాంకం అంకె_elements, దశాంశం tolerance);
దశాంశం* make_random_float(size_t N);

పూర్ణాంకం ప్రధాన(పూర్ణాంకం argc, అక్షరం **argv) {
    srand(0);

    పూర్ణాంకం B = 8;
    పూర్ణాంకం T = 1024;
    పూర్ణాంకం C = 768;
    పూర్ణాంకం OC = 768 * 4; // expansion of 4, e.g. in the MLP
    పూర్ణాంకం RUNS = 4; // number of times to run a kernel for benchmarks

    srand(137);

    దశాంశం* బహిర్గతం = make_random_float(B * T * OC);
    దశాంశం* inp = make_random_float(B * T * C);
    దశాంశం* బరువు = make_random_float(OC * C);
    దశాంశం* bias = make_random_float(OC);

    దశాంశం* grad_బహిర్గతం = make_random_float(B * T * OC);
    దశాంశం* grad_inp = make_random_float(B * T * C);
    దశాంశం* grad_బరువు = make_random_float(OC * C);
    దశాంశం* grad_bias = make_random_float(OC);

    రాయి("> Calculating reference\n");
    matmul_forward_cpu(బహిర్గతం, inp, బరువు, bias, B, T, C, OC);

    కొరకు (పూర్ణాంకం kernel_అంకె = 0; kernel_అంకె < అంకె_KERNELS; kernel_అంకె++) {
        రాయి("> Verifying kernel #%d\n", kernel_అంకె);

        srand(137);

        దశాంశం* kernel_బహిర్గతం = make_random_float(B * T * OC);
        దశాంశం* kernel_inp = make_random_float(B * T * C);
        దశాంశం* kernel_బరువు = make_random_float(OC * C);
        దశాంశం* kernel_bias = make_random_float(OC);

        matmul_forward(kernel_అంకె, kernel_బహిర్గతం, kernel_inp, kernel_బరువు, kernel_bias, B, T, C, OC);

        validate_results_cpu(kernel_బహిర్గతం, బహిర్గతం, "out", B * T * OC, 1e-5);

        విడుదల(kernel_బహిర్గతం);
        విడుదల(kernel_inp);
        విడుదల(kernel_బరువు);
        విడుదల(kernel_bias);
    }

    రాయి("All kernels passed! Starting benchmarks.\n\n");

    కొరకు (పూర్ణాంకం kernel_అంకె = 0; kernel_అంకె < అంకె_KERNELS; kernel_అంకె++) {
        రాయి("> Running kernel #%d\n", kernel_అంకె);
        నిర్మాణం timespec ప్రారంభం, చివర;
        clock_gettime(CLOCK_MONOTONIC, &ప్రారంభం);

        కొరకు (పూర్ణాంకం i = 0; i < RUNS; i++) {
            matmul_forward(kernel_అంకె, బహిర్గతం, inp, బరువు, bias, B, T, C, OC);
        }

        clock_gettime(CLOCK_MONOTONIC, &చివర);
        ద్విగుణ సమయం_elapsed_s = (చివర.tv_sec - ప్రారంభం.tv_sec) + (చివర.tv_nsec - ప్రారంభం.tv_nsec) / 1e9;
        రాయి("> Kernel #%d, (took %f ms)\n", kernel_అంకె, సమయం_elapsed_s * 1000);
    }

    // free memory
    విడుదల(బహిర్గతం);
    విడుదల(inp);
    విడుదల(బరువు);
    విడుదల(bias);

    విడుదల(grad_బహిర్గతం);
    విడుదల(grad_inp);
    విడుదల(grad_బరువు);
    విడుదల(grad_bias);

    తిరిగి 0;
}

దశాంశం* make_random_float(size_t N) {
    దశాంశం* arr = (దశాంశం*)జ్ఞాపకం_ఇవ్వు(N * పరిమాణం(దశాంశం));
    కొరకు (size_t i = 0; i < N; i++) {
        arr[i] = ((దశాంశం)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    తిరిగి arr;
}

శూన్య validate_results_cpu(స్థిరరాశి దశాంశం* kernel_ఫలితం, స్థిరరాశి దశాంశం* cpu_reference, స్థిరరాశి అక్షరం* name, పూర్ణాంకం అంకె_elements, దశాంశం tolerance) {
    పూర్ణాంకం nfaults = 0;
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_elements; i++) {
        // print the first few comparisons
        ఒకవేళ (i < 5) {
            రాయి("%f %f\n", cpu_reference[i], kernel_ఫలితం[i]);
        }
        దశాంశం t_eff = tolerance + fabs(cpu_reference[i]);
        // ensure correctness for all elements.
        ఒకవేళ (fabs(cpu_reference[i] - kernel_ఫలితం[i]) > t_eff) {
            రాయి("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n", name, i, cpu_reference[i], kernel_ఫలితం[i]);
            nfaults++;
            ఒకవేళ (nfaults >= 10) {
                నిష్క్రమించు(EXIT_FAILURE);
            }
        }
    }
    ఒకవేళ (nfaults > 0) {
        నిష్క్రమించు(EXIT_FAILURE);
    }
    రాయి("OK\n");
}