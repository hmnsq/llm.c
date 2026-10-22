/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ "../../llmc/outlier_detector.h"

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ମୁଖ୍ୟ(ଶୂନ୍ୟ) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        ଦ୍ୱିଗୁଣ val = (ଦ୍ୱିଗୁଣ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        ଦ୍ୱିଗୁଣ zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        ଯଦି (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            ଯଦି (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                ଫେରନ୍ତୁ EXIT_FAILURE;
            }
        } ଅନ୍ୟଥା {
            // check that the zscore is within reasonable bounds
            ଯଦି (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                ଫେରନ୍ତୁ EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    ଦ୍ୱିଗୁଣ outlier = 10.0; // <--- loss spike
    ଦ୍ୱିଗୁଣ zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    ଯଦି (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        ଫେରନ୍ତୁ EXIT_FAILURE;
    }

    printf("OK\n");
    ଫେରନ୍ତୁ EXIT_SUCCESS;
}
