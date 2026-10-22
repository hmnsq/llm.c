/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#include <stdlib.h>
#include "../../llmc/outlier_detector.h"

số_nguyên main(void) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    cho (số_nguyên i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        double val = (double)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        double zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        nếu (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            nếu (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                trả_về EXIT_FAILURE;
            }
        } không_thì {
            // check that the zscore is within reasonable bounds
            nếu (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                trả_về EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    double outlier = 10.0; // <--- loss spike
    double zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    nếu (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        trả_về EXIT_FAILURE;
    }

    printf("OK\n");
    trả_về EXIT_SUCCESS;
}
