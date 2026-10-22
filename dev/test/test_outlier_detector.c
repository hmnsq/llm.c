/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#অন্তর্ভাৱ <stdlib.h>
#অন্তর্ভাৱ "../../llmc/outlier_detector.h"

পূর্ণাঙ্ক main(শূন্য) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    বাবে (পূর্ণাঙ্ক i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        দ্ৱিগুণ val = (দ্ৱিগুণ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        দ্ৱিগুণ zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        যদি (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            যদি (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                ঘূৰাই EXIT_FAILURE;
            }
        } অন্যথা {
            // check that the zscore is within reasonable bounds
            যদি (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                ঘূৰাই EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    দ্ৱিগুণ outlier = 10.0; // <--- loss spike
    দ্ৱিগুণ zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    যদি (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        ঘূৰাই EXIT_FAILURE;
    }

    printf("OK\n");
    ঘূৰাই EXIT_SUCCESS;
}
