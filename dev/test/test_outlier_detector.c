/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ "../../llmc/outlier_detector.h"

முழு முதன்மை(வெறுமை) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    க்கு (முழு i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        இரட்டை val = (இரட்டை)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        இரட்டை zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        எனில் (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            எனில் (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                திரும்பு EXIT_FAILURE;
            }
        } இல்லையெனில் {
            // check that the zscore is within reasonable bounds
            எனில் (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                திரும்பு EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    இரட்டை outlier = 10.0; // <--- loss spike
    இரட்டை zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    எனில் (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        திரும்பு EXIT_FAILURE;
    }

    printf("OK\n");
    திரும்பு EXIT_SUCCESS;
}
