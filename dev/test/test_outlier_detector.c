/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ "../../llmc/outlier_detector.h"

പൂർണ്ണസംഖ്യ പ്രധാന(ശൂന്യം) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        ഇരട്ട val = (ഇരട്ട)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        ഇരട്ട zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        എങ്കിൽ (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            എങ്കിൽ (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                തിരികെ EXIT_FAILURE;
            }
        } അല്ലെങ്കിൽ {
            // check that the zscore is within reasonable bounds
            എങ്കിൽ (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                തിരികെ EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    ഇരട്ട outlier = 10.0; // <--- loss spike
    ഇരട്ട zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    എങ്കിൽ (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        തിരികെ EXIT_FAILURE;
    }

    printf("OK\n");
    തിരികെ EXIT_SUCCESS;
}
