/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ "../../llmc/outlier_detector.h"

ಪೂರ್ಣಾಙ್ಕ main(ಶೂನ್ಯ) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        ದ್ವಿಗುಣ val = (ದ್ವಿಗುಣ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        ದ್ವಿಗುಣ zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        ಯದಿ (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            ಯದಿ (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                ಪ್ರತ್ಯಾಗಮ EXIT_FAILURE;
            }
        } ಅನ್ಯಥಾ {
            // check that the zscore is within reasonable bounds
            ಯದಿ (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                ಪ್ರತ್ಯಾಗಮ EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    ದ್ವಿಗುಣ outlier = 10.0; // <--- loss spike
    ದ್ವಿಗುಣ zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    ಯದಿ (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        ಪ್ರತ್ಯಾಗಮ EXIT_FAILURE;
    }

    printf("OK\n");
    ಪ್ರತ್ಯಾಗಮ EXIT_SUCCESS;
}
