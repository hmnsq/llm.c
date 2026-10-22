/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ "../../llmc/outlier_detector.h"

ಪೂರ್ಣಾಂಕ ಮುಖ್ಯ(ಶೂನ್ಯ) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ * 2; i++) {
        ದ್ವಿಗುಣ val = (ದ್ವಿಗುಣ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        ದ್ವಿಗುಣ zscore = update_detector(&detector, val);

        ಬರೆ("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        ಆದರೆ (i < OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ) {
            ಆದರೆ (!isnan(zscore)) {
                ಬರೆ("Error: Expected nan, got %.4f\n", zscore);
                ಹಿಂತಿರುಗಿ EXIT_FAILURE;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // check that the zscore is within reasonable bounds
            ಆದರೆ (zscore < -3.0 || zscore > 3.0) {
                ಬರೆ("Error: Z-score %.4f is outside of expected range\n", zscore);
                ಹಿಂತಿರುಗಿ EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    ದ್ವಿಗುಣ outlier = 10.0; // <--- loss spike
    ದ್ವಿಗುಣ zscore = update_detector(&detector, outlier);
    ಬರೆ("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    ಆದರೆ (zscore < 5.0) {
        ಬರೆ("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        ಹಿಂತಿರುಗಿ EXIT_FAILURE;
    }

    ಬರೆ("OK\n");
    ಹಿಂತಿರುಗಿ EXIT_SUCCESS;
}
