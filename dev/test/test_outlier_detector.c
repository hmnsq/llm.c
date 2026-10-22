/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "../../llmc/outlier_detector.h"

ਪੂਰਨ ਮੁੱਖ(ਖਾਲੀ) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    ਲਈ (ਪੂਰਨ i = 0; i < OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ * 2; i++) {
        ਦੁੱਗਣਾ val = (ਦੁੱਗਣਾ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        ਦੁੱਗਣਾ zscore = update_detector(&detector, val);

        ਲਿਖੋ("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        ਜੇ (i < OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ) {
            ਜੇ (!isnan(zscore)) {
                ਲਿਖੋ("Error: Expected nan, got %.4f\n", zscore);
                ਵਾਪਸ EXIT_FAILURE;
            }
        } ਨਹੀਂਤਾਂ {
            // check that the zscore is within reasonable bounds
            ਜੇ (zscore < -3.0 || zscore > 3.0) {
                ਲਿਖੋ("Error: Z-score %.4f is outside of expected range\n", zscore);
                ਵਾਪਸ EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    ਦੁੱਗਣਾ outlier = 10.0; // <--- loss spike
    ਦੁੱਗਣਾ zscore = update_detector(&detector, outlier);
    ਲਿਖੋ("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    ਜੇ (zscore < 5.0) {
        ਲਿਖੋ("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        ਵਾਪਸ EXIT_FAILURE;
    }

    ਲਿਖੋ("OK\n");
    ਵਾਪਸ EXIT_SUCCESS;
}
