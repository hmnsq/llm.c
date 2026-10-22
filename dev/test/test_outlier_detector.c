/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#ضمن <stdlib.h>
#ضمن "../../llmc/outlier_detector.h"

صحيح اصلی(فراغ) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_پنجره_اندازه * 2; i++) {
        مزدوج val = (مزدوج)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        مزدوج zscore = update_detector(&detector, val);

        چاپ_کن("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        إذا (i < OUTLIER_DETECTOR_پنجره_اندازه) {
            إذا (!isnan(zscore)) {
                چاپ_کن("Error: Expected nan, got %.4f\n", zscore);
                أرجع EXIT_FAILURE;
            }
        } وإلا {
            // check that the zscore is within reasonable bounds
            إذا (zscore < -3.0 || zscore > 3.0) {
                چاپ_کن("Error: Z-score %.4f is outside of expected range\n", zscore);
                أرجع EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    مزدوج outlier = 10.0; // <--- loss spike
    مزدوج zscore = update_detector(&detector, outlier);
    چاپ_کن("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    إذا (zscore < 5.0) {
        چاپ_کن("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        أرجع EXIT_FAILURE;
    }

    چاپ_کن("OK\n");
    أرجع EXIT_SUCCESS;
}
