/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#include <stdlib.h>
#include "../../llmc/outlier_detector.h"

පූර්ණ_සංඛ්‍යා main(void) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        dviguu_na val = (dviguu_na)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        dviguu_na zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        නම් (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            නම් (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                ආපසු EXIT_FAILURE;
            }
        } නැතහොත් {
            // check that the zscore is within reasonable bounds
            නම් (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                ආපසු EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    dviguu_na outlier = 10.0; // <--- loss spike
    dviguu_na zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    නම් (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        ආපසු EXIT_FAILURE;
    }

    printf("OK\n");
    ආපසු EXIT_SUCCESS;
}
