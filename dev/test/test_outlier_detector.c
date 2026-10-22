/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#شامل <stdlib.h>
#شامل "../../llmc/outlier_detector.h"

صحیح اصلی(تهی) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    برای (صحیح i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        دوگانه val = (دوگانه)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        دوگانه zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        اگر (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            اگر (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                بازگشت EXIT_FAILURE;
            }
        } وگرنه {
            // check that the zscore is within reasonable bounds
            اگر (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                بازگشت EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    دوگانه outlier = 10.0; // <--- loss spike
    دوگانه zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    اگر (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        بازگشت EXIT_FAILURE;
    }

    printf("OK\n");
    بازگشت EXIT_SUCCESS;
}
