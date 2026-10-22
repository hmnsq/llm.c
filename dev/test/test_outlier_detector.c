/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#شامل <stdlib.h>
#شامل "../../llmc/outlier_detector.h"

صحیح مین(خالی) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    برائے (صحیح i = 0; i < OUTLIER_DETECTOR_کھڑکی_سائز * 2; i++) {
        دوہرا val = (دوہرا)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        دوہرا zscore = update_detector(&detector, val);

        لکھو("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        اگر (i < OUTLIER_DETECTOR_کھڑکی_سائز) {
            اگر (!isnan(zscore)) {
                لکھو("Error: Expected nan, got %.4f\n", zscore);
                واپس EXIT_FAILURE;
            }
        } ورنہ {
            // check that the zscore is within reasonable bounds
            اگر (zscore < -3.0 || zscore > 3.0) {
                لکھو("Error: Z-score %.4f is outside of expected range\n", zscore);
                واپس EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    دوہرا outlier = 10.0; // <--- loss spike
    دوہرا zscore = update_detector(&detector, outlier);
    لکھو("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    اگر (zscore < 5.0) {
        لکھو("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        واپس EXIT_FAILURE;
    }

    لکھو("OK\n");
    واپس EXIT_SUCCESS;
}
