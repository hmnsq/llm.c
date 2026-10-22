/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ "../../llmc/outlier_detector.h"

પૂર્ણાંક મુખ્ય(શૂન્ય) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    માટે (પૂર્ણાંક i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        દ્વિગુણ val = (દ્વિગુણ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        દ્વિગુણ zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        જો (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            જો (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                પાછા EXIT_FAILURE;
            }
        } નહીંતર {
            // check that the zscore is within reasonable bounds
            જો (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                પાછા EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    દ્વિગુણ outlier = 10.0; // <--- loss spike
    દ્વિગુણ zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    જો (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        પાછા EXIT_FAILURE;
    }

    printf("OK\n");
    પાછા EXIT_SUCCESS;
}
