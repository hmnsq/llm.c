/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#అన్తర్భావ <stdlib.h>
#అన్తర్భావ "../../llmc/outlier_detector.h"

పూర్ణాంకం ప్రధాన(శూన్య) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    కొరకు (పూర్ణాంకం i = 0; i < OUTLIER_DETECTOR_కిటికీ_కొలత * 2; i++) {
        ద్విగుణ val = (ద్విగుణ)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        ద్విగుణ zscore = update_detector(&detector, val);

        రాయి("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        ఒకవేళ (i < OUTLIER_DETECTOR_కిటికీ_కొలత) {
            ఒకవేళ (!isnan(zscore)) {
                రాయి("Error: Expected nan, got %.4f\n", zscore);
                తిరిగి EXIT_FAILURE;
            }
        } లేకపోతే {
            // check that the zscore is within reasonable bounds
            ఒకవేళ (zscore < -3.0 || zscore > 3.0) {
                రాయి("Error: Z-score %.4f is outside of expected range\n", zscore);
                తిరిగి EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    ద్విగుణ outlier = 10.0; // <--- loss spike
    ద్విగుణ zscore = update_detector(&detector, outlier);
    రాయి("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    ఒకవేళ (zscore < 5.0) {
        రాయి("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        తిరిగి EXIT_FAILURE;
    }

    రాయి("OK\n");
    తిరిగి EXIT_SUCCESS;
}
