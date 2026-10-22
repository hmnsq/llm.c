/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#अन्तर्भाव <stdlib.h>
#अन्तर्भाव "../../llmc/outlier_detector.h"

पूर्णाङ्क main(शून्य) {
    OutlierDetector detector;
    init_detector(&detector);

    srand(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    प्रति (पूर्णाङ्क i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE * 2; i++) {
        द्विगुण val = (द्विगुण)rand() / RAND_MAX * 2 - 1;  // Random number between -1 and 1
        द्विगुण zscore = update_detector(&detector, val);

        printf("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        यदि (i < OUTLIER_DETECTOR_WINDOW_SIZE) {
            यदि (!isnan(zscore)) {
                printf("Error: Expected nan, got %.4f\n", zscore);
                प्रत्यागम EXIT_FAILURE;
            }
        } अन्यथा {
            // check that the zscore is within reasonable bounds
            यदि (zscore < -3.0 || zscore > 3.0) {
                printf("Error: Z-score %.4f is outside of expected range\n", zscore);
                प्रत्यागम EXIT_FAILURE;
            }
        }
    }

    // simulate an outlier
    द्विगुण outlier = 10.0; // <--- loss spike
    द्विगुण zscore = update_detector(&detector, outlier);
    printf("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    यदि (zscore < 5.0) {
        printf("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        प्रत्यागम EXIT_FAILURE;
    }

    printf("OK\n");
    प्रत्यागम EXIT_SUCCESS;
}
