/*
Tests our OutlierDetector

compile and run as (from dev/test directory)
gcc -O3 -I../../llmc -o test_outlier_detector test_outlier_detector.c -lm && ./test_outlier_detector
*/

#समावेश <stdlib.h>
#समावेश "../../llmc/outlier_detector.h"

पूर्णांक मुख्य(व्योम) {
    OutlierDetector detector;
    init_detector(&detector);

    बेक्रम(1337); // init rng

    // generate OUTLIER_DETECTOR_WINDOW_SIZE * 2 random numbers between -1 and 1
    क्रम (पूर्णांक i = 0; i < OUTLIER_DETECTOR_WINDOW_आकार * 2; i++) {
        द्विगुन val = (द्विगुन)अक्रम() / अक्रम_उच्च * 2 - 1;  // Random number between -1 and 1
        द्विगुन zscore = update_detector(&detector, val);

        म_लिखो("Step %d: Value = %.4f, zscore = %.4f\n", i, val, zscore);

        // check that the first OUTLIER_DETECTOR_WINDOW_SIZE values return nan
        अगर (i < OUTLIER_DETECTOR_WINDOW_आकार) {
            अगर (!isnan(zscore)) {
                म_लिखो("Error: Expected nan, got %.4f\n", zscore);
                वापस निकास_त्रुटि;
            }
        } अन्यथा {
            // check that the zscore is within reasonable bounds
            अगर (zscore < -3.0 || zscore > 3.0) {
                म_लिखो("Error: Z-score %.4f is outside of expected range\n", zscore);
                वापस निकास_त्रुटि;
            }
        }
    }

    // simulate an outlier
    द्विगुन outlier = 10.0; // <--- loss spike
    द्विगुन zscore = update_detector(&detector, outlier);
    म_लिखो("Outlier Step: Value = %.4f, zscore = %.4f\n", outlier, zscore);

    // check that the z-score here is large
    अगर (zscore < 5.0) {
        म_लिखो("Error: Z-score %.4f is not large enough for an outlier\n", zscore);
        वापस निकास_त्रुटि;
    }

    म_लिखो("OK\n");
    वापस निकास_सफल;
}
