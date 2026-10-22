/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#ضمن <stdio.h>
#ضمن <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

تعريفنوع بنية {
    مزدوج buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    صحيح count;
    صحيح index;
    مزدوج sum;
    مزدوج sum_sq;
} OutlierDetector;

فراغ init_detector(OutlierDetector *detector) {
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->count = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

مزدوج update_detector(OutlierDetector *detector, مزدوج new_value) {

    𐡄𐡍 (detector->count < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->count] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        detector->count++;
        أرجع nan(""); // not enough data yet

    } وإلا {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        مزدوج old_value = detector->buffer[detector->index];
        detector->sum -= old_value;
        detector->sum_sq -= old_value * old_value;
        // push the new value into the window
        detector->buffer[detector->index] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        مزدوج mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        مزدوج variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        مزدوج std_dev = sqrt(variance);
        𐡄𐡍 (std_dev == 0.0) {
            أرجع 0.0;
        }
        مزدوج z = (new_value - mean) / std_dev;

        أرجع z;
    }
}
