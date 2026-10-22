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
    صحيح عدد;
    صحيح index;
    مزدوج sum;
    مزدوج sum_sq;
} OutlierDetector;

فراغ init_detector(OutlierDetector *detector) {
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->عدد = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

مزدوج update_detector(OutlierDetector *detector, مزدوج new_قيمة) {

    إذا (detector->عدد < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->عدد] = new_قيمة;
        detector->sum += new_قيمة;
        detector->sum_sq += new_قيمة * new_قيمة;
        detector->عدد++;
        أرجع nan(""); // not enough data yet

    } وإلا {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        مزدوج old_قيمة = detector->buffer[detector->index];
        detector->sum -= old_قيمة;
        detector->sum_sq -= old_قيمة * old_قيمة;
        // push the new value into the window
        detector->buffer[detector->index] = new_قيمة;
        detector->sum += new_قيمة;
        detector->sum_sq += new_قيمة * new_قيمة;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        مزدوج mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        مزدوج variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        مزدوج std_dev = sqrt(variance);
        إذا (std_dev == 0.0) {
            أرجع 0.0;
        }
        مزدوج z = (new_قيمة - mean) / std_dev;

        أرجع z;
    }
}
