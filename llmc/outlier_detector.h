/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

തരനിർവചനം ഘടന {
    ഇരട്ട buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    പൂർണ്ണസംഖ്യ എണ്ണം;
    പൂർണ്ണസംഖ്യ index;
    ഇരട്ട sum;
    ഇരട്ട sum_sq;
} OutlierDetector;

ശൂന്യം init_detector(OutlierDetector *detector) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->എണ്ണം = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

ഇരട്ട update_detector(OutlierDetector *detector, ഇരട്ട new_മൂല്യം) {

    എങ്കിൽ (detector->എണ്ണം < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->എണ്ണം] = new_മൂല്യം;
        detector->sum += new_മൂല്യം;
        detector->sum_sq += new_മൂല്യം * new_മൂല്യം;
        detector->എണ്ണം++;
        തിരികെ nan(""); // not enough data yet

    } അല്ലെങ്കിൽ {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        ഇരട്ട old_മൂല്യം = detector->buffer[detector->index];
        detector->sum -= old_മൂല്യം;
        detector->sum_sq -= old_മൂല്യം * old_മൂല്യം;
        // push the new value into the window
        detector->buffer[detector->index] = new_മൂല്യം;
        detector->sum += new_മൂല്യം;
        detector->sum_sq += new_മൂല്യം * new_മൂല്യം;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        ഇരട്ട mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        ഇരട്ട variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        ഇരട്ട std_dev = sqrt(variance);
        എങ്കിൽ (std_dev == 0.0) {
            തിരികെ 0.0;
        }
        ഇരട്ട z = (new_മൂല്യം - mean) / std_dev;

        തിരികെ z;
    }
}
