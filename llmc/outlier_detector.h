/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#অন্তর্ভাব <stdio.h>
#অন্তর্ভাব <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

প্রকারনির্দেশ সংরচনা {
    দ্বিগুণ buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    পূর্ণাঙ্ক গুনতি;
    পূর্ণাঙ্ক index;
    দ্বিগুণ sum;
    দ্বিগুণ sum_sq;
} OutlierDetector;

শূন্য init_detector(OutlierDetector *detector) {
    প্রতি (পূর্ণাঙ্ক i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->গুনতি = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

দ্বিগুণ update_detector(OutlierDetector *detector, দ্বিগুণ new_মান) {

    যদি (detector->গুনতি < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->গুনতি] = new_মান;
        detector->sum += new_মান;
        detector->sum_sq += new_মান * new_মান;
        detector->গুনতি++;
        প্রত্যাগম nan(""); // not enough data yet

    } অন্যথা {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        দ্বিগুণ old_মান = detector->buffer[detector->index];
        detector->sum -= old_মান;
        detector->sum_sq -= old_মান * old_মান;
        // push the new value into the window
        detector->buffer[detector->index] = new_মান;
        detector->sum += new_মান;
        detector->sum_sq += new_মান * new_মান;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        দ্বিগুণ mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        দ্বিগুণ variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        দ্বিগুণ std_dev = sqrt(variance);
        যদি (std_dev == 0.0) {
            প্রত্যাগম 0.0;
        }
        দ্বিগুণ z = (new_মান - mean) / std_dev;

        প্রত্যাগম z;
    }
}
