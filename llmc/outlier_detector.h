/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#include <stdio.h>
#include <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

typedef sa_nracanaa {
    dviguu_na buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    puur_naa_nka count;
    puur_naa_nka index;
    dviguu_na sum;
    dviguu_na sum_sq;
} OutlierDetector;

void init_detector(OutlierDetector *detector) {
    for (puur_naa_nka i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->count = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

dviguu_na update_detector(OutlierDetector *detector, dviguu_na new_value) {

    ᤀᤡ᤺ (detector->count < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->count] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        detector->count++;
        ᤀᤡᤴ nan(""); // not enough data yet

    } ᤀᤡ᤺ᤆᤡ {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        dviguu_na old_value = detector->buffer[detector->index];
        detector->sum -= old_value;
        detector->sum_sq -= old_value * old_value;
        // push the new value into the window
        detector->buffer[detector->index] = new_value;
        detector->sum += new_value;
        detector->sum_sq += new_value * new_value;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        dviguu_na mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        dviguu_na variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        dviguu_na std_dev = sqrt(variance);
        ᤀᤡ᤺ (std_dev == 0.0) {
            ᤀᤡᤴ 0.0;
        }
        dviguu_na z = (new_value - mean) / std_dev;

        ᤀᤡᤴ z;
    }
}
