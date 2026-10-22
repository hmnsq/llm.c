/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

ಪ್ರಕಾರನಿರ್ದೇಶ ಸಂರಚನಾ {
    ದ್ವಿಗುಣ buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    ಪೂರ್ಣಾಙ್ಕ ಎಣಿಕೆ;
    ಪೂರ್ಣಾಙ್ಕ index;
    ದ್ವಿಗುಣ sum;
    ದ್ವಿಗುಣ sum_sq;
} OutlierDetector;

ಶೂನ್ಯ init_detector(OutlierDetector *detector) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->ಎಣಿಕೆ = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

ದ್ವಿಗುಣ update_detector(OutlierDetector *detector, ದ್ವಿಗುಣ new_ಮೌಲ್ಯ) {

    ಯದಿ (detector->ಎಣಿಕೆ < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->ಎಣಿಕೆ] = new_ಮೌಲ್ಯ;
        detector->sum += new_ಮೌಲ್ಯ;
        detector->sum_sq += new_ಮೌಲ್ಯ * new_ಮೌಲ್ಯ;
        detector->ಎಣಿಕೆ++;
        ಪ್ರತ್ಯಾಗಮ nan(""); // not enough data yet

    } ಅನ್ಯಥಾ {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        ದ್ವಿಗುಣ old_ಮೌಲ್ಯ = detector->buffer[detector->index];
        detector->sum -= old_ಮೌಲ್ಯ;
        detector->sum_sq -= old_ಮೌಲ್ಯ * old_ಮೌಲ್ಯ;
        // push the new value into the window
        detector->buffer[detector->index] = new_ಮೌಲ್ಯ;
        detector->sum += new_ಮೌಲ್ಯ;
        detector->sum_sq += new_ಮೌಲ್ಯ * new_ಮೌಲ್ಯ;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        ದ್ವಿಗುಣ mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        ದ್ವಿಗುಣ variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        ದ್ವಿಗುಣ std_dev = sqrt(variance);
        ಯದಿ (std_dev == 0.0) {
            ಪ್ರತ್ಯಾಗಮ 0.0;
        }
        ದ್ವಿಗುಣ z = (new_ಮೌಲ್ಯ - mean) / std_dev;

        ಪ್ರತ್ಯಾಗಮ z;
    }
}
