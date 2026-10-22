/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#अन्तर्भाव <stdio.h>
#अन्तर्भाव <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

प्रकारव्याख्या संरचना {
    द्विगुण buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    पूर्णांक मोजणी;
    पूर्णांक index;
    द्विगुण sum;
    द्विगुण sum_sq;
} OutlierDetector;

शून्य init_detector(OutlierDetector *detector) {
    साठी (पूर्णांक i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->मोजणी = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

द्विगुण update_detector(OutlierDetector *detector, द्विगुण new_मूल्य) {

    जर (detector->मोजणी < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->मोजणी] = new_मूल्य;
        detector->sum += new_मूल्य;
        detector->sum_sq += new_मूल्य * new_मूल्य;
        detector->मोजणी++;
        परत nan(""); // not enough data yet

    } अन्यथा {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        द्विगुण old_मूल्य = detector->buffer[detector->index];
        detector->sum -= old_मूल्य;
        detector->sum_sq -= old_मूल्य * old_मूल्य;
        // push the new value into the window
        detector->buffer[detector->index] = new_मूल्य;
        detector->sum += new_मूल्य;
        detector->sum_sq += new_मूल्य * new_मूल्य;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        द्विगुण mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        द्विगुण variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        द्विगुण std_dev = वर्ग_मूल(variance);
        जर (std_dev == 0.0) {
            परत 0.0;
        }
        द्विगुण z = (new_मूल्य - mean) / std_dev;

        परत z;
    }
}
