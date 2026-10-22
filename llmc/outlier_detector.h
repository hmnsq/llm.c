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

प्रकारनिर्देश संरचना {
    द्विगुण buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    पूर्णाङ्क गन्ती;
    पूर्णाङ्क index;
    द्विगुण sum;
    द्विगुण sum_sq;
} OutlierDetector;

शून्य init_detector(OutlierDetector *detector) {
    लागि (पूर्णाङ्क i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->गन्ती = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

द्विगुण update_detector(OutlierDetector *detector, द्विगुण new_मान) {

    यदि (detector->गन्ती < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->गन्ती] = new_मान;
        detector->sum += new_मान;
        detector->sum_sq += new_मान * new_मान;
        detector->गन्ती++;
        फर्काऊ nan(""); // not enough data yet

    } अन्यथा {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        द्विगुण old_मान = detector->buffer[detector->index];
        detector->sum -= old_मान;
        detector->sum_sq -= old_मान * old_मान;
        // push the new value into the window
        detector->buffer[detector->index] = new_मान;
        detector->sum += new_मान;
        detector->sum_sq += new_मान * new_मान;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        द्विगुण mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        द्विगुण variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        द्विगुण std_dev = वर्ग_मूल(variance);
        यदि (std_dev == 0.0) {
            फर्काऊ 0.0;
        }
        द्विगुण z = (new_मान - mean) / std_dev;

        फर्काऊ z;
    }
}
