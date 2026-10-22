/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#समावेश <stdio.h>
#समावेश <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#घोषणा OUTLIER_DETECTOR_WINDOW_आकार 128

प्रकार काष्ठा {
    द्विगुन बफर[OUTLIER_DETECTOR_WINDOW_आकार];
    पूर्णांक गिनती;
    पूर्णांक सूचकांक_2;
    द्विगुन योग_2;
    द्विगुन योग_sq;
} OutlierDetector;

व्योम init_detector(OutlierDetector *detector) {
    क्रम (पूर्णांक i = 0; i < OUTLIER_DETECTOR_WINDOW_आकार; i++) {
        detector->बफर[i] = 0.0;
    }
    detector->गिनती = 0;
    detector->सूचकांक_2 = 0;
    detector->योग_2 = 0.0;
    detector->योग_sq = 0.0;
}

द्विगुन update_detector(OutlierDetector *detector, द्विगुन new_मूल्य) {

    अगर (detector->गिनती < OUTLIER_DETECTOR_WINDOW_आकार) {
        // here we are still building up a window of observations
        detector->बफर[detector->गिनती] = new_मूल्य;
        detector->योग_2 += new_मूल्य;
        detector->योग_sq += new_मूल्य * new_मूल्य;
        detector->गिनती++;
        वापस nan(""); // not enough data yet

    } अन्यथा {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        द्विगुन old_मूल्य = detector->बफर[detector->सूचकांक_2];
        detector->योग_2 -= old_मूल्य;
        detector->योग_sq -= old_मूल्य * old_मूल्य;
        // push the new value into the window
        detector->बफर[detector->सूचकांक_2] = new_मूल्य;
        detector->योग_2 += new_मूल्य;
        detector->योग_sq += new_मूल्य * new_मूल्य;
        // move the index to the next position
        detector->सूचकांक_2 = (detector->सूचकांक_2 + 1) % OUTLIER_DETECTOR_WINDOW_आकार;
        // calculate the z-score of the new value
        द्विगुन mean = detector->योग_2 / OUTLIER_DETECTOR_WINDOW_आकार;
        द्विगुन variance = (detector->योग_sq / OUTLIER_DETECTOR_WINDOW_आकार) - (mean * mean);
        द्विगुन std_dev = वर्ग_मूल(variance);
        अगर (std_dev == 0.0) {
            वापस 0.0;
        }
        द्विगुन z = (new_मूल्य - mean) / std_dev;

        वापस z;
    }
}
