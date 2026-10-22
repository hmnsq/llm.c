/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

પ્રકારવ્યાખ્યા સંરચના {
    દ્વિગુણ buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    પૂર્ણાંક ગણતરી;
    પૂર્ણાંક index;
    દ્વિગુણ sum;
    દ્વિગુણ sum_sq;
} OutlierDetector;

શૂન્ય init_detector(OutlierDetector *detector) {
    માટે (પૂર્ણાંક i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->ગણતરી = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

દ્વિગુણ update_detector(OutlierDetector *detector, દ્વિગુણ new_કિંમત) {

    જો (detector->ગણતરી < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->ગણતરી] = new_કિંમત;
        detector->sum += new_કિંમત;
        detector->sum_sq += new_કિંમત * new_કિંમત;
        detector->ગણતરી++;
        પાછા nan(""); // not enough data yet

    } નહીંતર {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        દ્વિગુણ old_કિંમત = detector->buffer[detector->index];
        detector->sum -= old_કિંમત;
        detector->sum_sq -= old_કિંમત * old_કિંમત;
        // push the new value into the window
        detector->buffer[detector->index] = new_કિંમત;
        detector->sum += new_કિંમત;
        detector->sum_sq += new_કિંમત * new_કિંમત;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        દ્વિગુણ mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        દ્વિગુણ variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        દ્વિગુણ std_dev = sqrt(variance);
        જો (std_dev == 0.0) {
            પાછા 0.0;
        }
        દ્વિગુણ z = (new_કિંમત - mean) / std_dev;

        પાછા z;
    }
}
