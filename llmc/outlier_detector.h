/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

வகைவரை கட்டமைப்பு {
    இரட்டை buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    முழு மொத்தம்;
    முழு index;
    இரட்டை sum;
    இரட்டை sum_sq;
} OutlierDetector;

வெறுமை init_detector(OutlierDetector *detector) {
    க்கு (முழு i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->மொத்தம் = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

இரட்டை update_detector(OutlierDetector *detector, இரட்டை new_மதிப்பு) {

    எனில் (detector->மொத்தம் < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->மொத்தம்] = new_மதிப்பு;
        detector->sum += new_மதிப்பு;
        detector->sum_sq += new_மதிப்பு * new_மதிப்பு;
        detector->மொத்தம்++;
        திரும்பு nan(""); // not enough data yet

    } இல்லையெனில் {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        இரட்டை old_மதிப்பு = detector->buffer[detector->index];
        detector->sum -= old_மதிப்பு;
        detector->sum_sq -= old_மதிப்பு * old_மதிப்பு;
        // push the new value into the window
        detector->buffer[detector->index] = new_மதிப்பு;
        detector->sum += new_மதிப்பு;
        detector->sum_sq += new_மதிப்பு * new_மதிப்பு;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        இரட்டை mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        இரட்டை variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        இரட்டை std_dev = sqrt(variance);
        எனில் (std_dev == 0.0) {
            திரும்பு 0.0;
        }
        இரட்டை z = (new_மதிப்பு - mean) / std_dev;

        திரும்பு z;
    }
}
