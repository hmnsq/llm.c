/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#شامل <stdio.h>
#شامل <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

تعریفنوع ساختار {
    دوگانه buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    صحیح شمار;
    صحیح index;
    دوگانه sum;
    دوگانه sum_sq;
} OutlierDetector;

تهی init_detector(OutlierDetector *detector) {
    برای (صحیح i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->شمار = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

دوگانه update_detector(OutlierDetector *detector, دوگانه new_مقدار) {

    اگر (detector->شمار < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->شمار] = new_مقدار;
        detector->sum += new_مقدار;
        detector->sum_sq += new_مقدار * new_مقدار;
        detector->شمار++;
        بازگشت nan(""); // not enough data yet

    } وگرنه {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        دوگانه old_مقدار = detector->buffer[detector->index];
        detector->sum -= old_مقدار;
        detector->sum_sq -= old_مقدار * old_مقدار;
        // push the new value into the window
        detector->buffer[detector->index] = new_مقدار;
        detector->sum += new_مقدار;
        detector->sum_sq += new_مقدار * new_مقدار;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        دوگانه mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        دوگانه variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        دوگانه std_dev = sqrt(variance);
        اگر (std_dev == 0.0) {
            بازگشت 0.0;
        }
        دوگانه z = (new_مقدار - mean) / std_dev;

        بازگشت z;
    }
}
