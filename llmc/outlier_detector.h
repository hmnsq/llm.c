/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#define OUTLIER_DETECTOR_WINDOW_SIZE 128

ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    ଦ୍ୱିଗୁଣ buffer[OUTLIER_DETECTOR_WINDOW_SIZE];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣନା_2;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା index;
    ଦ୍ୱିଗୁଣ sum;
    ଦ୍ୱିଗୁଣ sum_sq;
} OutlierDetector;

ଶୂନ୍ୟ init_detector(OutlierDetector *detector) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < OUTLIER_DETECTOR_WINDOW_SIZE; i++) {
        detector->buffer[i] = 0.0;
    }
    detector->ଗଣନା_2 = 0;
    detector->index = 0;
    detector->sum = 0.0;
    detector->sum_sq = 0.0;
}

ଦ୍ୱିଗୁଣ update_detector(OutlierDetector *detector, ଦ୍ୱିଗୁଣ new_ମୂଲ୍ୟ) {

    ଯଦି (detector->ଗଣନା_2 < OUTLIER_DETECTOR_WINDOW_SIZE) {
        // here we are still building up a window of observations
        detector->buffer[detector->ଗଣନା_2] = new_ମୂଲ୍ୟ;
        detector->sum += new_ମୂଲ୍ୟ;
        detector->sum_sq += new_ମୂଲ୍ୟ * new_ମୂଲ୍ୟ;
        detector->ଗଣନା_2++;
        ଫେରନ୍ତୁ nan(""); // not enough data yet

    } ଅନ୍ୟଥା {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        ଦ୍ୱିଗୁଣ old_ମୂଲ୍ୟ = detector->buffer[detector->index];
        detector->sum -= old_ମୂଲ୍ୟ;
        detector->sum_sq -= old_ମୂଲ୍ୟ * old_ମୂଲ୍ୟ;
        // push the new value into the window
        detector->buffer[detector->index] = new_ମୂଲ୍ୟ;
        detector->sum += new_ମୂଲ୍ୟ;
        detector->sum_sq += new_ମୂଲ୍ୟ * new_ମୂଲ୍ୟ;
        // move the index to the next position
        detector->index = (detector->index + 1) % OUTLIER_DETECTOR_WINDOW_SIZE;
        // calculate the z-score of the new value
        ଦ୍ୱିଗୁଣ mean = detector->sum / OUTLIER_DETECTOR_WINDOW_SIZE;
        ଦ୍ୱିଗୁଣ variance = (detector->sum_sq / OUTLIER_DETECTOR_WINDOW_SIZE) - (mean * mean);
        ଦ୍ୱିଗୁଣ std_dev = sqrt(variance);
        ଯଦି (std_dev == 0.0) {
            ଫେରନ୍ତୁ 0.0;
        }
        ଦ୍ୱିଗୁଣ z = (new_ମୂଲ୍ୟ - mean) / std_dev;

        ଫେରନ୍ତୁ z;
    }
}
