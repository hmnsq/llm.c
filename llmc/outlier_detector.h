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
#define OUTLIER_DETECTOR_WINDOW_আকার 128

প্রকারলেখ কাঠামো {
    দ্বিগুণ বাফার[OUTLIER_DETECTOR_WINDOW_আকার];
    পূর্ণসংখ্যা গুনতি;
    পূর্ণসংখ্যা সূচক;
    দ্বিগুণ যোগ_2;
    দ্বিগুণ যোগ_sq;
} OutlierDetector;

শূন্য init_detector(OutlierDetector *detector) {
    জন্য (পূর্ণসংখ্যা i = 0; i < OUTLIER_DETECTOR_WINDOW_আকার; i++) {
        detector->বাফার[i] = 0.0;
    }
    detector->গুনতি = 0;
    detector->সূচক = 0;
    detector->যোগ_2 = 0.0;
    detector->যোগ_sq = 0.0;
}

দ্বিগুণ update_detector(OutlierDetector *detector, দ্বিগুণ new_মান) {

    যদি (detector->গুনতি < OUTLIER_DETECTOR_WINDOW_আকার) {
        // here we are still building up a window of observations
        detector->বাফার[detector->গুনতি] = new_মান;
        detector->যোগ_2 += new_মান;
        detector->যোগ_sq += new_মান * new_মান;
        detector->গুনতি++;
        ফেরত nan(""); // not enough data yet

    } অন্যথা {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        দ্বিগুণ old_মান = detector->বাফার[detector->সূচক];
        detector->যোগ_2 -= old_মান;
        detector->যোগ_sq -= old_মান * old_মান;
        // push the new value into the window
        detector->বাফার[detector->সূচক] = new_মান;
        detector->যোগ_2 += new_মান;
        detector->যোগ_sq += new_মান * new_মান;
        // move the index to the next position
        detector->সূচক = (detector->সূচক + 1) % OUTLIER_DETECTOR_WINDOW_আকার;
        // calculate the z-score of the new value
        দ্বিগুণ mean = detector->যোগ_2 / OUTLIER_DETECTOR_WINDOW_আকার;
        দ্বিগুণ variance = (detector->যোগ_sq / OUTLIER_DETECTOR_WINDOW_আকার) - (mean * mean);
        দ্বিগুণ std_dev = sqrt(variance);
        যদি (std_dev == 0.0) {
            ফেরত 0.0;
        }
        দ্বিগুণ z = (new_মান - mean) / std_dev;

        ফেরত z;
    }
}
