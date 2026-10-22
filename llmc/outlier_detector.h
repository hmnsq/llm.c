/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#ضمن <stdio.h>
#ضمن <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#تعریف_کن OUTLIER_DETECTOR_پنجره_اندازه 128

تعريفنوع بنية {
    مزدوج میانگیر[OUTLIER_DETECTOR_پنجره_اندازه];
    صحيح شمار;
    صحيح شاخص;
    مزدوج جمع;
    مزدوج جمع_sq;
} OutlierDetector;

فراغ init_detector(OutlierDetector *detector) {
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_پنجره_اندازه; i++) {
        detector->میانگیر[i] = 0.0;
    }
    detector->شمار = 0;
    detector->شاخص = 0;
    detector->جمع = 0.0;
    detector->جمع_sq = 0.0;
}

مزدوج update_detector(OutlierDetector *detector, مزدوج new_مقدار) {

    إذا (detector->شمار < OUTLIER_DETECTOR_پنجره_اندازه) {
        // here we are still building up a window of observations
        detector->میانگیر[detector->شمار] = new_مقدار;
        detector->جمع += new_مقدار;
        detector->جمع_sq += new_مقدار * new_مقدار;
        detector->شمار++;
        أرجع nan(""); // not enough data yet

    } وإلا {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        مزدوج old_مقدار = detector->میانگیر[detector->شاخص];
        detector->جمع -= old_مقدار;
        detector->جمع_sq -= old_مقدار * old_مقدار;
        // push the new value into the window
        detector->میانگیر[detector->شاخص] = new_مقدار;
        detector->جمع += new_مقدار;
        detector->جمع_sq += new_مقدار * new_مقدار;
        // move the index to the next position
        detector->شاخص = (detector->شاخص + 1) % OUTLIER_DETECTOR_پنجره_اندازه;
        // calculate the z-score of the new value
        مزدوج میانگین = detector->جمع / OUTLIER_DETECTOR_پنجره_اندازه;
        مزدوج variance = (detector->جمع_sq / OUTLIER_DETECTOR_پنجره_اندازه) - (میانگین * میانگین);
        مزدوج std_dev = جذر(variance);
        إذا (std_dev == 0.0) {
            أرجع 0.0;
        }
        مزدوج z = (new_مقدار - میانگین) / std_dev;

        أرجع z;
    }
}
