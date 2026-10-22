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
#تعریف_کړه OUTLIER_DETECTOR_کړکۍ_اندازه 128

تعريفنوع بنية {
    مزدوج بفر[OUTLIER_DETECTOR_کړکۍ_اندازه];
    صحيح شمېر;
    صحيح شاخص;
    مزدوج مجموعه;
    مزدوج مجموعه_sq;
} OutlierDetector;

فراغ init_detector(OutlierDetector *detector) {
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_کړکۍ_اندازه; i++) {
        detector->بفر[i] = 0.0;
    }
    detector->شمېر = 0;
    detector->شاخص = 0;
    detector->مجموعه = 0.0;
    detector->مجموعه_sq = 0.0;
}

مزدوج update_detector(OutlierDetector *detector, مزدوج new_ارزښت) {

    إذا (detector->شمېر < OUTLIER_DETECTOR_کړکۍ_اندازه) {
        // here we are still building up a window of observations
        detector->بفر[detector->شمېر] = new_ارزښت;
        detector->مجموعه += new_ارزښت;
        detector->مجموعه_sq += new_ارزښت * new_ارزښت;
        detector->شمېر++;
        أرجع nan(""); // not enough data yet

    } وإلا {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        مزدوج old_ارزښت = detector->بفر[detector->شاخص];
        detector->مجموعه -= old_ارزښت;
        detector->مجموعه_sq -= old_ارزښت * old_ارزښت;
        // push the new value into the window
        detector->بفر[detector->شاخص] = new_ارزښت;
        detector->مجموعه += new_ارزښت;
        detector->مجموعه_sq += new_ارزښت * new_ارزښت;
        // move the index to the next position
        detector->شاخص = (detector->شاخص + 1) % OUTLIER_DETECTOR_کړکۍ_اندازه;
        // calculate the z-score of the new value
        مزدوج منځنۍ = detector->مجموعه / OUTLIER_DETECTOR_کړکۍ_اندازه;
        مزدوج variance = (detector->مجموعه_sq / OUTLIER_DETECTOR_کړکۍ_اندازه) - (منځنۍ * منځنۍ);
        مزدوج std_dev = جذر(variance);
        إذا (std_dev == 0.0) {
            أرجع 0.0;
        }
        مزدوج z = (new_ارزښت - منځنۍ) / std_dev;

        أرجع z;
    }
}
