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
#وصف_ڪريو OUTLIER_DETECTOR_دري_سائيز 128

تعريفنوع بنية {
    مزدوج بفر[OUTLIER_DETECTOR_دري_سائيز];
    صحيح ڳڻپ;
    صحيح اشاريو;
    مزدوج جوڙ;
    مزدوج جوڙ_sq;
} OutlierDetector;

فراغ init_detector(OutlierDetector *detector) {
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_دري_سائيز; i++) {
        detector->بفر[i] = 0.0;
    }
    detector->ڳڻپ = 0;
    detector->اشاريو = 0;
    detector->جوڙ = 0.0;
    detector->جوڙ_sq = 0.0;
}

مزدوج update_detector(OutlierDetector *detector, مزدوج new_قدر) {

    إذا (detector->ڳڻپ < OUTLIER_DETECTOR_دري_سائيز) {
        // here we are still building up a window of observations
        detector->بفر[detector->ڳڻپ] = new_قدر;
        detector->جوڙ += new_قدر;
        detector->جوڙ_sq += new_قدر * new_قدر;
        detector->ڳڻپ++;
        أرجع nan(""); // not enough data yet

    } وإلا {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        مزدوج old_قدر = detector->بفر[detector->اشاريو];
        detector->جوڙ -= old_قدر;
        detector->جوڙ_sq -= old_قدر * old_قدر;
        // push the new value into the window
        detector->بفر[detector->اشاريو] = new_قدر;
        detector->جوڙ += new_قدر;
        detector->جوڙ_sq += new_قدر * new_قدر;
        // move the index to the next position
        detector->اشاريو = (detector->اشاريو + 1) % OUTLIER_DETECTOR_دري_سائيز;
        // calculate the z-score of the new value
        مزدوج سراسري = detector->جوڙ / OUTLIER_DETECTOR_دري_سائيز;
        مزدوج variance = (detector->جوڙ_sq / OUTLIER_DETECTOR_دري_سائيز) - (سراسري * سراسري);
        مزدوج std_dev = جذر(variance);
        إذا (std_dev == 0.0) {
            أرجع 0.0;
        }
        مزدوج z = (new_قدر - سراسري) / std_dev;

        أرجع z;
    }
}
