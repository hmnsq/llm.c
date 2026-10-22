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
#تعریف_کرو OUTLIER_DETECTOR_کھڑکی_ناپ 128

تعريفنوع بنية {
    مزدوج بفر[OUTLIER_DETECTOR_کھڑکی_ناپ];
    صحيح گنتی;
    صحيح اشاریہ;
    مزدوج جوڑ;
    مزدوج جوڑ_sq;
} OutlierDetector;

فراغ init_detector(OutlierDetector *detector) {
    لكل (صحيح i = 0; i < OUTLIER_DETECTOR_کھڑکی_ناپ; i++) {
        detector->بفر[i] = 0.0;
    }
    detector->گنتی = 0;
    detector->اشاریہ = 0;
    detector->جوڑ = 0.0;
    detector->جوڑ_sq = 0.0;
}

مزدوج update_detector(OutlierDetector *detector, مزدوج new_قدر) {

    إذا (detector->گنتی < OUTLIER_DETECTOR_کھڑکی_ناپ) {
        // here we are still building up a window of observations
        detector->بفر[detector->گنتی] = new_قدر;
        detector->جوڑ += new_قدر;
        detector->جوڑ_sq += new_قدر * new_قدر;
        detector->گنتی++;
        أرجع nan(""); // not enough data yet

    } وإلا {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        مزدوج old_قدر = detector->بفر[detector->اشاریہ];
        detector->جوڑ -= old_قدر;
        detector->جوڑ_sq -= old_قدر * old_قدر;
        // push the new value into the window
        detector->بفر[detector->اشاریہ] = new_قدر;
        detector->جوڑ += new_قدر;
        detector->جوڑ_sq += new_قدر * new_قدر;
        // move the index to the next position
        detector->اشاریہ = (detector->اشاریہ + 1) % OUTLIER_DETECTOR_کھڑکی_ناپ;
        // calculate the z-score of the new value
        مزدوج اوسط = detector->جوڑ / OUTLIER_DETECTOR_کھڑکی_ناپ;
        مزدوج variance = (detector->جوڑ_sq / OUTLIER_DETECTOR_کھڑکی_ناپ) - (اوسط * اوسط);
        مزدوج std_dev = جذر(variance);
        إذا (std_dev == 0.0) {
            أرجع 0.0;
        }
        مزدوج z = (new_قدر - اوسط) / std_dev;

        أرجع z;
    }
}
