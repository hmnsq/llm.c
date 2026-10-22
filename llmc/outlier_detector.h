/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#నిర్వచించు OUTLIER_DETECTOR_కిటికీ_కొలత 128

రకనిర్వచనం నిర్మాణం {
    ద్విగుణ బఫరు[OUTLIER_DETECTOR_కిటికీ_కొలత];
    పూర్ణాంకం లెక్క;
    పూర్ణాంకం సూచీ;
    ద్విగుణ కూడిక;
    ద్విగుణ కూడిక_sq;
} OutlierDetector;

శూన్య init_detector(OutlierDetector *detector) {
    కొరకు (పూర్ణాంకం i = 0; i < OUTLIER_DETECTOR_కిటికీ_కొలత; i++) {
        detector->బఫరు[i] = 0.0;
    }
    detector->లెక్క = 0;
    detector->సూచీ = 0;
    detector->కూడిక = 0.0;
    detector->కూడిక_sq = 0.0;
}

ద్విగుణ update_detector(OutlierDetector *detector, ద్విగుణ new_విలువ) {

    ఒకవేళ (detector->లెక్క < OUTLIER_DETECTOR_కిటికీ_కొలత) {
        // here we are still building up a window of observations
        detector->బఫరు[detector->లెక్క] = new_విలువ;
        detector->కూడిక += new_విలువ;
        detector->కూడిక_sq += new_విలువ * new_విలువ;
        detector->లెక్క++;
        తిరిగి nan(""); // not enough data yet

    } లేకపోతే {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        ద్విగుణ old_విలువ = detector->బఫరు[detector->సూచీ];
        detector->కూడిక -= old_విలువ;
        detector->కూడిక_sq -= old_విలువ * old_విలువ;
        // push the new value into the window
        detector->బఫరు[detector->సూచీ] = new_విలువ;
        detector->కూడిక += new_విలువ;
        detector->కూడిక_sq += new_విలువ * new_విలువ;
        // move the index to the next position
        detector->సూచీ = (detector->సూచీ + 1) % OUTLIER_DETECTOR_కిటికీ_కొలత;
        // calculate the z-score of the new value
        ద్విగుణ సగటు = detector->కూడిక / OUTLIER_DETECTOR_కిటికీ_కొలత;
        ద్విగుణ variance = (detector->కూడిక_sq / OUTLIER_DETECTOR_కిటికీ_కొలత) - (సగటు * సగటు);
        ద్విగుణ std_dev = వర్గమూలం(variance);
        ఒకవేళ (std_dev == 0.0) {
            తిరిగి 0.0;
        }
        ద్విగుణ z = (new_విలువ - సగటు) / std_dev;

        తిరిగి z;
    }
}
