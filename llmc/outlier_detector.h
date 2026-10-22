/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ 128

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਦੁੱਗਣਾ ਬਫਰ[OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ];
    ਪੂਰਨ ਗਿਣਤੀ_2;
    ਪੂਰਨ ਸੂਚਕਾਂਕ;
    ਦੁੱਗਣਾ ਜੋੜ;
    ਦੁੱਗਣਾ ਜੋੜ_sq;
} OutlierDetector;

ਖਾਲੀ init_detector(OutlierDetector *detector) {
    ਲਈ (ਪੂਰਨ i = 0; i < OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ; i++) {
        detector->ਬਫਰ[i] = 0.0;
    }
    detector->ਗਿਣਤੀ_2 = 0;
    detector->ਸੂਚਕਾਂਕ = 0;
    detector->ਜੋੜ = 0.0;
    detector->ਜੋੜ_sq = 0.0;
}

ਦੁੱਗਣਾ update_detector(OutlierDetector *detector, ਦੁੱਗਣਾ new_ਮੁੱਲ) {

    ਜੇ (detector->ਗਿਣਤੀ_2 < OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ) {
        // here we are still building up a window of observations
        detector->ਬਫਰ[detector->ਗਿਣਤੀ_2] = new_ਮੁੱਲ;
        detector->ਜੋੜ += new_ਮੁੱਲ;
        detector->ਜੋੜ_sq += new_ਮੁੱਲ * new_ਮੁੱਲ;
        detector->ਗਿਣਤੀ_2++;
        ਵਾਪਸ nan(""); // not enough data yet

    } ਨਹੀਂਤਾਂ {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        ਦੁੱਗਣਾ old_ਮੁੱਲ = detector->ਬਫਰ[detector->ਸੂਚਕਾਂਕ];
        detector->ਜੋੜ -= old_ਮੁੱਲ;
        detector->ਜੋੜ_sq -= old_ਮੁੱਲ * old_ਮੁੱਲ;
        // push the new value into the window
        detector->ਬਫਰ[detector->ਸੂਚਕਾਂਕ] = new_ਮੁੱਲ;
        detector->ਜੋੜ += new_ਮੁੱਲ;
        detector->ਜੋੜ_sq += new_ਮੁੱਲ * new_ਮੁੱਲ;
        // move the index to the next position
        detector->ਸੂਚਕਾਂਕ = (detector->ਸੂਚਕਾਂਕ + 1) % OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ;
        // calculate the z-score of the new value
        ਦੁੱਗਣਾ ਔਸਤ = detector->ਜੋੜ / OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ;
        ਦੁੱਗਣਾ variance = (detector->ਜੋੜ_sq / OUTLIER_DETECTOR_ਖਿੜਕੀ_ਮਾਪ) - (ਔਸਤ * ਔਸਤ);
        ਦੁੱਗਣਾ std_dev = ਵਰਗਮੂਲ(variance);
        ਜੇ (std_dev == 0.0) {
            ਵਾਪਸ 0.0;
        }
        ਦੁੱਗਣਾ z = (new_ਮੁੱਲ - ਔਸਤ) / std_dev;

        ਵਾਪਸ z;
    }
}
