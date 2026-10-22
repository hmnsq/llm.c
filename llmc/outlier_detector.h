/*
Simple OutlierDetector that we can use to monitor the loss and grad norm
Internally, it keeps track of a window of measurements and each time we
add a measurement, it returns the z-score of the new value with respect to
the window of measurements. This can be used to detect outliers in the data.

We use double so that the detector doesn't drift too much, because we
update the mean and variance with += on each step for efficiency. We could
reconsider this choice in the future, as the compute cost here is minimal.
*/

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <math.h>

// use compile-time constant for window size to avoid dynamic memory allocations
#ವ್ಯಾಖ್ಯಾನಿಸು OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ 128

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ದ್ವಿಗುಣ ಬಫರ್[OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ];
    ಪೂರ್ಣಾಂಕ ಎಣಿಕೆ_2;
    ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ;
    ದ್ವಿಗುಣ ಮೊತ್ತ;
    ದ್ವಿಗುಣ ಮೊತ್ತ_sq;
} OutlierDetector;

ಶೂನ್ಯ init_detector(OutlierDetector *detector) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ; i++) {
        detector->ಬಫರ್[i] = 0.0;
    }
    detector->ಎಣಿಕೆ_2 = 0;
    detector->ಸೂಚ್ಯಂಕ = 0;
    detector->ಮೊತ್ತ = 0.0;
    detector->ಮೊತ್ತ_sq = 0.0;
}

ದ್ವಿಗುಣ update_detector(OutlierDetector *detector, ದ್ವಿಗುಣ new_ಮೌಲ್ಯ) {

    ಆದರೆ (detector->ಎಣಿಕೆ_2 < OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ) {
        // here we are still building up a window of observations
        detector->ಬಫರ್[detector->ಎಣಿಕೆ_2] = new_ಮೌಲ್ಯ;
        detector->ಮೊತ್ತ += new_ಮೌಲ್ಯ;
        detector->ಮೊತ್ತ_sq += new_ಮೌಲ್ಯ * new_ಮೌಲ್ಯ;
        detector->ಎಣಿಕೆ_2++;
        ಹಿಂತಿರುಗಿ nan(""); // not enough data yet

    } ಇಲ್ಲದಿದ್ದರೆ {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        ದ್ವಿಗುಣ old_ಮೌಲ್ಯ = detector->ಬಫರ್[detector->ಸೂಚ್ಯಂಕ];
        detector->ಮೊತ್ತ -= old_ಮೌಲ್ಯ;
        detector->ಮೊತ್ತ_sq -= old_ಮೌಲ್ಯ * old_ಮೌಲ್ಯ;
        // push the new value into the window
        detector->ಬಫರ್[detector->ಸೂಚ್ಯಂಕ] = new_ಮೌಲ್ಯ;
        detector->ಮೊತ್ತ += new_ಮೌಲ್ಯ;
        detector->ಮೊತ್ತ_sq += new_ಮೌಲ್ಯ * new_ಮೌಲ್ಯ;
        // move the index to the next position
        detector->ಸೂಚ್ಯಂಕ = (detector->ಸೂಚ್ಯಂಕ + 1) % OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ;
        // calculate the z-score of the new value
        ದ್ವಿಗುಣ ಸರಾಸರಿ = detector->ಮೊತ್ತ / OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ;
        ದ್ವಿಗುಣ variance = (detector->ಮೊತ್ತ_sq / OUTLIER_DETECTOR_ಕಿಟಕಿ_ಗಾತ್ರ) - (ಸರಾಸರಿ * ಸರಾಸರಿ);
        ದ್ವಿಗುಣ std_dev = ವರ್ಗಮೂಲ(variance);
        ಆದರೆ (std_dev == 0.0) {
            ಹಿಂತಿರುಗಿ 0.0;
        }
        ದ್ವಿಗುಣ z = (new_ಮೌಲ್ಯ - ಸರಾಸರಿ) / std_dev;

        ಹಿಂತಿರುಗಿ z;
    }
}
