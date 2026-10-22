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
#تعریف_کرو OUTLIER_DETECTOR_کھڑکی_سائز 128

قسمتعریف ساخت {
    دوہرا بفر[OUTLIER_DETECTOR_کھڑکی_سائز];
    صحیح گنتی;
    صحیح اشاریہ;
    دوہرا مجموع;
    دوہرا مجموع_sq;
} OutlierDetector;

خالی init_detector(OutlierDetector *detector) {
    برائے (صحیح i = 0; i < OUTLIER_DETECTOR_کھڑکی_سائز; i++) {
        detector->بفر[i] = 0.0;
    }
    detector->گنتی = 0;
    detector->اشاریہ = 0;
    detector->مجموع = 0.0;
    detector->مجموع_sq = 0.0;
}

دوہرا update_detector(OutlierDetector *detector, دوہرا new_قدر) {

    اگر (detector->گنتی < OUTLIER_DETECTOR_کھڑکی_سائز) {
        // here we are still building up a window of observations
        detector->بفر[detector->گنتی] = new_قدر;
        detector->مجموع += new_قدر;
        detector->مجموع_sq += new_قدر * new_قدر;
        detector->گنتی++;
        واپس nan(""); // not enough data yet

    } ورنہ {
        // we've filled the window, so now we can start detecting outliers

        // pop the oldest value from the window
        دوہرا old_قدر = detector->بفر[detector->اشاریہ];
        detector->مجموع -= old_قدر;
        detector->مجموع_sq -= old_قدر * old_قدر;
        // push the new value into the window
        detector->بفر[detector->اشاریہ] = new_قدر;
        detector->مجموع += new_قدر;
        detector->مجموع_sq += new_قدر * new_قدر;
        // move the index to the next position
        detector->اشاریہ = (detector->اشاریہ + 1) % OUTLIER_DETECTOR_کھڑکی_سائز;
        // calculate the z-score of the new value
        دوہرا اوسط = detector->مجموع / OUTLIER_DETECTOR_کھڑکی_سائز;
        دوہرا variance = (detector->مجموع_sq / OUTLIER_DETECTOR_کھڑکی_سائز) - (اوسط * اوسط);
        دوہرا std_dev = جذر(variance);
        اگر (std_dev == 0.0) {
            واپس 0.0;
        }
        دوہرا z = (new_قدر - اوسط) / std_dev;

        واپس z;
    }
}
