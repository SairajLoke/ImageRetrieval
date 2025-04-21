#include "feature_extractor.hpp"

using namespace cv;
using namespace std;

FeatureVector RGBFeatureExtractor::extractImgFeatures(const Mat& image) {
        vector<Mat> channels;
        cv::split(image, channels);
        FeatureVector hist;
        int bins = 64;
        for (int i = 0; i < 3; i++) {
            Mat h;
            calcHist(&channels[i], 1, 0, Mat(), h, 1, &bins, 0);
            h /= image.total(); // normalize
            hist.insert(hist.end(), h.begin<float>(), h.end<float>());
        }
        return hist;
}

FeatureVector HSVFeatureExtractor ::extractImgFeatures(const Mat& image) {
        FeatureVector hist;
        if (image.empty()) return hist;

        Mat hsv;
        cvtColor(image, hsv, COLOR_BGR2HSV);
        vector<Mat> channels;
        cv::split(hsv, channels);
        int bins = 64;
        for (int i = 0; i < 3; i++) {
            Mat h;
            calcHist(&channels[i], 1, 0, Mat(), h, 1, &bins, 0);
            h /= image.total();
            hist.insert(hist.end(), h.begin<float>(), h.end<float>());
        }
        return hist;
}

FeatureVector OpponentFeatureExtractor :: extractImgFeatures(const Mat& image) {
        vector<Mat> channels(3);
        Mat floatImg;
        image.convertTo(floatImg, CV_32F);
        split(floatImg, channels);

        Mat O1 = (channels[2] - channels[1]) / sqrt(2);
        Mat O2 = (channels[2] + channels[1] - 2 * channels[0]) / sqrt(6);
        Mat O3 = (channels[2] + channels[1] + channels[0]) / sqrt(3);

        FeatureVector hist;
        int bins = 64;
        for (Mat o : {O1, O2, O3}) {
            Mat h;
            calcHist(&o, 1, 0, Mat(), h, 1, &bins, 0);
            h /= image.total();
            hist.insert(hist.end(), h.begin<float>(), h.end<float>());
        }
        return hist;
}


