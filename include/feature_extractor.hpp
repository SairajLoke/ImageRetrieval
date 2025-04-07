// FeatureExtractor.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map> 

using namespace cv;
using namespace std;

typedef vector<float> FeatureVector;

class FeatureExtractor {
public:
    virtual FeatureVector extractImgFeatures(const Mat& image) = 0; // Pure virtual
    virtual ~FeatureExtractor() = default;
    map<string, FeatureVector> featureMap; // Map to store features
};


class RGBFeatureExtractor : public FeatureExtractor {
public:
    FeatureVector extractImgFeatures(const Mat& image) override;
};

class HSVFeatureExtractor : public FeatureExtractor {
public:
    FeatureVector extractImgFeatures(const Mat& image) override;
};

class OpponentFeatureExtractor : public FeatureExtractor {
public:
    FeatureVector extractImgFeatures(const Mat& image) override;
};
