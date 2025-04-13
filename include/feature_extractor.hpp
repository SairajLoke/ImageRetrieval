// FeatureExtractor.h
#pragma once
#include <opencv4/opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <map> 

typedef std::vector<float> FeatureVector;

class FeatureExtractor {
public:
    virtual FeatureVector extractImgFeatures(const cv::Mat& image) = 0; // Pure virtual
    virtual ~FeatureExtractor() = default;
    std::map<std::string, FeatureVector> featureMap; // Map to store features
};


class RGBFeatureExtractor : public FeatureExtractor {
public:
    FeatureVector extractImgFeatures(const cv::Mat& image) override;
};

class HSVFeatureExtractor : public FeatureExtractor {
public:
    FeatureVector extractImgFeatures(const cv::Mat& image) override;
};

class OpponentFeatureExtractor : public FeatureExtractor {
public:
    FeatureVector extractImgFeatures(const cv::Mat& image) override;
};