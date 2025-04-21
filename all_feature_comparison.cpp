#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <map>

namespace fs = std::filesystem;

struct ImageFeature {
    std::string path;
    std::string label;
    std::vector<float> lbp_feature;
    std::vector<float> color_histogram;
    std::vector<float> color_moments;
    std::vector<float> glcm_feature;
    std::vector<float> gabor_feature;
    std::vector<float> sift_feature;
};

cv::Mat computeLBP(const cv::Mat& gray) {
    cv::Mat lbp = cv::Mat::zeros(gray.rows - 2, gray.cols - 2, CV_8UC1);
    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            uchar center = gray.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (gray.at<uchar>(i-1, j-1) > center) << 7;
            code |= (gray.at<uchar>(i-1, j  ) > center) << 6;
            code |= (gray.at<uchar>(i-1, j+1) > center) << 5;
            code |= (gray.at<uchar>(i,   j+1) > center) << 4;
            code |= (gray.at<uchar>(i+1, j+1) > center) << 3;
            code |= (gray.at<uchar>(i+1, j  ) > center) << 2;
            code |= (gray.at<uchar>(i+1, j-1) > center) << 1;
            code |= (gray.at<uchar>(i,   j-1) > center) << 0;
            lbp.at<uchar>(i-1, j-1) = code;
        }
    }
    return lbp;
}

std::vector<float> computeLBPHistogram(const cv::Mat& lbp) {
    const int histSize = 256;
    std::vector<float> hist(histSize, 0);
    for (int i = 0; i < lbp.rows; i++) {
        for (int j = 0; j < lbp.cols; j++) {
            hist[lbp.at<uchar>(i, j)]++;
        }
    }
    float total = lbp.rows * lbp.cols;
    for (auto& val : hist) val /= total;
    return hist;
}

std::vector<float> computeColorHistogram(const cv::Mat& img) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    std::vector<float> feature;
    int histSize = 64;
    float range[] = {0, 256};
    const float* histRange = {range};

    for (auto& channel : channels) {
        cv::Mat hist;
        cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
        hist /= img.total();
        feature.insert(feature.end(), (float*)hist.datastart, (float*)hist.dataend);
    }
    return feature;
}

std::vector<float> computeColorMoments(const cv::Mat& img) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    std::vector<float> moments;

    for (auto& c : channels) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(c, mean, stddev);
        moments.push_back((float)mean[0]);
        moments.push_back((float)stddev[0]);

        cv::Mat temp;
        cv::pow(c - mean[0], 3, temp);
        moments.push_back((float)cv::mean(temp)[0]);
    }
    return moments;
}

std::vector<float> computeGLCMFeatures(const cv::Mat& gray) {
    cv::Mat glcm = cv::Mat::zeros(256, 256, CV_32F);
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols - 1; j++) {
            uchar ref = gray.at<uchar>(i, j);
            uchar neighbor = gray.at<uchar>(i, j + 1);
            glcm.at<float>(ref, neighbor)++;
        }
    }
    glcm /= cv::sum(glcm)[0];

    float contrast = 0, homogeneity = 0, energy = 0, entropy = 0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            float p = glcm.at<float>(i, j);
            contrast += p * (i - j) * (i - j);
            homogeneity += p / (1 + abs(i - j));
            energy += p * p;
            if (p > 0) entropy -= p * log2(p);
        }
    }
    return {contrast, homogeneity, energy, entropy};
}

std::vector<float> computeGaborFeatures(const cv::Mat& gray) {
    std::vector<float> features;
    for (double theta = 0; theta < CV_PI; theta += CV_PI / 4) {
        cv::Mat kernel = cv::getGaborKernel(cv::Size(21, 21), 4.0, theta, 10.0, 0.5);
        cv::Mat filtered;
        cv::filter2D(gray, filtered, CV_32F, kernel);
        cv::Scalar mean, stddev;
        cv::meanStdDev(filtered, mean, stddev);
        features.push_back((float)mean[0]);
        features.push_back((float)stddev[0]);
    }
    return features;
}

std::vector<float> computeSIFTFeatures(const cv::Mat& gray) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    auto sift = cv::SIFT::create();
    sift->detectAndCompute(gray, cv::Mat(), keypoints, descriptors);

    std::vector<float> feature(128, 0.0f);
    if (!descriptors.empty()) {
        cv::reduce(descriptors, descriptors, 0, cv::REDUCE_AVG);
        feature.assign((float*)descriptors.datastart, (float*)descriptors.dataend);
    }
    return feature;
}

float chiSquaredDistance(const std::vector<float>& A, const std::vector<float>& B) {
    float dist = 0.0f;
    for (size_t i = 0; i < A.size(); i++) {
        float sum = A[i] + B[i];
        if (sum > 0)
            dist += ((A[i] - B[i]) * (A[i] - B[i])) / sum;
    }
    return dist;
}

void evaluateFeatures(const std::vector<ImageFeature>& images, int K, const std::string& featureType) {
    std::map<std::string, int> class_counts;
    std::map<std::string, float> top1_hits;
    std::map<std::string, float> topk_hits;

    for (size_t i = 0; i < images.size(); i++) {
        using Pair = std::pair<float, int>;
        auto cmp = [](Pair a, Pair b) { return a.first < b.first; };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> topK(cmp);

        const std::vector<float>& query =
            featureType == "LBP" ? images[i].lbp_feature :
            featureType == "ColorHist" ? images[i].color_histogram :
            featureType == "ColorMoment" ? images[i].color_moments :
            featureType == "GLCM" ? images[i].glcm_feature :
            featureType == "Gabor" ? images[i].gabor_feature :
            images[i].sift_feature;

        for (size_t j = 0; j < images.size(); j++) {
            if (i == j) continue;
            const std::vector<float>& candidate =
                featureType == "LBP" ? images[j].lbp_feature :
                featureType == "ColorHist" ? images[j].color_histogram :
                featureType == "ColorMoment" ? images[j].color_moments :
                featureType == "GLCM" ? images[j].glcm_feature :
                featureType == "Gabor" ? images[j].gabor_feature :
                images[j].sift_feature;

            float dist = chiSquaredDistance(query, candidate);
            if (topK.size() < K) {
                topK.push({ dist, (int)j });
            } else if (dist < topK.top().first) {
                topK.pop();
                topK.push({ dist, (int)j });
            }
        }

        int sameClassInTopK = 0;
        std::vector<Pair> results;
        while (!topK.empty()) {
            results.push_back(topK.top());
            topK.pop();
        }
        std::reverse(results.begin(), results.end());

        if (!results.empty() && images[results[0].second].label == images[i].label)
            top1_hits[images[i].label] += 1.0;

        for (auto& r : results) {
            if (images[r.second].label == images[i].label)
                sameClassInTopK++;
        }
        topk_hits[images[i].label] += (float)sameClassInTopK / K;
        class_counts[images[i].label]++;
    }

    std::cout << "\n==== " << featureType << " Per-Class Top-1 and Top-" << K << " Accuracy ====\n";
    for (const auto& [label, count] : class_counts) {
        float top1 = top1_hits[label] / count;
        float topk = topk_hits[label] / count;
        std::cout << "Class: " << label
                  << " | Top-1 Accuracy: " << top1 * 100 << "%"
                  << " | Top-" << K << " Accuracy: " << topk * 100 << "%\n";
    }
}

int main() {
    std::string folderPath = "Datasets/wang/Images/train";
    int K = 10;
    std::vector<ImageFeature> images;

    for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            std::string label = entry.path().parent_path().filename().string();
            cv::Mat img = cv::imread(path);
            if (img.empty() || img.rows < 3 || img.cols < 3) continue;

            ImageFeature f;
            f.path = path;
            f.label = label;

            cv::Mat gray;
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            f.lbp_feature = computeLBPHistogram(computeLBP(gray));
            f.color_histogram = computeColorHistogram(img);
            f.color_moments = computeColorMoments(img);
            f.glcm_feature = computeGLCMFeatures(gray);
            f.gabor_feature = computeGaborFeatures(gray);
            f.sift_feature = computeSIFTFeatures(gray);

            images.push_back(f);
        }
    }

    std::cout << "Total images loaded: " << images.size() << "\n";

    evaluateFeatures(images, K, "LBP");
    evaluateFeatures(images, K, "ColorHist");
    evaluateFeatures(images, K, "ColorMoment");
    evaluateFeatures(images, K, "GLCM");
    evaluateFeatures(images, K, "Gabor");
    evaluateFeatures(images, K, "SIFT");

    return 0;
}
