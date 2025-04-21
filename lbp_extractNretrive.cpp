#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "metrics.hpp" 
#include "utils_data.hpp"

#define K_KNN 5

using FeatureVector = std::vector<float>;
using namespace std;
using namespace cv;

typedef vector<float> FeatureVector;

// ----------- LBP Calculation Function -------------------------------------
cv::Mat calculateLBP(const cv::Mat& image) {
    assert(image.type() == CV_8UC1);
    cv::Mat lbpImage = cv::Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);
    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            uchar center = image.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (image.at<uchar>(i-1, j-1) > center) << 7;
            code |= (image.at<uchar>(i-1, j  ) > center) << 6;
            code |= (image.at<uchar>(i-1, j+1) > center) << 5;
            code |= (image.at<uchar>(i  , j+1) > center) << 4;
            code |= (image.at<uchar>(i+1, j+1) > center) << 3;
            code |= (image.at<uchar>(i+1, j  ) > center) << 2;
            code |= (image.at<uchar>(i+1, j-1) > center) << 1;
            code |= (image.at<uchar>(i  , j-1) > center) << 0;
            lbpImage.at<uchar>(i-1, j-1) = code;
        }
    }
    assert(lbpImage.rows == image.rows - 2 && lbpImage.cols == image.cols - 2);
    return lbpImage;
}

// ----------- LBP Histogram Calculation Function ----------------------------
std::vector<int> computeLBPHistogram(const cv::Mat& lbpImage) {
    std::vector<int> hist(256, 0);
    for (int i = 0; i < lbpImage.rows; i++) {
        for (int j = 0; j < lbpImage.cols; j++) {
            hist[lbpImage.at<uchar>(i, j)]++;
        }
    }
    assert(hist.size() == 256);
    return hist;
}

// ----------- Process Image Function ----------------------------------------
void processImageLBP(const std::string& imagePath, std::ofstream& lbpFile) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << " Could not read: " << imagePath << std::endl;
        return;
    }

    std::cout << "Processing image: " << imagePath << " with size: " << image.cols << "x" << image.rows << std::endl;

    cv::Mat lbp = calculateLBP(image);
    std::vector<int> lbpHist = computeLBPHistogram(lbp);

    assert(lbpHist.size() == 256);

    lbpFile << imagePath;
    for (int val : lbpHist) {
        lbpFile << "," << val;
    }
    lbpFile << "\n";

    std::cout << "Processed: " << imagePath << std::endl;
}

// ----------- Load Features from CSV ---------------------------------------
void loadFromCSV(const std::string& filepath,
                 std::map<std::string, std::pair<std::string, FeatureVector>>& features) {
    std::ifstream file(filepath);
    std::string line;
    std::getline(file, line);  // Skip header

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string path;
        std::getline(ss, path, ',');

        std::vector<int> hist;
        std::string value;
        while (std::getline(ss, value, ',')) {
            hist.push_back(std::stoi(value));
        }

        FeatureVector floatHist(hist.begin(), hist.end());
        assert(floatHist.size() == 256);
        features[path] = std::make_pair(path, floatHist);
    }

    file.close();
    std::cout << " Loaded " << features.size() << " feature vectors from CSV.\n";
}

// ----------- Euclidean Distance Function -----------------------------------
float euclideanDistance(const FeatureVector& a, const FeatureVector& b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// ----------- KNN Implementation --------------------------------------------
vector<pair<string, float>> knn(const FeatureVector& query,
                                const map<string, pair<string, FeatureVector>>& db,
                                int k) {
    assert(!query.empty());
    vector<pair<string, float>> dists;

    for (const auto& [path, label_feat] : db) {
        float dist = euclideanDistance(query, label_feat.second);
        dists.push_back({path, dist});
    }

    sort(dists.begin(), dists.end(), [](auto& a, auto& b) {
        return a.second < b.second;
    });

    if ((int)dists.size() > k)
        dists.resize(k);

    std::cout << "  Top " << k << " Nearest Neighbors:\n";
    for (const auto& [path, dist] : dists) {
        std::cout << "    -> " << path << " (distance: " << dist << ")\n";
    }

    return dists;
}

// ----------- Main Program --------------------------------------------------
int main() {
    std::string datasetPath = "../Datasets/wang/Images/train";
    std::string testPath = "../Datasets/wang/Images/test";
    std::string outputCSV = "../Datasets/wang/FeatureDatabase/lbp_features.csv";
    bool GET_FEATURES = false;

    if (GET_FEATURES){
        // Step 1: Extract and save LBP features
        std::ofstream lbpFile(outputCSV);
        if (!lbpFile.is_open()) {
            std::cerr << " Failed to open output CSV file: " << outputCSV << std::endl;
            return -1;
        }

        lbpFile << "image_path";
        for (int i = 0; i < 256; i++) lbpFile << ",lbp_" << i;
        lbpFile << "\n";

        for (const auto& entry : std::filesystem::recursive_directory_iterator(datasetPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
                processImageLBP(entry.path().string(), lbpFile);
            }
        }

        lbpFile.close();
        std::cout << " All LBP features extracted and saved to " << outputCSV << std::endl;
    }


    // Step 2: Load extracted features
    std::map<std::string, pair<string, FeatureVector>> loadedFeatures;
    loadFromCSV(outputCSV, loadedFeatures);

    // Step 3: Process test images and evaluate
    EvaluationStats stats;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(testPath)) {
        if (!entry.is_regular_file()) continue;

        std::string testImgPath = entry.path().string();
        std::cout << "\nProcessing test image: " << testImgPath << std::endl;

        cv::Mat testImg = cv::imread(testImgPath);
        if (testImg.empty()) {
            std::cerr << "  [Error] Failed to load test image." << std::endl;
            continue;
        }

        cv::Mat grayTestImg;
        cvtColor(testImg, grayTestImg, COLOR_BGR2GRAY);

        cv::Mat lbpTest = calculateLBP(grayTestImg);
        std::vector<int> intHist = computeLBPHistogram(lbpTest);
        FeatureVector testFeatures(intHist.begin(), intHist.end());

        std::cout << "  Feature vector size: " << testFeatures.size() << std::endl;
        assert(testFeatures.size() == 256);

        auto neighbors = knn(testFeatures, loadedFeatures, K_KNN);
        stats.updateStats(testImgPath, neighbors);
    }

    stats.reportMetrics(K_KNN);
    stats.reportMetrics(10);
    return 0;
}
