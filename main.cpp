// #include <opencv4/opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

#include "feature_extractor.hpp"
#include "distance_metrics.hpp"
#include "knn.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

typedef std::vector<float> FeatureVector;

// Helper to extract class label from directory structure
std::string getClassLabel(const fs::path& imagePath) {
    return imagePath.parent_path().filename().string();
}

int main() {
    std::string dataDir = "../Datasets/wang/Images/train";
    std::string testDir = "../Datasets/wang/Images/test";
    std::string outputCSV = "../Datasets/wang/FeatureDatabase/rgb_features.csv";

    int k = 10;
    std::string metric = "chiSquare"; // Choose from: "euclidean", "manhattan", "cosine", "chiSquare"

    RGBFeatureExtractor extractor;
    std::map<std::string, std::pair<std::string, FeatureVector>> featureDatabase;

    // Step 1: Extract features from training images
    for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            cv::Mat img = cv::imread(path);
            if (img.empty()) {
                std::cerr << "⚠️ Failed to load image: " << path << std::endl;
                continue;
            }
            std::string label = getClassLabel(entry.path());
            FeatureVector features = extractor.extractImgFeatures(img);
            featureDatabase[path] = {label, features};
            std::cout << "✔ Processed: " << path << " [" << label << "]\n";
        }
    }

    // Step 2: Save to CSV
    saveToCSV(outputCSV, featureDatabase);
    std::cout << "Done" << std::endl;
}