#include <opencv2/opencv.hpp>
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

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

typedef vector<float> FeatureVector;

// Helper to extract class label from directory structure
string getClassLabel(const fs::path& imagePath) {
    return imagePath.parent_path().filename().string();
}



int main() {
    string dataDir = "/home/sai/Desktop/sem8/CV/Project/CS419_ImageRetrieval/Datasets/wang/Images/train";
    string testDir = "/home/sai/Desktop/sem8/CV/Project/CS419_ImageRetrieval/Datasets/wang/Images/test";
    string outputCSV = "/home/sai/Desktop/sem8/CV/Project/CS419_ImageRetrieval/Datasets/wang/FeatureDatabase/rgb_features.csv";

    int k = 10;
    string metric = "chiSquare"; // Choose from: "euclidean", "manhattan", "cosine", "chiSquare"

    RGBFeatureExtractor extractor;
    map<string, pair<string, FeatureVector>> featureDatabase;

    // Step 1: Extract features from training images
    for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            Mat img = imread(path);
            if (img.empty()) {
                cerr << "⚠️ Failed to load image: " << path << endl;
                continue;
            }
            string label = getClassLabel(entry.path());
            FeatureVector features = extractor.extractImgFeatures(img);
            featureDatabase[path] = {label, features};
            cout << "✔ Processed: " << path << " [" << label << "]\n";
        }
    }

    // Step 2: Save to CSV
    saveToCSV(outputCSV, featureDatabase);
    cout<<"Done"<<endl;
}
