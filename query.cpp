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

// Load feature database from CSV
map<string, pair<string, FeatureVector>> loadFromCSV(const string& csvPath) {
    map<string, pair<string, FeatureVector>> featureDB;
    ifstream in(csvPath);
    if (!in.is_open()) {
        cerr << " Failed to open CSV file: " << csvPath << endl;
        return featureDB;
    }

    string line;
    getline(in, line); // Skip header

    while (getline(in, line)) {
        stringstream ss(line);
        string label, path, token;
        FeatureVector vec;

        getline(ss, label, ',');
        getline(ss, path, ',');

        while (getline(ss, token, ',')) {
            vec.push_back(stof(token));
        }

        featureDB[path] = {label, vec};
    }

    in.close();
    return featureDB;
}

int main() {
    string testDir = "/home/sai/Desktop/sem8/CV/Project/CS419_ImageRetrieval/Datasets/wang/Images/test";
    string featureCSV = "/home/sai/Desktop/sem8/CV/Project/CS419_ImageRetrieval/Datasets/wang/FeatureDatabase/rgb_features.csv";
    string metric = "chiSquare"; // Options: euclidean, manhattan, cosine, chiSquare
    int k = 10;

    // Step 1: Load feature database
    auto featureDB = loadFromCSV(featureCSV);
    if (featureDB.empty()) {
        cerr << "Feature database is empty!" << endl;
        return 1;
    }

    // Step 2: Convert to simple path -> feature format for retrieval
    map<string, FeatureVector> simpleDB;
    for (const auto& [path, label_feat] : featureDB)
        simpleDB[path] = label_feat.second;

    // Step 3: Process and query test images
    RGBFeatureExtractor extractor;
    for (const auto& entry : fs::directory_iterator(testDir)) {
        if (!entry.is_regular_file()) continue;

        string queryPath = entry.path().string();
        Mat queryImage = imread(queryPath);
        if (queryImage.empty()) {
            cerr << "⚠️ Failed to load query image: " << queryPath << endl;
            continue;
        }

        FeatureVector queryFeatures = extractor.extractImgFeatures(queryImage);
        auto results = retrieveKNN(queryFeatures, simpleDB, k, metric);

        cout << "\n Query: " << queryPath << "\nTop " << k << " matches:\n";
        for (const auto& [dist, matchPath] : results) {
            cout << " - " << matchPath << " (Distance: " << dist << ")\n";
        }
    }

    return 0;
}
