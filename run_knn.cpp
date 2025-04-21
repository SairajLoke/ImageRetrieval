#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>

#define K_KNN 5

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

typedef vector<float> FeatureVector;



// ----------- Utility to extract class label from directory name -------------
string getClassLabel(const fs::path& imagePath) {
    return imagePath.parent_path().filename().string();
}

// ----------- RGB Histogram Feature Extractor -------------------------------
FeatureVector extractRGBHistogram(const Mat& img, int binsPerChannel = 8) {
    FeatureVector features;
    if (img.empty()) return features;

    vector<Mat> bgr;
    split(img, bgr);

    int histSize[] = {binsPerChannel};
    float range[] = {0, 256};
    const float* histRange[] = {range};
    bool uniform = true;
    bool accumulate = false;

    for (int i = 0; i < 3; ++i) {
        Mat hist;
        calcHist(&bgr[i], 1, 0, Mat(), hist, 1, histSize, histRange, uniform, accumulate);
        normalize(hist, hist, 1.0, 0.0, NORM_L1); // Normalize to sum = 1
        for (int j = 0; j < hist.rows; ++j)
            features.push_back(hist.at<float>(j));
    }
    return features;
}

// ----------- Save features to CSV ------------------------------------------
void saveToCSV(const string& filename, const map<string, pair<string, FeatureVector>>& db) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file to write: " << filename << endl;
        return;
    }

    for (const auto& [path, label_feat] : db) {
        file << path << "," << label_feat.first;
        for (float f : label_feat.second)
            file << "," << f;
        file << "\n";
    }
    file.close();
}


// ----------- Load features from CSV ----------------------------------------
void loadFromCSV(const string& filename, map<string, pair<string, FeatureVector>>& db) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file to read: " << filename << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string path, label, val;
        getline(ss, path, ',');
        getline(ss, label, ',');

        FeatureVector features;
        while (getline(ss, val, ',')) {
            features.push_back(stof(val));
        }
        db[path] = {label, features};
    }
    file.close();
}

// ----------- Euclidean Distance Function -----------------------------------
float euclideanDistance(const FeatureVector& a, const FeatureVector& b) {
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
    return dists;
}

// ----------- Main Program --------------------------------------------------
int main() {
    string dataDir = "Datasets/wang/Images/train";
    string testDir = "Datasets/wang/Images/test";
    string outputCSV = "Datasets/wang/FeatureDatabase/lbp_features.csv";

    // map<string, pair<string, FeatureVector>> featureDatabase;

    // // Step 1: Extract features from training images
    // for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
    //     if (entry.is_regular_file()) {
    //         string path = entry.path().string();
    //         Mat img = imread(path);
    //         if (img.empty()) {
    //             cerr << "Failed to load image: " << path << endl;
    //             continue;
    //         }
    //         string label = getClassLabel(entry.path());
    //         FeatureVector features = extractRGBHistogram(img);
    //         featureDatabase[path] = {label, features};
    //         cout << "Extracted: " << path << " [" << label << "]\n";
    //     }
    // }

    // // Step 2: Save features to CSV
    // saveToCSV(outputCSV, featureDatabase);
    // cout << "Features saved to: " << outputCSV << endl;

    // Step 3: Load features back from CSV
    map<string, pair<string, FeatureVector>> loadedFeatures;
    loadFromCSV(outputCSV, `);
    cout << "Features loaded from: " << outputCSV << endl;
    cout << "Total features loaded: " << loadedFeatures.size() << endl;
    // Check if loaded features are consistent
    // for (const auto& [path, label_feat] : loadedFeatures) {
    //     cout << "Loaded: " << path << " [" << label_feat.first << "]\n";
    // }


// Step 4: Extract test image features and find KNN
    for (const auto& entry : fs::recursive_directory_iterator(testDir)) {
        if (!entry.is_regular_file()) continue;

        string testPath = entry.path().string();
        cout << "\nProcessing test image: " << testPath << endl;

        Mat testImg = imread(testPath);
        if (testImg.empty()) {
            cerr << "  [Error] Failed to load test image." << endl;
            continue;
        }

        FeatureVector testFeatures = extractRGBHistogram(testImg);
        cout << "  Feature vector size: " << testFeatures.size() << endl;

        if (testFeatures.empty()) {
            cerr << "  [Error] Feature extraction failed." << endl;
            continue;
        }

        vector<pair<string, float>> neighbors = knn(testFeatures, loadedFeatures, K_KNN);

        cout << "  Top " << K_KNN << " Nearest Neighbors:\n";
        for (const auto& [path, dist] : neighbors) {
            cout << "    -> " << path << " (distance: " << dist << ")\n";
        }
    }


    return 0;
}