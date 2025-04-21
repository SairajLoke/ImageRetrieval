#include <utils_data.hpp>

void extractFeatures(string_view dataDir, const string& featureFile) {
    // Initialize the feature extractor
    FeatureExtractor extractor;
    FeatureDatabase featureDatabase;

    // Step 1: Create the output file
    ofstream outFile(featureFile);
    if (!outFile.is_open()) {
        cerr << "Failed to open output file: " << featureFile << endl;
        return;
    }

    // Step 2: Create a directory iterator for the training images
    namespace fs = std::filesystem;

// Step 1: Extract features from training images
    for (const auto& entry : fs::recursive_directory_iterator(dataDir)) {
        if (entry.is_regular_file()) {
            string path = entry.path().string();
            Mat img = imread(path);
            if (img.empty()) {
                cerr << "Failed to load image: " << path << endl;
                continue;
            }
            string label = getClassLabel(entry.path());
            FeatureVector features = extractor.extractImgFeatures(img);
            featureDatabase[path] = {label, features};
            cout << "Processed: " << path << " [" << label << "]\n";
        }
    }


    // Step 2: Save features to CSV
    outFile << "ImagePath,Label,Features\n";
    for (const auto& [path, pair] : featureDatabase) {
        outFile << path << "," << pair.first << ",";
        for (const auto& feature : pair.second) {
            outFile << feature << " ";
        }
        outFile << "\n";
    }
    outFile.close();

    cout << "Feature extraction completed. Features saved to " << featureFile << endl;
}