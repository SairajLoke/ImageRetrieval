
// #include "include/feature_extractor.hpp"
#pragma once


// Save features to CSV
void saveToCSV(const string& csvPath, const map<string, pair<string, FeatureVector>>& featureDB) {
    ofstream out(csvPath);
    if (!out.is_open()) {
        cerr << "Failed to open CSV file for writing: " << csvPath << endl;
        return;
    }

    // Write header
    out << "class,img_path";
    if (!featureDB.empty()) {
        for (size_t i = 0; i < featureDB.begin()->second.second.size(); ++i)
            out << ",f" << i;
    }
    out << "\n";

    // Write data rows
    for (const auto& [imgPath, data] : featureDB) {
        const string& label = data.first;
        const FeatureVector& vec = data.second;
        out << label << "," << imgPath;
        for (const auto& val : vec)
            out << "," << val;
        out << "\n";
    }

    out.close();
    cout << "✔ Saved features to: " << csvPath << endl;
}