#include<iostream>
#include<algorithm>
#include<map>
#include "knn.hpp" // Include the interface


// Retrieve KNN
vector<pair<float, string>> retrieveKNN(const FeatureVector& query,
                                        const map<string, FeatureVector>& database,
                                        int k,
                                        const string& metric) {
    vector<pair<float, string>> distances;

    for (const auto& [imgPath, features] : database) {
        float dist = 0.0f;
        if (metric == "euclidean")
            dist = DistanceMetrics::euclidean(query, features);
        else if (metric == "manhattan")
            dist = DistanceMetrics::manhattan(query, features);
        else if (metric == "cosine")
            dist = DistanceMetrics::cosine(query, features);
        else if (metric == "chiSquare")
            dist = DistanceMetrics::chiSquare(query, features);
        else {
            std::cerr << " Unknown distance metric: " << metric << std::endl;
            exit(1);
        }
        distances.emplace_back(dist, imgPath);
    }

    std::sort(distances.begin(), distances.end()); // sort by distance (ascending)

    if ((int)distances.size() > k)
        distances.resize(k);

    return distances;
}
