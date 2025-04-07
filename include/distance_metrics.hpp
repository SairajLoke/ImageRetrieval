#pragma once

#include <vector>
#include <string>

using namespace std;

typedef vector<float> FeatureVector;

class DistanceMetrics {
public:
    static float euclidean(const FeatureVector& a, const FeatureVector& b);
    static float manhattan(const FeatureVector& a, const FeatureVector& b);
    static float cosine(const FeatureVector& a, const FeatureVector& b);
    static float chiSquare(const FeatureVector& a, const FeatureVector& b, float epsilon = 1e-10);
};
