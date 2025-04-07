#include "distance_metrics.hpp"
#include <cmath>
#include <algorithm>

float DistanceMetrics::euclidean(const FeatureVector& a, const FeatureVector& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum += pow(a[i] - b[i], 2);
    return sqrt(sum);
}

float DistanceMetrics::manhattan(const FeatureVector& a, const FeatureVector& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum += abs(a[i] - b[i]);
    return sum;
}

float DistanceMetrics::cosine(const FeatureVector& a, const FeatureVector& b) {
    float dot = 0, normA = 0, normB = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return 1 - (dot / (sqrt(normA) * sqrt(normB)));
}

float DistanceMetrics::chiSquare(const FeatureVector& a, const FeatureVector& b, float epsilon) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum += pow(a[i] - b[i], 2) / (a[i] + b[i] + epsilon);
    return sum / 2;
}
