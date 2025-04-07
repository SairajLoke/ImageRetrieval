#pragma once

#include <vector>
#include <string>
#include <map>
#include "distance_metrics.hpp" // For FeatureVector and DistanceMetrics

using namespace std;

vector<pair<float, string>> retrieveKNN(const FeatureVector& query,
                                        const map<string, FeatureVector>& database,
                                        int k,
                                        const string& metric);
