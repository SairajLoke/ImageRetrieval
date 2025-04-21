#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>

#pragma once
#include <string>
#include <map>
#include <vector>

std::string extractClassName(const std::string& path);

struct EvaluationStats {
    std::map<std::string, int> correctTop1PerClass;
    std::map<std::string, int> correctTopKPerClass;
    std::map<std::string, int> totalQueriesPerClass;
    std::map<std::string, int> PrecisionatKPerClass;

    void updateStats(const std::string& queryPath, const std::vector<std::pair<std::string, float>>& neighbors);
    void reportMetrics(int k) const;
};
