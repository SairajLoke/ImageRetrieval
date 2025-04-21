
#include "metrics.hpp"  // <-- Include your metrics header



void EvaluationStats::updateStats(const std::string& queryPath, const std::vector<std::pair<std::string, float>>& neighbors) {
  
        std::string queryClass = extractClassName(queryPath);
        totalQueriesPerClass[queryClass]++;

        int matchCount = 0;
        for (const auto& [path, dist] : neighbors) {
            std::string retrievedClass = extractClassName(path);
            if (retrievedClass == queryClass) matchCount++;
        }

        if (matchCount > 0) {
            correctTopKPerClass[queryClass]++;
            PrecisionatKPerClass[queryClass] += matchCount; //++;
        }
        std::string topClass = extractClassName(neighbors.front().first);
        if (topClass == queryClass) correctTop1PerClass[queryClass]++;
    }



void EvaluationStats::reportMetrics(int k) const {
        std::cout << "\n===== 📈 Evaluation Metrics =====\n";
        std::cout << "Class\tTop-1 Acc\tTop-" << k << " Acc\tTotal\n";

        float totalTop1 = 0, totalTopK = 0, precatK = 0;
        int classCount = 0;

        for (const auto& [className, total] : totalQueriesPerClass) {
            int correct1 = correctTop1PerClass.count(className) ? correctTop1PerClass.at(className) : 0;
            int correctK = correctTopKPerClass.count(className) ? correctTopKPerClass.at(className) : 0;
            int precisionK = PrecisionatKPerClass.count(className) ? PrecisionatKPerClass.at(className) : 0;

            float acc1 = (float)correct1 / total * 100.0f;
            float accK = ((float)correctK / (total)) * 100.0f;
            float precK = ((float)precisionK / (total*k)) * 100.0f;

            // cout<< "  " << className << "\t" << acc1 << "%\t\t" << accK << "%\t\t" << total << "\n";
            // std::cout<<"---"<<correctTopKPerClass.count(className)<<std::endl;
            // std::cout<<correctTopKPerClass.at(className)<<std::endl;

            totalTop1 += (float)correct1 / total;
            totalTopK += (float)correctK / (total);
            precatK += (float)correctK / (total*k);
            classCount++;

            std::cout << className << "\t" << acc1 << "%\t\t" << accK << "%\t\t" << precK << "%\t\t" << total << "\n";
        }

        //average over the quereies in eahc class
        std::cout << "\nMacro Avg Top-1 Accuracy: " << (totalTop1 / classCount) * 100.0f << "%\n";
        std::cout << "Macro Avg Top-" << k << " Accuracy: " << (totalTopK / classCount) * 100.0f << "%\n";
        std::cout << "Macro Avg Precision at " << k << ": " << (precatK / classCount) * 100.0f << "%\n";
}
