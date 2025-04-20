#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// === Local Binary Pattern ===
cv::Mat calculateLBP(const cv::Mat& image) {
    cv::Mat lbpImage = cv::Mat::zeros(image.rows - 2, image.cols - 2, CV_8UC1);
    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            uchar center = image.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (image.at<uchar>(i-1, j-1) > center) << 7;
            code |= (image.at<uchar>(i-1, j  ) > center) << 6;
            code |= (image.at<uchar>(i-1, j+1) > center) << 5;
            code |= (image.at<uchar>(i  , j+1) > center) << 4;
            code |= (image.at<uchar>(i+1, j+1) > center) << 3;
            code |= (image.at<uchar>(i+1, j  ) > center) << 2;
            code |= (image.at<uchar>(i+1, j-1) > center) << 1;
            code |= (image.at<uchar>(i  , j-1) > center) << 0;
            lbpImage.at<uchar>(i-1, j-1) = code;
        }
    }
    return lbpImage;
}

std::vector<int> computeLBPHistogram(const cv::Mat& lbpImage) {
    std::vector<int> hist(256, 0);
    for (int i = 0; i < lbpImage.rows; i++)
        for (int j = 0; j < lbpImage.cols; j++)
            hist[lbpImage.at<uchar>(i, j)]++;
    return hist;
}

// === GLCM Histogram ===
std::vector<int> computeGLCMHistogram(const cv::Mat& grayImage, int bins = 64) {
    cv::Mat glcm = cv::Mat::zeros(256, 256, CV_32S);

    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols - 1; j++) {
            int row = grayImage.at<uchar>(i, j);
            int col = grayImage.at<uchar>(i, j + 1);
            glcm.at<int>(row, col)++;
        }
    }

    double maxVal = 0;
    cv::minMaxLoc(glcm, nullptr, &maxVal);
    std::vector<int> hist(bins, 0);

    for (int i = 0; i < glcm.rows; i++) {
        for (int j = 0; j < glcm.cols; j++) {
            int value = glcm.at<int>(i, j);
            int bin = static_cast<int>((double)value / (maxVal + 1e-5) * bins);
            bin = std::clamp(bin, 0, bins - 1);
            hist[bin]++;
        }
    }

    return hist;
}

// === Gabor Histogram ===
std::vector<int> computeGaborHistogram(const cv::Mat& grayImage, int bins = 64) {
    double lambda = 8.0, theta = CV_PI / 4, psi = 0, gamma = 0.5, sigma = 4.0;
    cv::Mat gaborKernel = cv::getGaborKernel(cv::Size(21, 21), sigma, theta, lambda, gamma, psi, CV_64F);

    cv::Mat filtered;
    cv::filter2D(grayImage, filtered, CV_32F, gaborKernel);

    double minVal, maxVal;
    cv::minMaxLoc(filtered, &minVal, &maxVal);

    std::vector<int> hist(bins, 0);
    for (int i = 0; i < filtered.rows; i++) {
        for (int j = 0; j < filtered.cols; j++) {
            float val = filtered.at<float>(i, j);
            int bin = static_cast<int>(((val - minVal) / (maxVal - minVal + 1e-5)) * bins);
            bin = std::clamp(bin, 0, bins - 1);
            hist[bin]++;
        }
    }
    return hist;
}

// === Process Single Image ===
void processImage(const std::string& imagePath,
                  std::ofstream& lbpFile,
                  std::ofstream& glcmFile,
                  std::ofstream& gaborFile) {

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Warning: Cannot read " << imagePath << std::endl;
        return;
    }

    // LBP
    cv::Mat lbp = calculateLBP(image);
    auto lbpHist = computeLBPHistogram(lbp);
    lbpFile << imagePath;
    for (auto val : lbpHist) lbpFile << "," << val;
    lbpFile << "\n";

    // GLCM
    auto glcmHist = computeGLCMHistogram(image);
    glcmFile << imagePath;
    for (auto val : glcmHist) glcmFile << "," << val;
    glcmFile << "\n";

    // Gabor
    auto gaborHist = computeGaborHistogram(image);
    gaborFile << imagePath;
    for (auto val : gaborHist) gaborFile << "," << val;
    gaborFile << "\n";

    std::cout << "Extracted: " << imagePath << std::endl;
}

// === Main ===
int main() {
    std::string datasetPath = "Datasets/wang/Images/train";

    std::ofstream lbpFile("lbp_features.csv");
    std::ofstream glcmFile("glcm_features.csv");
    std::ofstream gaborFile("gabor_features.csv");

    // Headers
    lbpFile << "image_path";
    for (int i = 0; i < 256; i++) lbpFile << ",hist_" << i;
    lbpFile << "\n";

    glcmFile << "image_path";
    for (int i = 0; i < 64; i++) glcmFile << ",glcm_hist_" << i;
    glcmFile << "\n";

    gaborFile << "image_path";
    for (int i = 0; i < 64; i++) gaborFile << ",gabor_hist_" << i;
    gaborFile << "\n";

    for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            processImage(entry.path().string(), lbpFile, glcmFile, gaborFile);
        }
    }

    lbpFile.close();
    glcmFile.close();
    gaborFile.close();

    std::cout << "✅ All features extracted and saved.\n";
    return 0;
}
