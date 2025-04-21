#include "utils_data.hpp"

std::string extractClassName(const std::string& path) {
    std::filesystem::path p(path);
    return p.parent_path().filename().string();
}