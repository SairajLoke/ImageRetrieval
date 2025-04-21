cmake_minimum_required(VERSION 3.10)
project(CS419_ImageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV
# sudo apt install libopencv-dev

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES src/*.cpp)

# Output binary to bin/ folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Create executable
add_executable(image_retrieval lbp_extractNretrive ${SOURCES})

# Link OpenCV
target_link_libraries(image_retrieval ${OpenCV_LIBS})
