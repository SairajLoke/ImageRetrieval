cmake_minimum_required(VERSION 3.10)
project(CS419_ImageRetrieval)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(include)

# Source files
file(GLOB SOURCES
    src/*.cpp
)

add_executable(image_retrieval main.cpp ${SOURCES})

target_link_libraries(image_retrieval ${OpenCV_LIBS})
