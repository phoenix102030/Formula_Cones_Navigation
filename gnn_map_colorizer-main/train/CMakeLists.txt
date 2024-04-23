cmake_minimum_required(VERSION 3.22.1)
project(fs_track_generator)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# Set optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Define target
add_library(fsgenerator MODULE src/track_gen.cpp)

# Include directories
target_include_directories(fsgenerator PRIVATE include)

# Link libraries
target_link_libraries(fsgenerator PRIVATE pybind11::module OpenMP::OpenMP_CXX)

# Set properties for Python module
set_target_properties(fsgenerator PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Enable C++17
target_compile_features(fsgenerator PRIVATE cxx_std_17)
