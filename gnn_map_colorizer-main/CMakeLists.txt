cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)
project(gnn_map_colorizer)

add_compile_options(-std=c++17)

# It's good practice to explicitly set these variables
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)

message(STATUS "Before finding packages")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  std_msgs
  fs_msgs
  rospack
  roslib
)
message(STATUS "After finding packages")

set(Torch_DIR "/ws/pytorch/torch/share/cmake/Torch")
find_package(Torch REQUIRED)


catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp tf std_msgs fs_msgs rospack
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(gnn_map_colorizer src/main.cpp)

target_link_libraries(gnn_map_colorizer
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)


