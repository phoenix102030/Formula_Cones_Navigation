#include <ros/ros.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>
#include <string>

#include "ros_interface.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "gnn_map_colorizer");
    ROS_GNNColorizer colorizer;
    ros::spin();
    return 0;
}