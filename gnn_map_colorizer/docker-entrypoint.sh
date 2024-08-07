#!/bin/bash
set -e
source /root/.bashrc

ROS_DISTRO=noetic  #<--- TODO: change to your ROS version

source "/opt/ros/$ROS_DISTRO/setup.bash"

if [ ! -e "CONTAINER_INITIALIZED_PLACEHOLDER" ]; then
    echo "-- First container startup --"
    catkin init
    #catkin config
    catkin build
    touch "CONTAINER_INITIALIZED_PLACEHOLDER"
    # This placeholder file used in the github action to check when catkin build is done, do not remove
else
    echo "-- Not first container startup --"
    source "/ws/devel/setup.bash"
fi


exec "$@"