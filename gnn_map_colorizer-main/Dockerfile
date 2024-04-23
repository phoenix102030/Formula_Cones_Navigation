#!/usr/bin/env docker
FROM ros:noetic-ros-base-focal
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_ROOT=/opt/ros/noetic

RUN mkdir -p /ws/src
WORKDIR /ws

RUN apt-get update && apt-get upgrade -y 
RUN apt-get install -y \
        git \
        nano \
        python3-pip \
        python3-catkin-tools \
    rm -rf /var/lib/apt/lists/* && apt-get clean
    
RUN pip3 install \
    numpy \
    torch \
    torch_geometric \
    catkin_tools \
    rospkg \
    typing_extensions

RUN git clone https://github.com/Kitware/CMake.git
RUN cd CMake && \
    ./bootstrap && \
    make -j 8 && \
    make install -j 8

RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git submodule update --init --recursive

ENV MAX_JOBS=1
RUN cd pytorch && \
    python3 setup.py develop

