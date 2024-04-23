FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /ws
WORKDIR /ws

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    g++ \
    cmake \
    git \
    libboost-all-dev \
    pybind11-dev


RUN pip3 install torch \
    torch_geometric \
    numpy \
    catkin_tools \
    rospkg \
    tqdm
