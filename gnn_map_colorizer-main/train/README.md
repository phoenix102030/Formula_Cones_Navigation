This folder is for training the network!

To start training, please follow **all** the instruction below:

## 1. Build the docker image
````
docker pull ubuntu
docker build -t train_gnn_map_colorizer .
````

## 2. Enter the docker image
````
docker run -it -v ./:/ws train_gnn_map_colorizer /bin/bash
````

## 3. Build the track generator
Now that you are inside the docker container, run the following commands:
````
cd /ws
mkdir build
cd build
cmake ..
make
````

## 4. Start training
````
cd /ws
python3 train.py
````
