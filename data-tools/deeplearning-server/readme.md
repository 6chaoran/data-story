

## Set up system

install anaconda
https://www.anaconda.com/download/#linux


## Install GPU drivers

1. install NVIDA CUDA toolkit - 9.0
https://developer.nvidia.com/cuda-toolkit

Installation Instructions:
```
sudo dpkg -i cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

2. NVIDIA cuDNN



### install opencv 3.4
https://github.com/opencv/opencv/tree/3.4

```
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
```

```
cd ~/
git clone -b 3.4 https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake ~/opencv
make -j7 # depends on your cores
sudo make install
```