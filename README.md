
# ShuttlecockInspection
Visual Inspection of Shuttlecock(Badminton Shuttles).  
Good and defective shuttlecock are classified.  
The predict result, accuracy, and FPS are displayed in the upper left corner of the window.  

![demo](https://github.com/Souya-Co-Ltd/shuttlecock_inspection/blob/main/imgs/demo.gif)

# Requirement
## Hardware
* Jetson AGX Xavier Development Kit
* [Logicool C270n(HD WEBCAM)](https://www.logicool.co.jp/ja-jp/products/webcams/hd-webcam-c270n.960-001265.html)

## Software
* Jetpack 5.0.1
* Python 3.8.10
* Tensorflow 2.9.1+nv22.06
* CV2 4.5.1(GSTREAMER support)

# Setup
## Clone project
```console
$ git clone https://github.com/Souya-Co-Ltd/shuttlecock_inspection
```

## Install and upgrade pip3
```console
$ sudo apt-get -y update
$ sudo apt-get install -y python3-pip
$ pip install -U pip testresources setuptools==49.6.0
```

## Build CV2
```console
$ sudo apt install -y libtbb-dev
$ pip install -U numpy --no-cache-dir --no-binary numpy
$ wget https://github.com/opencv/opencv/archive/4.5.1.zip
$ unzip 4.5.1.zip
$ rm 4.5.1.zip
$ mv opencv-4.5.1 OpenCV
$ cd OpenCV
$ wget https://github.com/opencv/opencv_contrib/archive/4.5.1.zip
$ unzip 4.5.1.zip
$ rm 4.5.1.zip
$ mkdir build
$ cd build
$ cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.1/modules/ \
    -D OPENCV_DNN_CUDA=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_DOCS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_JASPER=OFF \
    -D BUILD_OPENEXR=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_opencv_ml=OFF \
    -D ENABLE_FAST_MATH=ON \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TBB=ON \
    -D WITH_OPENMP=ON \
    -D WITH_CUDA=ON \
    -D WITH_NVCUVID=OFF \
    -D BUILD_opencv_cudacodec=OFF \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUFFT=ON \
    -D build_opencv_python3=ON \
    -D build_opencv_python2=OFF \
    -D WITH_PYTHON=ON \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    ..
$ make all -j4
$ make install
```

## Install TensorFlow
```console
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
$ pip install -U pip testresources setuptools==49.6.0
$ pip install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig packaging
$ sudo env H5PY_SETUP_REQUIRES=0
$ pip install -U h5py==3.1.0
$ pip install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 tensorflow==2.9.1+nv22.06 tensorflow-io
```

# Running
```console
$ export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:/lib/aarch64-linux-gnu/libGLdispatch.so.0
$ cd [ShuttlecockInspection path]
$ python3 run_shuttlecock_inspection.py
```

The camera device can be specified as an argument.  
(The default value is "/dev/video0")  
For example, to use the camera in "/dev/video1", execute as follows.  
```
$ python3 run_shuttlecock_inspection.py -d /dev/video1
```

# Video
[YouTube](https://youtu.be/AlcIkxrt85A)

# Author
Souya Co., Ltd
