FROM nvcr.io/nvidia/l4t-base:r32.5.0

# RUN add-apt-repository universe

#
# setup environment
#
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV LLVM_CONFIG="/usr/bin/llvm-config-9"
ENV HOME /root
ARG MAKEFLAGS=-j4
ARG OPENCV_VERSION=4.5.2

RUN printenv

#
# apt packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          python3-pip \
		  python3-dev \
          python3-matplotlib \
		  build-essential \
		  gfortran \
		  git \
		  cmake \
		  curl \
		  libopenblas-dev \
		  liblapack-dev \
		  libblas-dev \
		  libhdf5-serial-dev \
		  hdf5-tools \
		  libhdf5-dev \
		  zlib1g-dev \
		  zip \
		  libjpeg8-dev \
		  libopenmpi2 \
          openmpi-bin \
          openmpi-common \
		  protobuf-compiler \
          libprotoc-dev \
		llvm-9 \
          llvm-9-dev \
          nano \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# RUN python3 -m pip install protobuf

# RUN apt-get -y update
# RUN apt-get -y upgrade
# RUN apt-get -y autoremove
RUN sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf" && \
	ldconfig
# third-party libraries

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		build-essential cmake git unzip pkg-config \
		libjpeg-dev libpng-dev libtiff-dev \ 
		libavcodec-dev libavformat-dev libswscale-dev \ 
		libgtk2.0-dev libcanberra-gtk* \ 
		python3-dev python3-numpy python3-pip \ 
		libxvidcore-dev libx264-dev libgtk-3-dev \ 
		libtbb2 libtbb-dev libdc1394-22-dev \ 
		libv4l-dev v4l-utils \ 
		libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \ 
		libavresample-dev libvorbis-dev libxine2-dev \ 
		libfaac-dev libmp3lame-dev libtheora-dev \ 
		libopencore-amrnb-dev libopencore-amrwb-dev \ 
		libopenblas-dev libatlas-base-dev libblas-dev \ 
		liblapack-dev libeigen3-dev gfortran \ 
		libhdf5-dev protobuf-compiler \ 
		libprotobuf-dev libgoogle-glog-dev libgflags-dev \ 
	&& rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Opencv

WORKDIR ${HOME}
# Build OpenCV
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv_contrib-${OPENCV_VERSION} OpenCV/opencv_contrib

#
# restore missing cuDNN headers
#
#RUN ln -s /usr/include/aarch64-linux-gnu/cudnn_v8.h /usr/include/cudnn.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_version_v8.h /usr/include/cudnn_version.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_backend_v8.h /usr/include/cudnn_backend.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_adv_infer_v8.h /usr/include/cudnn_adv_infer.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_adv_train_v8.h /usr/include/cudnn_adv_train.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_cnn_infer_v8.h /usr/include/cudnn_cnn_infer.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_cnn_train_v8.h /usr/include/cudnn_cnn_train.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_ops_infer_v8.h /usr/include/cudnn_ops_infer.h && \
# 	ln -s /usr/include/aarch64-linux-gnu/cudnn_ops_train_v8.h /usr/include/cudnn_ops_train.h && \
# 	ls -ll /usr/include/cudnn*

#COPY cudnn-10.2-linux-x64-v8.2.0.53.tgz .
#RUN tar -xzvf cudnn-10.2-linux-x64-v8.2.0.53.tgz && \
#	cp cuda/include/cudnn*.h /usr/local/cuda/include && \
#	cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
#	chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
	
WORKDIR ${HOME}/OpenCV/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr \
	-D OPENCV_EXTRA_MODULES_PATH=${HOME}/OpenCV/opencv_contrib/modules \
	-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
	-D WITH_OPENCL=OFF \
	-D WITH_CUDA=ON \
	-D CUDA_ARCH_BIN=7.2 \
	-D CUDA_ARCH_PTX="" \
	-D WITH_CUDNN=ON \
	-D WITH_CUBLAS=ON \
	-D ENABLE_FAST_MATH=ON \
	-D CUDA_FAST_MATH=ON \
	-D OPENCV_DNN_CUDA=ON \
	-D ENABLE_NEON=ON \
	-D WITH_QT=OFF \
	-D WITH_OPENMP=ON \
	-D WITH_OPENGL=ON \
	-D BUILD_TIFF=ON \
	-D WITH_FFMPEG=ON \
	-D WITH_GSTREAMER=ON \
	-D WITH_TBB=ON \
	-D BUILD_TBB=ON \
	-D BUILD_TESTS=OFF \
	-D WITH_EIGEN=ON \
	-D WITH_V4L=ON \
	-D WITH_LIBV4L=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D INSTALL_C_EXAMPLES=OFF \
	-D INSTALL_PYTHON_EXAMPLES=OFF \
	-D BUILD_opencv_python3=TRUE \
	-D OPENCV_GENERATE_PKGCONFIG=ON \
	-D BUILD_EXAMPLES=OFF .. && \
	make -j$(nproc) && \
    make install && \
    ldconfig && \
    apt-get autoremove

RUN rm -rf ${HOME}/OpenCV && \
    rm -rf /var/lib/apt/lists/*

RUN export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include && \
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install pycuda --verbose
RUN pip3 install scipy --verbose
RUN pip3 install numpy==1.19.4 --verbose

WORKDIR ${HOME}/lld
COPY lld.zip .
RUN unzip lld.zip && \
    unzip videos.zip && \
    rm lld.zip && \
    rm videos.zip


