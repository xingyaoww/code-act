FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Timezone set to US
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python 3.9 and pip
RUN apt-get update -y && \
    apt-get install -y \
    jq \
    git \
    cmake \
    python3-pip \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN apt update -y
RUN apt install -y \
    wget \
    curl

# symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install project dependency
COPY requirements.txt .
# RUN pip install \
#     torch==1.13.1+cu117 \
#     torchvision==0.14.1+cu117 \
#     torchaudio==0.13.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu117
# RUN git clone https://github.com/sachit-menon/GLIP.git && \
#     cd GLIP && \
#     python setup.py build develop && \
#     cd ..
RUN pip3 install -r requirements.txt
RUN pip3 install Pillow==9.5.0
RUN pip3 install pip install --upgrade numpy
# Command to build the image:
# docker build -t xingyaoww/mint-dev:v1.0 .
