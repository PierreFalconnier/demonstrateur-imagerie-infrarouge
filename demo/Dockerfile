FROM pytorchlightning/pytorch_lightning:latest-py3.6-torch1.9
LABEL maintainer="Christophe Karam <christophe.karam@grenoble-inp.org>"

ENV DEBIAN_FRONTEND=noninteractive

# Set username and hostname (machine name)
# should be the same as the ones that generated the license
ARG USER
ARG HOST
ARG WHEEL

# Set user
RUN useradd -ms /bin/bash $USER
RUN echo "root:Docker!" | chpasswd
ENV shell=/bin/bash
ENV HOME=/home/$USER
ENV PATH=$HOME/.local/bin:$PATH

# Install Python3.6 and other system dependencies
USER root
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
        libusb-1.0-0 libdbus-1.3 \
        libgl1-mesa-glx libglib2.0-0 \
	libopencv-dev libopencv-highgui-dev \
        wget curl iputils-ping software-properties-common
 
RUN apt-get update \
  && apt-get -y install sudo python3-tk libopencv-dev \
  && pip install --no-cache-dir pyparsing pydot \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
#Installation des lib python
COPY Requirements.txt Requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools && pip3 install -r Requirements.txt



# Install Lynred-SDK
COPY wheel/"$WHEEL" /tmp/"$WHEEL"
RUN python3.6 -m pip install /tmp/"$WHEEL"

# Copy license file
COPY licenses/licence_root_"$HOST".xml /home/"$USER"/.config/license.xml

WORKDIR /workspace


ENTRYPOINT ["python3.6", "yolov5-face-master/camera.py"]

