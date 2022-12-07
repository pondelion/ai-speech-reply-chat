FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    software-properties-common tzdata git cmake \
    &&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    &&  apt-get clean \
    &&  rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3.7 python3.7-distutils python3-pip python3.7-dev libpng-dev libfreetype-dev
RUN python3.7 -m pip install -U pip wheel setuptools
RUN pip install --upgrade pip
RUN pip install packaging
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
COPY tacotron2 /tacotron2
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" tacotron2/apex
# RUN pip install --no-cache-dir -r tacotron2/requirements.txt
RUN pip install --no-cache-dir matplotlib tensorflow==1.15.2 numpy inflect==0.2.5 librosa==0.6.3 scipy Unidecode==1.0.22 pillow
RUN apt-get -y install libsndfile1
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U protobuf==3.20.*
RUN pip install -U numba==0.48 resampy==0.3.1
COPY scripts /scripts
RUN python3.7 scripts/initial_model_preparation.py