FROM nvcr.io/nvidia/tensorrt:21.04-py3

EXPOSE 8888
EXPOSE 6006

WORKDIR /workspace

RUN ln -snf /usr/share/zoneinfo/Europe/London /etc/localtime && echo Europe/London > /etc/timezone \
    && apt-get update ##[edited] \
    && apt-get install -y ffmpeg libsm6 libxext6

RUN python3 -m pip install --upgrade pip \
    && pip3 install jupyter pandas opencv-python matplotlib sklearn tqdm cython git+https://github.com/lucasb-eyer/pydensecrf.git \
    && pip3 install --upgrade --force-reinstall tensorflow-gpu keras tensorboard==2.4.1 tensorboard-plugin-profile
