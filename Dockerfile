FROM nvcr.io/nvidia/tensorrt:21.04-py3

RUN ln -snf /usr/share/zoneinfo/Europe/London /etc/localtime && echo Europe/London > /etc/timezone
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install jupyter
RUN python3 -m pip install --upgrade pip
RUN pip3 install pandas opencv-python matplotlib sklearn tqdm cython pika
RUN pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git
RUN pip3 uninstall -y keras-nightly
RUN pip3 uninstall -y tensorboard
RUN pip3 uninstall tensorflow
RUN pip3 uninstall tensorflow-gpu
RUN pip3 uninstall tensorflow-estimator

RUN pip3 install keras
RUN pip3 install tensorflow
RUN pip3 install tensorboard==2.4.1
RUN pip3 install -U tensorboard-plugin-profile

EXPOSE 8888
EXPOSE 6006
WORKDIR /workspace
