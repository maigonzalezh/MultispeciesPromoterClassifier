FROM nvcr.io/nvidia/tensorflow:23.03-tf2-py3

ARG WANDB_API_KEY
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

WORKDIR /app
RUN mkdir /app/temp/

COPY ./docker/tensorflow/requirements.txt /app/requirements.txt

ENV PYTHONPATH /app:$PYTHONPATH
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN pip3 install -r requirements.txt

WORKDIR /opt

#### get necessary packages
RUN apt-get -y update && apt-get install -y wget build-essential zlib1g zlib1g-dev


#### download and compile cd-hit 4.8.1
RUN wget https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz && \
    tar xvf cd-hit-v4.8.1-2019-0228.tar.gz && \
    mv cd-hit-v4.8.1-2019-0228 cd-hit && \
    cd /opt/cd-hit && \
    make && \
    cd /opt/cd-hit/cd-hit-auxtools && \
    make 


#### set system PATH
ENV PATH="/opt/cd-hit:/opt/cd-hit/cd-hit-auxtools:/opt/cd-hit/psi-cd-hit:${PATH}"

WORKDIR /app
