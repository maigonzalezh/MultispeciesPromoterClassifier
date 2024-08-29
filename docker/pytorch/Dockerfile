FROM nvcr.io/nvidia/pytorch:23.03-py3

ARG WANDB_API_KEY
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

WORKDIR /app

COPY ./docker/pytorch/requirements.txt /app/requirements.txt

ENV PYTHONPATH /app:$PYTHONPATH
ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

RUN pip3 install -r requirements.txt

# uninstall triton, incompatible with RTX4000 gpu series
RUN pip3 uninstall triton -y