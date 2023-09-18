FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt-get install wget git libglib2.0-0 libgl1-mesa-glx sudo nano zip ffmpeg libxcb-xinerama0 -y && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y

RUN apt-get install python3-opencv python3-pip python3-dev -y

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN chmod -R 777 .

RUN pip3 install -r requirements-cuda.txt 
