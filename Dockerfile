# FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.07-py3
# FROM nvcr.io/nvidia/pytorch:23.07-py3
FROM python:3.8

ARG GITHUB_TOKEN

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /radfm/

COPY requirements.txt /radfm/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install git+https://$GITHUB_TOKEN@github.com/spreemohealth/cvtools_augmentations@v1.0.0
RUN pip install git+https://$GITHUB_TOKEN@github.com/spreemohealth/MhdHelpers

RUN pip install --no-binary :all: nmslib

COPY ./ /radfm/

WORKDIR /radfm/src/

RUN wandb login 5bbb91ca10e5f4957e9bde0eb6859b70cfb53c62
