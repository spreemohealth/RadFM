# FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.07-py3
FROM python:3.10.12

LABEL author="Will Deng"
LABEL author-email="will.deng@coverahealth.com"

WORKDIR /radfm/

COPY requirements.txt /radfm/

RUN pip install --upgrade pip
# RUN pip install pip==20.0.2
RUN python -m pip cache purge
RUN pip install --no-cache-dir --upgrade -r requirements.txt
# RUN pip install torch
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# ENV WITH_IMAGE="true"

# ENV MODEL_TYPE="radfm"
# ENV BATCH_SIZE = 1
ENV LOG_FILE = "onprem.log"
# EXPOSE 8501

# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["python", "app.py"]

COPY ./ /radfm/
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# WORKDIR /radfm/Quick_demo/
WORKDIR /radfm/src/
