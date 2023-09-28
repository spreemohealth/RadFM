# FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.07-py3
FROM python:3.10.12-slim

LABEL author="Will Deng"
LABEL author-email="will.deng@coverahealth.com"

WORKDIR /radfm/

COPY requirements.txt /radfm/

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENV MODEL_FOLDER="/mnt/team_s3_synced/msandora/RadFM/pytorch_model.bin"
ENV WITH_IMAGE="true"

ENV MODEL_TYPE="radfm"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# CMD ["python", "app.py"]

COPY ./ /radfm/

WORKDIR /radfm/Quick_demo/
