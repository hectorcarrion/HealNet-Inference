# syntax=docker/dockerfile:1

FROM python:3.8-slim

ADD healnet_inference.py .

ADD requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

CMD ["python", "./healnet_inference.py"]