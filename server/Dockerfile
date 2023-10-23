FROM python:3.11-slim-bullseye
LABEL authors="notpetya"

## download deepb model for now,
## cd to /app, copy from dependencies or entire folder, pip install, set entrypoint

# install dependencies  --no-cache
RUN apt-get update
RUN apt-get -qq install ffmpeg exiftool wget zip

COPY server/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /models
RUN wget -q https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip -O model.zip && unzip model.zip -d ./ && rm model.zip
#COPY model.zip ./model.zip
#RUN unzip model.zip -d ./ && rm model.zip

WORKDIR /app
COPY config/default-config.ini /app/config/config.ini
COPY server/server.py /app/server.py
COPY dependencies /app/dependencies

ENTRYPOINT ["python3", "server.py"]
