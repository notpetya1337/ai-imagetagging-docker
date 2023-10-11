FROM python:3.11-slim-bullseye
LABEL authors="notpetya"


## download deepb model for now,
## cd to /app, copy from dependencies or entire folder, pip install, set entrypoint

# install dependencies  --no-cache
RUN apt-get update
RUN apt-get -y install ffmpeg exiftool

COPY server/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
# temporarily copy in config.ini
#COPY config.ini ./config.ini
COPY server/server.py ./server.py
COPY dependencies ./dependencies

ENTRYPOINT ["python3", "server.py"]
