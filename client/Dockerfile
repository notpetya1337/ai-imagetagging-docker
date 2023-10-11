FROM python:3.11-slim-bullseye
LABEL authors="notpetya1337"
## cd to /app, copy from dependencies or entire folder, pip install, set entrypoint

# install dependencies
RUN apt-get update
# maybe download ffmpeg and exiftool if I need the newest version
RUN apt-get -y install ffmpeg exiftool

COPY client/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
# temporarily copy in config.ini
#COPY config.ini ./config.ini
COPY client/client.py ./client.py
COPY dependencies ./dependencies

ENTRYPOINT ["python3", "client.py"]
