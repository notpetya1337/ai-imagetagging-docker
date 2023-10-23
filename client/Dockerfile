FROM python:3.10-slim-bullseye
LABEL org.opencontainers.image.source="https://github.com/notpetya1337/ai-imagetagging-docker/"
LABEL authors="notpetya1337"

# install dependencies
RUN apt-get update
# maybe download ffmpeg and exiftool from their websites if I need the newest version
RUN apt-get -qq install ffmpeg exiftool

COPY client/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY config/default-config.ini /app/config/config.ini
COPY client/client.py /app/client.py
COPY dependencies /app/dependencies

ENTRYPOINT ["python3", "client.py"]
