FROM python:3.10-slim-bullseye
LABEL org.opencontainers.image.source="https://github.com/notpetya1337/ai-imagetagging-docker/"
LABEL authors="notpetya1337"

# install dependencies
RUN apt-get update && apt-get -qq install ffmpeg

COPY client/requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# The version of exiftool in the debian repos is too old to support WEBP files
RUN apt-get install make perl wget -qq && wget -q https://exiftool.org/Image-ExifTool-12.68.tar.gz && tar -xzf Image-ExifTool-12.68.tar.gz \
    && rm Image-ExifTool-12.68.tar.gz && cd Image-ExifTool-12.68 && perl Makefile.PL && make install && cd ..  \
    && rm -rf Image-ExifTool-12.68 && apt-get remove make perl -qq

WORKDIR /app
COPY config/default-config.ini /app/config/config.ini
COPY client/client.py /app/client.py
COPY dependencies /app/dependencies

ENTRYPOINT ["python3", "client.py"]

