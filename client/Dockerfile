FROM python:3.10-slim-bullseye
LABEL org.opencontainers.image.source="https://github.com/notpetya1337/ai-imagetagging-docker/"
LABEL authors="notpetya1337"

COPY client/requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# The version of exiftool in the debian repos is too old to support WEBP files
RUN apt-get update && apt-get install --no-install-recommends xz-utils make perl wget -qq

RUN wget -q https://exiftool.org/Image-ExifTool-12.68.tar.gz && tar -xzf Image-ExifTool-12.68.tar.gz \
    && rm Image-ExifTool-12.68.tar.gz && cd Image-ExifTool-12.68 && perl Makefile.PL && make install && cd ..  \
    && rm -rf Image-ExifTool-12.68

RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && tar -xJf ffmpeg-release-amd64-static.tar.xz \
    && rm ffmpeg-release-amd64-static.tar.xz && mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe && rm -rf ffmpeg-*-amd64-static

RUN apt-get remove make perl wget -qq

WORKDIR /app
COPY config/default-config.ini /app/config/config.ini
COPY client/client.py /app/client.py
COPY dependencies /app/dependencies

ENTRYPOINT ["python3", "client.py"]
