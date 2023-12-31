FROM python:3.10-slim-bullseye
LABEL org.opencontainers.image.source="https://github.com/notpetya1337/ai-imagetagging-docker/"
LABEL authors="notpetya1337"

COPY server/requirements.txt server/paddlerequirements.txt ./
RUN --mount=type=cache,target=/root/.cache pip3 install --no-deps -r paddlerequirements.txt
RUN --mount=type=cache,target=/root/.cache pip3 install -r requirements.txt

RUN apt-get update && apt-get install --no-install-recommends wget unzip xz-utils libgomp1 libgl1 libglib2.0-0 -qq

RUN apt-get install perl make -qq && wget -q https://exiftool.org/Image-ExifTool-12.68.tar.gz && tar -xzf Image-ExifTool-12.68.tar.gz \
    && rm Image-ExifTool-12.68.tar.gz && cd Image-ExifTool-12.68 && perl Makefile.PL && make install && cd ..  \
    && rm -rf Image-ExifTool-12.68 && apt-get remove perl make -qq

RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && tar -xJf ffmpeg-release-amd64-static.tar.xz \
    && rm ffmpeg-release-amd64-static.tar.xz && mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe && rm -rf ffmpeg-*-amd64-static

WORKDIR /models
RUN wget -q https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip -O model.zip && unzip model.zip -d ./ && rm model.zip
#COPY model.zip ./model.zip
#RUN unzip model.zip -d ./ && rm model.zip

# run paddleocr once to download models
RUN paddleocr --use_gpu=False --image_dir https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg --lang en --show_log False && rm tmp.jpg

WORKDIR /app
COPY config/default-config.ini /app/config/config.ini
COPY server/server.py /app/server.py
COPY dependencies /app/dependencies

ENTRYPOINT ["python3", "server.py"]
