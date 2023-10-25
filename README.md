# ai-imagetagging-docker

Project to run AI image tagging and OCR on a folder. \
Uncomment #- ./config/:/app/config/ in docker-compose.yml to use your own config file (see config/default-config.yml for an example) 

Supports PaddleOCR and Google Vision for OCR and Google Vision for image tagging. \
DeepDanbooru is also supported for anime-style images.

TODO: add support for a self-hosted object detection model, add EXIF tag writing.
