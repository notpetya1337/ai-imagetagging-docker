import concurrent.futures
import datetime
import json
import logging
import os
import sys
import threading
import time

import pymongo
from redis import Redis

from dependencies.configops import MainConfig
from dependencies.fileops import get_image_md5, get_video_content_md5, listdirs, listimages, listvideos

# initialize logger
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# read config
# TODO: read this path from an environment variable passed through docker-compose
configpath = "/app/config/config.ini"
if not os.path.isfile(configpath):
    print(f"Config file {configpath} not found, exiting")
    exit(1)
config = MainConfig(configpath)

# initialize DBs
currentdb = pymongo.MongoClient(config.connectstring)[config.mongodbname]
collection = currentdb[config.mongocollection]
screenshotcollection = currentdb[config.mongoscreenshotcollection]
videocollection = currentdb[config.mongovideocollection]
collection.create_index([("md5", pymongo.TEXT)], name="md5_index", unique=True)
collection.create_index("vision_tags")
screenshotcollection.create_index([("md5", pymongo.TEXT)], name="md5_index", unique=True)
videocollection.create_index("content_md5")
videocollection.create_index("vision_tags")

# Initialize variables
imagecount, videocount, queuedimagecount, queuedvideocount, exifcount, exifvideocount, foldercount = 0, 0, 0, 0, 0, 0, 0
imagecount_lock, videocount_lock, queuedimagecount_lock, queuedvideocount_lock, exifcount_lock, exifvideocount_lock = (
    threading.Lock(), threading.Lock(), threading.Lock(), threading.Lock(), threading.Lock(), threading.Lock())

REDIS_CLIENT = Redis(host=config.redishost, port=config.redisport, db=0)

logger.info("Loading md5s from MongoDB")
allmd5s = set([x["md5"] for x in collection.find({}, {"md5": 1, "_id": 0})])
# possibly because of entryies with no content md5?
videomd5s = set()
for x in videocollection.find({"content_md5": {"$exists": True}}, {"content_md5": 1, "_id": 0}):
    if isinstance(x["content_md5"], list):
        for y in x["content_md5"]: videomd5s.add(y)
    else:
        videomd5s.add(x["content_md5"])
# TODO: fix these so entries with `null` aren't added to the set
deepbmd5s = set([x["md5"] for x in collection.find({"deepbtags": {"$exists": True, "$type": "array"}},
                                                   {"md5": 1, "_id": 0})])
paddlemd5s = set([x["md5"] for x in collection.find({"paddleocrtext": {"$exists": True, "$type": "string"}},
                                                    {"md5": 1, "_id": 0})])
visionmd5s = set([x["md5"] for x in collection.find({"$or": [{"vision_tags": {"$ne": None, "$exists": True}},
                                                             {"vision_text": {"$ne": None, "$exists": True}},
                                                             {"explicit_detection": {"$ne": None, "$exists": True}}]},
                                                    {"md5": 1, "_id": 0})])
visiontagmd5s = set([x["md5"] for x in collection.find({"vision_tags": {"$ne": None, "$exists": True}},
                                                       {"md5": 1, "_id": 0})])
visiontextmd5s = set([x["md5"] for x in collection.find({"vision_text": {"$ne": None, "$exists": True}},
                                                        {"md5": 1, "_id": 0})])
visionexplicitmd5s = set([x["md5"] for x in collection.find({"explicit_detection": {"$ne": None, "$exists": True}},
                                                            {"md5": 1, "_id": 0})])
logger.info("Loaded md5s from MongoDB")


def push(key, value):
    REDIS_CLIENT.lpush(key, value)


def process_image_folder(workingdir, is_screenshot, subdiv, models):
    global imagecount, imagecount_lock, queuedimagecount, queuedimagecount_lock, exifcount, exifcount_lock
    try:
        workingimages = listimages(workingdir, config.process_images)
    except FileNotFoundError as e:
        print(e)
        print("Folder not found. Check your config.ini and docker-compose folder mounts.")
        exit(1)
    # "Process only new" loop here
    if config.process_only_new:
        for imagepath in workingimages:
            with imagecount_lock:
                imagecount += 1
            process_models = []
            im_md5 = get_image_md5(imagepath)
            if im_md5 not in allmd5s:
                if subdiv in config.deepbdivs and "deepb" in config.configmodels: process_models.append(
                    "deepb"), deepbmd5s.add(im_md5)
                if "vision" in config.configmodels: process_models.append("vision"), visionmd5s.add(im_md5)
                if "paddleocr" in config.configmodels: process_models.append("paddleocr"), paddlemd5s.add(im_md5)
                if process_models:
                    push("queue", json.dumps(
                        {"type": 'image', "op": "recognition", "path": imagepath, "is_screenshot": is_screenshot,
                         "subdiv": subdiv, "models": process_models}))
                with queuedimagecount_lock:
                    queuedimagecount += 1
            print(f"Processed {imagecount} images with {queuedimagecount} new ", end="\r")  # "Process all" loop here
    elif not config.process_only_new:
        for imagepath in workingimages:
            with imagecount_lock:
                imagecount += 1
            process_models = []
            im_md5 = get_image_md5(imagepath)
            if "vision" in config.configmodels and im_md5 not in visiontagmd5s:
                process_models.append("visiontags"), visiontagmd5s.add(im_md5)
            if "vision" in config.configmodels and im_md5 not in visiontextmd5s:
                process_models.append("visiontext"), visiontextmd5s.add(im_md5)
            if "vision" in config.configmodels and im_md5 not in visionexplicitmd5s:
                process_models.append("explicit"), visionexplicitmd5s.add(im_md5)
            if "paddleocr" in config.configmodels and im_md5 not in paddlemd5s:
                process_models.append("paddleocr"), paddlemd5s.add(im_md5)
            if subdiv in config.deepbdivs and im_md5 not in deepbmd5s: process_models.append("deepb"), deepbmd5s.add(
                im_md5)
            if len(process_models) > 0:
                push("queue", json.dumps({"type": 'image', "op": "recognition", "path": imagepath,
                                          "is_screenshot": is_screenshot, "subdiv": subdiv, "models": process_models}))
                with queuedimagecount_lock: queuedimagecount += 1
            print(f"Processed {imagecount} images with {queuedimagecount} updated ", end="\r")
    print("")

    if config.write_exif:
        for imagepath in workingimages:
            with exifcount_lock: exifcount += 1
            push("queue", json.dumps({"type": 'image', "op": "write_exif", "path": imagepath, "subdiv": subdiv,
                                      "models": models}))
            print(f"Queued EXIF writing for {exifcount} images ", end="\r")
    print("")


def process_video_folder(workingdir, subdiv):
    rootdir = config.getdiv(subdiv)
    global videocount, videocount_lock, queuedvideocount, queuedvideocount_lock, exifvideocount, exifvideocount_lock
    workingvideos = listvideos(workingdir, config.process_videos)
    # Process only new video loop here
    if config.process_only_new:
        for videopath in workingvideos:
            process_models = []
            with videocount_lock:
                videocount += 1
            vid_md5 = get_video_content_md5(videopath)
            if vid_md5 not in videomd5s:
                if "vision" in config.configmodels: process_models.append("vision"), videomd5s.add(vid_md5)
            if process_models:
                with queuedvideocount_lock: queuedvideocount += 1
                push("queue", json.dumps({"type": 'video', "op": "recognition", "path": videopath, "subdiv": subdiv,
                                          "models": process_models}))
            print(f'Processed {videocount} videos with {queuedvideocount} new', end="\r")
    # Process all videos loop here
    elif not config.process_only_new:
        for videopath in workingvideos:
            process_models = []
            with videocount_lock:
                videocount += 1
            vid_md5 = get_video_content_md5(videopath)
            if "vision" in config.configmodels and vid_md5 not in visionmd5s: (process_models.append("vision"),
                                                                               visionmd5s.add(vid_md5))
            if "vision" in config.configmodels and vid_md5 not in visionexplicitmd5s: process_models.append(
                "explicit"), visionexplicitmd5s.add(vid_md5)
            if subdiv in config.deepbdivs and vid_md5 not in deepbmd5s: (process_models.append("deepb"),
                                                                         deepbmd5s.add(vid_md5))
            if process_models:
                logger.info("Processing video %s", videopath)
                push("queue", json.dumps({"type": 'video', "op": "recognition", "path": videopath, "subdiv": subdiv,
                                          "models": process_models}))
            print(f'Processed {videocount} videos with {queuedvideocount} new', end="\r")
    print("")

    if config.write_exif:
        for videopath in workingvideos:
            with exifvideocount_lock: exifvideocount += 1
            # TODO: add configvideomodels
            push("queue", json.dumps({"type": 'video', "op": "write_exif", "path": videopath, "subdiv": subdiv,
                                      "models": config.configmodels}))
            print(f"Queued EXIF writing for {exifvideocount} videos ", end="\r")
    print("")


def main():
    global foldercount, imagecount, videocount, queuedimagecount, queuedvideocount, exifcount
    foldercount, imagecount, videocount, queuedimagecount, queuedvideocount, exifcount = 0, 0, 0, 0, 0, 0
    start_time = time.time()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.threads)
    for div in config.subdivs:
        rootdir = config.getdiv(div)
        allfolders = listdirs(rootdir)
        while allfolders:
            workingdir = allfolders.pop(0)
            foldercount += 1
            # TODO: limit number of threads
            if config.process_images:
                if workingdir.lower().find("screenshot") != -1:
                    is_screenshot = 1
                else:
                    is_screenshot = 0
                models = config.configmodels
                if div in config.deepbdivs and "deepb" in config.configmodels and "deepb" not in models: models.append("deepb")
                pool.submit(process_image_folder(workingdir, is_screenshot, div, models))
            if config.process_videos:
                pool.submit(process_video_folder(workingdir, div))

    # Wait until all threads exit
    pool.shutdown(wait=True)
    elapsed_time = time.time() - start_time
    final_time = str(datetime.timedelta(seconds=elapsed_time))
    logger.info("All entries processed. Root divs: %s, Folder count: %s, Image count: %s, Video count: %s",
                config.subdivs, foldercount, imagecount, videocount)
    print(queuedimagecount, "images and ", queuedvideocount, "videos queued.")
    print("Processing took ", final_time)


if __name__ == "__main__":
    while True:
        main()
        if config.sleeptime == 0:
            logger.info("Only running once.")
            exit(0)
        time.sleep(config.sleeptime)
