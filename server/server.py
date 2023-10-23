import json
import logging
import os
import sys
import threading

import exiftool
import pymongo
from redis import Redis

from dependencies.configops import MainConfig
from dependencies.fileops import (get_image_content, get_image_md5, get_video_content, get_video_content_md5, )
from dependencies.vision import Tagging
from dependencies.vision_video import VideoData

# read config
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

# Initialize models
if "deepb" in config.configmodels:
    import dependencies.deepb as deepb

    modelpath = config.deepbmodelpath
    tagfile = config.deepbtagfile
    threshold = config.deepbthreshold
    deepb_tagger = deepb.deepdanbooruModel(threshold, modelpath, tagfile)

if "paddleocr" in config.configmodels:
    import paddleocr
    ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang="en")

# TODO: find out if I need to spawn several of these for multithreading.
et = exiftool.ExifToolHelper(logger=logging.getLogger(__name__).setLevel(logging.INFO), encoding="utf-8", )
# Initialize variables
tagging = Tagging(config.google_credentials, config.google_project, tags_backend="google-vision")
imagecount = 0
videocount = 0
foldercount = 0
# TODO: add a rolling count on the same line with uptime, images, videos, folders
imagecount_lock = threading.Lock()
videocount_lock = threading.Lock()
# Names of likelihood from google.cloud.vision.enums
likelihood_name = ("UNKNOWN", "Very unlikely", "Unlikely", "Possible", "Likely", "Very likely",)

# initialize logger
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.debug("logging started")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REDIS_CLIENT = Redis(host=config.redishost, port=config.redisport, db=0)


def pull(key):
    return REDIS_CLIENT.brpop(key)


def recognize_image(imagepath, workingcollection, subdiv, is_screenshot, models):
    # TODO: handle update/create in redis instead of checking, support getting explicit tags
    im_md5 = get_image_md5(imagepath)
    image_content = get_image_content(imagepath)
    imagepath_array = [imagepath]
    # Check if entry exists in MongoDB, then create new entry or update entry by MD5
    entry = collection.find_one({"md5": im_md5},
                                {"md5": 1, "vision_tags": 1, "vision_text": 1, "deepbtags": 1, "explicit_detection": 1})
    if entry is None:
        mongo_entry = create_imagedoc(image_content, im_md5, imagepath_array, is_screenshot, subdiv, models)
        workingcollection.insert_one(mongo_entry)
        logger.info("Added new entry in MongoDB for image %s: %s\n", imagepath, mongo_entry)
    else:
        # Make sure tagging doesn't run twice
        if "deepb" in models and (entry.get("deepbtags") is None or len(entry.get("deepbtags")) == 0):
            logger.info("Processing DeepB tags for image %s", imagepath)
            deepbtags = deepb_tagger.classify_image(imagepath)
            collection.update_one({"md5": im_md5}, {"$set": {"deepbtags": deepbtags[1]}})
        if "vision" in models and entry.get("vision_tags") is None:
            # TODO: always check MongoDB before processing with Vision because of expense
            logger.warning("Not processing vision tags yet")  # collection.update_one()
        if "paddleocr" in models and entry.get("paddleocrtext") is None:
            logger.info("Processing PaddleOCR text for image %s", imagepath)
            result = ocr.ocr(imagepath, cls=True)
            ocrtext = ""
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        ocrtext += line[1][0] + " "
            collection.update_one({"md5": im_md5}, {"$set": {"paddleocrtext": ocrtext}})


def create_imagedoc(image_content, im_md5, image_array, is_screenshot, subdiv, models):
    print("Processing:", im_md5, image_array, is_screenshot, subdiv, models)
    tags, text, easytext, ocrtext, safe, deepbtags, explicit_detection = None, None, None, None, None, None, None
    if "deepb" in models and "deepb" not in config.configmodels:
        logger.error("Client requested DeepB tags but DeepB is disabled in config")
    if "deepb" in models and is_screenshot != 1:
        deepbtags = deepb_tagger.classify_image(image_array[0])
        deepbtags = deepbtags[1]
    if "vision" in models:
        text = tagging.get_text(image_binary=image_content)
        text = [text[0]]
        if is_screenshot != 1:
            tags = tagging.get_tags(image_binary=image_content)
            safe = tagging.get_explicit(image_binary=image_content)
            explicit_detection = {"adult": f"{likelihood_name[safe.adult]}",
                                  "medical": f"{likelihood_name[safe.medical]}",
                                  "spoofed": f"{likelihood_name[safe.spoof]}",
                                  "violence": f"{likelihood_name[safe.violence]}",
                                  "racy": f"{likelihood_name[safe.racy]}", }
            explicit_detection = [explicit_detection][0]
    if "paddleocr" in models:
        result = ocr.ocr(image_array[0], cls=True)
        for idx in range(len(result)):
            res = result[idx]
            if res is not None:
                ocrtext = ""
                for line in res:
                    ocrtext += line[1][0] + " "
    mongo_entry = {"md5": im_md5, "vision_tags": tags, "vision_text": text, "explicit_detection": explicit_detection,
                   "deepbtags": deepbtags, "easyocrtext": easytext, "paddleocrtext": ocrtext, "path": image_array,
                   "subdiv": subdiv, "is_screenshot": is_screenshot}
    logger.info("Generated MongoDB entry: %s", mongo_entry)
    return mongo_entry


def mongo_image_data(imagepath, workingcollection, models):
    """
    :param imagepath:
    :param workingcollection:
    :param models: str
        models to get tags from, e.g. "vision", "deepb", "explicit"
        only reads from MongoDB
    :return:
    """
    im_md5 = get_image_md5(imagepath)
    tagslist, textlist = [], []
    if "vision" in models:
        visiontext = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"vision_text": 1, "_id": 0})))
        visiontagsjson = json.loads(
            json.dumps(workingcollection.find_one({"md5": im_md5}, {"vision_tags": 1, "_id": 0})))
        if visiontagsjson is not None: tagslist.extend(visiontagsjson["vision_tags"])
        if visiontext is not None: textlist.extend(visiontext["vision_text"])
    if "deepb" in models:
        deepbtags = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"deepbtags": 1, "_id": 0})))
        if deepbtags is not None: tagslist.extend(deepbtags["deepbtags"])
    if "explicit" in models:
        explicit_mongo = workingcollection.find_one({"md5": im_md5}, {"explicit_detection": 1, "_id": 0})
        if explicit_mongo:
            detobj = explicit_mongo["explicit_detection"]
            explicit_results = (
                f"Adult: {detobj['adult']}", f"Medical: {detobj['medical']}", f"Spoofed: {detobj['spoofed']}",
                f"Violence: {detobj['violence']}", f"Racy: {detobj['racy']}")
            tagslist.extend(explicit_results)
    text = " ".join(textlist)
    if "paddleocr" in models:
        paddleocrtext = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"paddleocrtext": 1, "_id": 0})))
        if text == "" and paddleocrtext is not None:
            text = paddleocrtext["paddleocrtext"]
    text = (text[:config.maxlength] + " truncated...") if len(text) > config.maxlength else text
    text = text.replace("\n", " ")
    return tagslist, text


def write_image_exif(imagepath, workingcollection, models):
    tags, text = mongo_image_data(imagepath, workingcollection, models)
    if tags and text:
        try:
            # TODO: check back and see if -ec is necessary
            et.set_tags(imagepath, tags={"Subject": tags, "xmp:Title": text}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.warning('Error "%s " writing tags, Exiftool output was %s', e, et.last_stderr)
    elif tags:
        try:
            et.set_tags(imagepath, tags={"Subject": tags}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.warning('Error "%s " writing tags, Exiftool output was %s', e, et.last_stderr)


def process_video(videopath, workingcollection, subdiv, models, rootdir=""):
    video_content_md5 = str(get_video_content_md5(videopath))
    # TODO: update to check explicit_detection
    entry = videocollection.find_one({"content_md5": video_content_md5},
                                     {"content_md5": 1, "vision_tags": 1, "vision_text": 1, "vision_transcript": 1,
                                      "explicit_detection": 1})
    if "vision" in models and entry is None:
        relpath = os.path.relpath(videopath, rootdir)
        video_content = get_video_content(videopath)
        vidpath_array = [videopath]
        relpath_array = [relpath]
        mongo_entry = create_videodoc(video_content, video_content_md5, vidpath_array, relpath_array, subdiv)
        workingcollection.insert_one(mongo_entry)
        logger.info("Added new entry in MongoDB for video %s \n", videopath)
    else:
        logger.info("MongoDB entry for video %s already exists\n", videopath)


def create_videodoc(video_content, video_content_md5, vidpath_array, relpath_array, subdiv):
    videoobj = VideoData(config.google_credentials, config.google_project)
    videoobj.video_vision_all(video_content)
    mongo_entry = {"content_md5": video_content_md5, "vision_tags": videoobj.labels, "vision_text": videoobj.text,
                   "vision_transcript": videoobj.transcripts, "explicit_detection": videoobj.pornography,
                   "path": vidpath_array, "subdiv": subdiv, "relativepath": relpath_array, }
    return mongo_entry


while True:
    logger.info("Waiting for job")
    job = json.loads(pull("queue")[1])
    if job["type"] == "image":
        print("Processing image, job is", job)
        if job["op"] == "recognition":
            if job["subdiv"] == "screenshot" and job['models'] is not None:
                recognize_image(job["path"], screenshotcollection, job["subdiv"], job["is_screenshot"], job["models"], )
            elif job['models'] is not None:
                recognize_image(job["path"], collection, job["subdiv"], job["is_screenshot"], job["models"], )
        if job["op"] == "write_exif":
            print("Writing EXIF for", job["path"])
            write_image_exif(job["path"], collection, job["models"])

    elif job["type"] == "video":
        if job["op"] == "recognition" and job['models'] is not None:
            print("Processing video, job is", job)
            process_video(job["path"], videocollection, job["subdiv"], job["models"])
