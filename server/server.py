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
import concurrent.futures

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

if "selfhosted-tags" in config.configmodels:
    from transformers import pipeline
    from PIL import Image
    classifier = pipeline("image-classification", "google/vit-base-patch16-224")


# TODO: find out if I need to spawn several of these for multithreading.
# et = exiftool.ExifToolHelper(logger=logging.getLogger(__name__).setLevel(logging.INFO), encoding="utf-8")
# Initialize variables
tagging = Tagging(config.google_credentials, config.google_project, tags_backend="google-vision")
imagecount, videocount, foldercount = 0, 0, 0
# TODO: add a rolling count on the same line with uptime, images, videos, folders. Might not be possible with Docker.
imagecount_lock, videocount_lock = threading.Lock(), threading.Lock()
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
    """
    Create or update the MongoDB entry for a given image
    :param imagepath: path to image
    :param workingcollection: PyMongo collection to write to
    :param subdiv: subdiv that the image folder is classified as
    :param is_screenshot: 1 if image is a screenshot, 0 if not
    :param models: AI models to use for tagging
    :return:
    """
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
        if "visiontags" in models and entry.get("vision_tags") is None:
            # always checking MongoDB before processing with Vision because of expense
            # Client may create several jobs for the same image, so we're checking Mongo on every run of this function
            if workingcollection.find_one({"md5": im_md5}, {"vision_tags": 1, "_id": 0})['vision_tags'] is None:
                logger.info("Processing Vision tags for image %s", imagepath)
                tags = tagging.get_tags(image_binary=image_content)
                collection.update_one({"md5": im_md5}, {"$set": {"vision_tags": tags}})
        if "visiontext" in models and entry.get("vision_text") is None:
            if workingcollection.find_one({"md5": im_md5}, {"vision_text": 1, "_id": 0})['vision_text'] is None:
                logger.info("Processing Vision text for image %s", imagepath)
                text = tagging.get_text(image_binary=image_content)
                collection.update_one({"md5": im_md5}, {"$set": {"vision_text": text[0]}})
        if "explicit" in models and entry.get("explicit_detection") is None:
            try:
                if workingcollection.find_one({"md5": im_md5}, {"explicit_detection": 1, "_id": 0})['explicit_detection'] is None:
                    flag = True
            except KeyError: flag = True  # in case there's no explicit_detection key
            else: flag = False
            if flag:
                logger.info("Processing Vision explicit detection for image %s", imagepath)
                safe = tagging.get_explicit(image_binary=image_content)
                explicit_detection = {"adult": f"{likelihood_name[safe.adult]}",
                                      "medical": f"{likelihood_name[safe.medical]}",
                                      "spoofed": f"{likelihood_name[safe.spoof]}",
                                      "violence": f"{likelihood_name[safe.violence]}",
                                      "racy": f"{likelihood_name[safe.racy]}"}
                collection.update_one({"md5": im_md5}, {"$set": {"explicit_detection": explicit_detection}})
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


def recognize_video(videopath, workingcollection, subdiv, models, rootdir=""):
    video_content_md5 = str(get_video_content_md5(videopath))
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
    elif "vision" in models and entry is not None:
        # TODO: implement this
        logger.info("Not updating video entries yet")


def create_imagedoc(image_content, im_md5, image_array, is_screenshot, subdiv, models):
    """
    Create a properly formatted MongoDB entry for an image that isn't already in the database
    :param image_content: raw image binary
    :param im_md5: MD5 of image
    :param image_array: image path in array format
    :param is_screenshot: 1 if image is a screenshot, 0 if not
    :param subdiv: subdiv that the image folder is classified as
    :param models: AI models to use for tagging
    :return:
    """
    logger.info("Processing: %s %s %s %s", im_md5, image_array, subdiv, models)
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


def create_videodoc(video_content, video_content_md5, vidpath_array, relpath_array, subdiv):
    videoobj = VideoData(config.google_credentials, config.google_project)
    videoobj.video_vision_all(video_content)
    mongo_entry = {"content_md5": video_content_md5, "vision_tags": videoobj.labels, "vision_text": videoobj.text,
                   "vision_transcript": videoobj.transcripts, "explicit_detection": videoobj.pornography,
                   "path": vidpath_array, "subdiv": subdiv, "relativepath": relpath_array, }
    return mongo_entry


def mongo_image_data(imagepath, workingcollection, models):
    """
    :param imagepath:
    :param workingcollection:
    :param models: str
        models to get tags from, e.g. "vision", "deepb", "explicit"
        only reads from MongoDB
    :return: list of tags from MongoDB, string of text from MongoDB
    """
    im_md5 = get_image_md5(imagepath)
    tagslist, textlist, text = [], [], ""
    if "vision" in models:
        visiontext = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"vision_text": 1, "_id": 0})))
        visiontags = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"vision_tags": 1, "_id": 0})))
        if visiontags and visiontags["vision_tags"] is not None: tagslist.extend(visiontags["vision_tags"])
        if visiontext and visiontext["vision_text"] is not None: textlist = visiontext["vision_text"]
    if "deepb" in models:
        deepbtags = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"deepbtags": 1, "_id": 0})))
        if deepbtags and deepbtags["deepbtags"] is not None: tagslist.extend(deepbtags["deepbtags"])
    if "explicit" in models:
        explicit_mongo = workingcollection.find_one({"md5": im_md5}, {"explicit_detection": 1, "_id": 0})
        if explicit_mongo:
            detobj = explicit_mongo["explicit_detection"]
            explicit_results = (f"Adult: {detobj['adult']}", f"Medical: {detobj['medical']}",
                                f"Spoofed: {detobj['spoofed']}", f"Violence: {detobj['violence']}",
                                f"Racy: {detobj['racy']}")
            tagslist.extend(explicit_results)
    if isinstance(textlist, str):
        text = " ".join(textlist.splitlines())
    elif isinstance(textlist, list) and textlist:
        text = " ".join(textlist[0].splitlines())  # + " "
    if "paddleocr" in models:
        paddleocrtext = json.loads(json.dumps(workingcollection.find_one({"md5": im_md5}, {"paddleocrtext": 1, "_id": 0})))
        if len(text) == 0 and paddleocrtext and (paddleocrtext["paddleocrtext"] is not None):
            text = paddleocrtext["paddleocrtext"]
    text = (text[:config.maxlength] + " truncated...") if len(text) > config.maxlength else text
    logger.info("Text is %s", text)
    return tagslist, text


def mongo_video_data(videopath, workingcollection, models):
    """
    :param videopath: full path to video file
    :param workingcollection: MongoDB collection to read from
    :param models: str
        models to get tags from, e.g. "vision"
    :return:
    """
    tagslist, textlist = [], []
    video_content_md5 = str(get_video_content_md5(videopath))
    if "vision" in models:
        visiontagsjson = json.loads(json.dumps(videocollection.find_one({"content_md5": video_content_md5}, {"vision_tags": 1, "_id": 0})))
        if visiontagsjson and visiontagsjson["vision_tags"] is not None: tagslist.extend(visiontagsjson["vision_tags"])
        explicitagsjson = json.loads(json.dumps(videocollection.find_one({"content_md5": video_content_md5}, {"explicit_detection": 1, "_id": 0})))
        if explicitagsjson and explicitagsjson["explicit_detection"] is not None: tagslist.extend(explicitagsjson["explicit_detection"])
        visiontranscript = json.loads(json.dumps(videocollection.find_one({"content_md5": video_content_md5}, {"vision_transcript": 1, "_id": 0})))
        if visiontranscript and visiontranscript["vision_transcript"] is not None: textlist.extend(visiontranscript["vision_transcript"])
        visiontext = json.loads(json.dumps(videocollection.find_one({"content_md5": video_content_md5}, {"vision_text": 1, "_id": 0})))
        if visiontext and visiontext["vision_text"] is not None: textlist.extend(visiontext["vision_text"])
        text = (" ".join(textlist)).replace("\n", " ")
        text = (text[:config.maxlength] + " truncated...") if len(text) > config.maxlength else text
        return tagslist, text
    else:
        logger.warning("No models specified for video %s", videopath)
        return [], []


def write_image_exif(imagepath, workingcollection, models, et):
    try:
        tags, text = mongo_image_data(imagepath, workingcollection, models)
    except pymongo.errors.ConnectionFailure as e:
        logger.warning("Error connecting to MongoDB: %s", e)
        retry
    if tags and text:
        try:
            # TODO: don't commit the logging line
            logger.warning("Writing text %s to %s", text, imagepath)
            et.set_tags(imagepath, tags={"Subject": tags, "xmp:Title": text}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.warning('Error "%s " writing tags, Exiftool output was %s', e, et.last_stderr)
    elif tags:
        try:
            et.set_tags(imagepath, tags={"Subject": tags}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.warning('Error "%s " writing tags, Exiftool output was %s', e, et.last_stderr)


def write_video_exif(videopath, workingcollection, models, et):
    # TODO: implement retrying
    tags, text = mongo_video_data(videopath, workingcollection, models)
    if tags:
        try:
            et.set_tags(videopath, tags={"Subject": tags}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.warning('Error "%s " writing tags, Exiftool output was %s', e, et.last_stderr)
    if text:
        try:
            et.set_tags(videopath, tags={"Title": text}, params=["-P", "-overwrite_original"])
        except exiftool.exceptions.ExifToolExecuteError as e:
            logger.warning('Error "%s " writing tags, Exiftool output was %s', e, et.last_stderr)


def exifprocessor():
    while True:
        et = exiftool.ExifToolHelper(logger=logging.getLogger(__name__).setLevel(logging.INFO), encoding="utf-8")
        logger.info("Waiting for job")
        job = json.loads(pull("queue")[1])
        if job["type"] == "image":
            logger.info("Processing image, job is %s", job)
            if job["op"] == "write_exif":
                logger.info("Writing EXIF for %s with models %s", job["path"], job["models"])
                write_image_exif(job["path"], collection, job["models"], et)
        elif job["type"] == "video":
            if job["op"] == "write_exif" and "vision" in job['models']:
                logger.info("Writing EXIF for %s with models %s", job["path"], job["models"])
                write_video_exif(job["path"], videocollection, job["models"], et)


def recognitionprocessor():
    while True:
        logger.info("Waiting for job")
        job = json.loads(pull("queue")[1])
        # TODO: use multithreading, work out any possible issues with exiftool
        if job["type"] == "image":
            logger.info("Processing image, job is %s", job)
            if job["op"] == "recognition":
                if job["subdiv"] == "screenshot" and job['models'] is not None:
                    recognize_image(job["path"], screenshotcollection, job["subdiv"], job["is_screenshot"], job["models"], )
                elif job['models'] is not None:
                    recognize_image(job["path"], collection, job["subdiv"], job["is_screenshot"], job["models"], )
        elif job["type"] == "video":
            if job["op"] == "recognition" and job['models'] is not None:
                logger.info("Processing video, job is %s", job)
                recognize_video(job["path"], videocollection, job["subdiv"], job["models"])
            # TODO: client.py:178


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.threads) as executor:
        executor.submit(exifprocessor)
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.threads) as executor:
        executor.submit(recognitionprocessor)


if __name__ == '__main__':
    main()
