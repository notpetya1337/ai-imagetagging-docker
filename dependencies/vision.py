import logging
import os

from google.cloud import vision

logger = logging.getLogger(__name__)


class Tagging:
    def __init__(self, google_credentials, google_project, tags_backend):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
        os.environ["GOOGLE_CLOUD_PROJECT"] = google_project
        if tags_backend:
            self.tags_backend = tags_backend
        else:
            self.tags_backend = 'google-vision'

    def get_tags(self, image_binary):
        tags = self.google_vision_labels(image_binary=image_binary)
        return tags

    def get_text(self, image_binary):
        text = self.google_vision_light_ocr(image_binary=image_binary)
        return text

    # TODO: this doesn't work yet
    def get_ocr_text(self, image_binary):
        ocrtext = self.google_vision_heavy_ocr(image_binary=image_binary)
        return ocrtext

    def google_vision_labels(self, image_binary):
        client = vision.ImageAnnotatorClient()
        # Loads the image into memory
        image = vision.Image(content=image_binary)
        # Performs label detection on the image file
        responsetags = client.label_detection(image=image)
        labels = responsetags.label_annotations
        tags = []
        for label in labels:
            tags.append(label.description)
        return tags

    def get_explicit(self, image_binary):
        text = self.google_vision_explicit_detection(image_binary=image_binary)
        return text

    def google_vision_light_ocr(self, image_binary):
        client = vision.ImageAnnotatorClient()
        # Loads the image into memory
        image = vision.Image(content=image_binary)
        # Performs label detection on the image file
        responsetags = client.text_detection(image=image)
        textobject = responsetags.text_annotations
        returntext = []
        for text in textobject:
            returntext.append(text.description)
        if not returntext:
            logger.info("Text not found in image, appending placeholder")
            returntext.append("No text detected.")
        return returntext

    # TODO: this doesn't work yet
    def google_vision_heavy_ocr(self, image_binary):
        client = vision.ImageAnnotatorClient()
        # Loads the image into memory
        image = vision.Image(content=image_binary)
        # Performs label detection on the image file
        response = client.document_text_detection(image=image)
        textobject = response.text_annotations
        return textobject

    def google_vision_explicit_detection(self, image_binary):
        client = vision.ImageAnnotatorClient()
        # Loads the image into memory
        image = vision.Image(content=image_binary)
        # Performs label detection on the image file
        response = client.safe_search_detection(image=image)
        safe = response.safe_search_annotation
        # Names of likelihood from google.cloud.vision.enums
        likelihood_name = ("UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY",)
        detectionobject = safe
        return safe
