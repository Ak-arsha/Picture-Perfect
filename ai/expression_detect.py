from transformers import pipeline
from PIL import Image
import cv2

emotion_classifier = pipeline(
    "image-classification",
    model="nateraw/fer2013"
)

def detect_expression(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    result = emotion_classifier(pil_img)
    return result[0]["label"]  
