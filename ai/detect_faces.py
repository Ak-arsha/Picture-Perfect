import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

def detect_faces(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found. Check the path.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=None
        ),
        min_detection_confidence=0.3
    )

    detector = vision.FaceDetector.create_from_options(options)

    detection_result = detector.detect(mp_image)

    boxes = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        boxes.append(
            (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
        )

    detector.close()
    return boxes
