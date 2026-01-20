import cv2
import os

def detect_faces_and_eyes(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        "models/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        "models/haarcascade_eye.xml"
    )

    faces_strict = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(60, 60)
    )

    faces_loose = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(30, 30)
    )

    faces = list(faces_strict)

    for face in faces_loose:
        if not any(
            abs(face[0] - f[0]) < 20 and abs(face[1] - f[1]) < 20
            for f in faces
        ):
            faces.append(face)

    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )

        results.append({
            "face": (x, y, w, h),
            "eyes": eyes
        })

    return image, results
