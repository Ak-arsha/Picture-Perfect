import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "models/shape_predictor_68_face_landmarks.dat"
)

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    all_faces = []

    for face in faces:
        shape = predictor(gray, face)
        points = []

        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            points.append((x, y))

        all_faces.append(points)

    return all_faces
