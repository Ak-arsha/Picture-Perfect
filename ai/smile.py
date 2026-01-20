import cv2
import numpy as np

def enhance_smile(image, landmarks, intensity=6):
    """
    image: original image (BGR)
    landmarks: list of (x, y) for one face
    intensity: how strong the smile is
    """

    img = image.copy()

    LEFT = 25
    RIGHT = 29

    left_x, left_y = landmarks[LEFT]
    right_x, right_y = landmarks[RIGHT]

    new_left = (left_x, left_y - intensity)
    new_right = (right_x, right_y - intensity)

    cv2.circle(img, new_left, 2, (0, 255, 0), -1)
    cv2.circle(img, new_right, 2, (0, 255, 0), -1)

    return img
