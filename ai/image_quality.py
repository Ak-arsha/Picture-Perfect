import cv2
import numpy as np

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    img = img.astype(np.int16)
    img = img * (1 + contrast / 100) + brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def apply_softness(img, softness=0.0):
    if softness <= 0:
        return img
    k = int(softness * 30) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_sharpness(img, sharpness=0.0):
    if sharpness <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + sharpness, blur, -sharpness, 0)

def apply_warmth(img, warmth=0):
    if warmth == 0:
        return img
    b, g, r = cv2.split(img)
    r = np.clip(r + warmth, 0, 255)
    b = np.clip(b - warmth, 0, 255)
    return cv2.merge([b, g, r])
