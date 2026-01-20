import cv2

def crop_face(image, bbox, padding=30):
    x, y, w, h = bbox

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    face = image[y1:y2, x1:x2]
    return face, (x1, y1, x2, y2)
