import cv2
import numpy as np

# MediaPipe Face Mesh indices
LEFT = 61   # Left corner of mouth
RIGHT = 291 # Right corner of mouth
UPPER_LIP = 0 # Upper lip top (midterm)
LOWER_LIP = 17 # Lower lip bottom (midterm)

def warp_smile(image, landmarks, intensity=6):
    img = image.copy()

    lx, ly = landmarks[LEFT]
    rx, ry = landmarks[RIGHT]
    
    # Use upper and lower lip for vertical centering
    # This prevents the smile from being shifted downwards
    _, uy = landmarks[UPPER_LIP]
    _, dy = landmarks[LOWER_LIP]

    # Mouth center
    cx = (lx + rx) // 2
    # cy is midpoint between upper and lower lip vertical extents
    cy = (uy + dy) // 2

    w = abs(rx - lx) + 20
    h = 40

    x1 = max(0, cx - w // 2)
    x2 = min(img.shape[1], cx + w // 2)
    y1 = max(0, cy - h // 2)
    y2 = min(img.shape[0], cy + h // 2)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return img

    roi_h, roi_w = roi.shape[:2]

    map_y, map_x = np.meshgrid(
        np.arange(roi_h),
        np.arange(roi_w),
        indexing='ij'
    )

    influence = np.zeros_like(map_y, dtype=np.float32)

    left_x = lx - x1
    right_x = rx - x1

    for x in range(roi_w):
        if x < left_x:
            influence[:, x] = 0
        elif x > right_x:
            influence[:, x] = 0
        else:
            t = (x - left_x) / (right_x - left_x)
            influence[:, x] = np.sin(np.pi * t)

    map_y = map_y.astype(np.float32) - influence * intensity

    warped = cv2.remap(
        roi,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (roi_w // 2, roi_h // 2),
        (roi_w // 2, roi_h // 2),
        0, 0, 360,
        (255, 255, 255),
        -1
    )

    center = (cx, cy)
    img = cv2.seamlessClone(
        warped,
        img,
        mask,
        center,
        cv2.NORMAL_CLONE
    )

    return img
