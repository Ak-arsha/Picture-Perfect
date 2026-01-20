def paste_face(image, edited_face, coords):
    x1, y1, x2, y2 = coords
    resized = cv2.resize(edited_face, (x2 - x1, y2 - y1))
    image[y1:y2, x1:x2] = resized
    return image
