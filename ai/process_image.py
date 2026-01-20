from ai.face_crop import crop_face
from ai.expression_detect import detect_expression
from ai.expression_edit import edit_expression
from ai.face_paste import paste_face

def process_image(image, faces, target_expression):
    for face_bbox in faces:
        face_img, coords = crop_face(image, face_bbox)
        edited_face = edit_expression(face_img, target_expression)
        image = paste_face(image, edited_face, coords)
    return image
