import cv2
from ai.face_mesh import get_face_landmarks
from ai.smile_warp import warp_smile
from ai.gaze_correction import correct_gaze

image = cv2.imread("test_images/group.jpg")
faces = get_face_landmarks(image)

for face in faces:
    # 0. Debug Gaze (Draw points)
    # from ai.gaze_correction import draw_debug_gaze
    # draw_debug_gaze(image, face, True)
    # draw_debug_gaze(image, face, False)

    # 1. Correct Gaze (Eyes looking at camera)
    # Intensity=1.0 moves the iris to the center of the eye.
    image = correct_gaze(image, face, intensity=1.0)
    
    # 2. Warp Smile
    image = warp_smile(image, face, intensity=6)

cv2.imwrite("test_images/result_smile_warp.jpg", image)
cv2.imwrite("test_images/result_gaze_debug.jpg", image) # Save copy for easy finding
cv2.imshow("Smile Warp Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
