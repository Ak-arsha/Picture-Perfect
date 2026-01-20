import cv2
import numpy as np

def get_face_landmarks(image):
    """
    Input: BGR image
    Output: list of faces, each face = list of (x, y) landmarks using MediaPipe Face Mesh
    """
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        # Convert the BGR image to RGB
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks_all_faces = []

        if results.multi_face_landmarks:
            h, w, _ = image.shape
            print(f"Detected {len(results.multi_face_landmarks)} face(s) using MediaPipe")
            
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for lm in face_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                
                landmarks_all_faces.append(points)
        else:
            print("No faces detected.")

        return landmarks_all_faces
