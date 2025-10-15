import numpy as np
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh

def get_face_embedding(image):
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        # Lấy toàn bộ 468 điểm khuôn mặt
        landmarks = results.multi_face_landmarks[0].landmark
        embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return embedding.astype('float32')
