# FAISS.py  (giữ tên nếu bạn đã dùng)
import numpy as np
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh

def get_face_embedding(image):
    """
    image: BGR OpenCV image (numpy array)
    return: 1D float32 embedding (landmark vector) hoặc None nếu không có face
    """
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        embedding = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        return embedding.astype('float32')
