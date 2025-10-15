# FAISS.py
import numpy as np
import cv2
import insightface

# Tải model
model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image):
    """
    image: BGR image (numpy array)
    return: 1D float32 embedding or None
    """
    faces = model.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding.astype('float32')
