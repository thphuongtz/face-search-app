import os
import numpy as np
import face_recognition
import faiss
from PIL import Image

def load_database(database_folder="database"):
    """Load toàn bộ ảnh trong thư mục database và tạo embeddings"""
    paths_db = [
        os.path.join(database_folder, f)
        for f in os.listdir(database_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    embeddings, imgs = [], []
    for path in paths_db:
        img = np.array(Image.open(path).convert("RGB"))
        encs = face_recognition.face_encodings(img)
        if encs:
            embeddings.append(encs[0].astype("float32"))
            imgs.append(img)
    if len(embeddings) == 0:
        raise ValueError("Không có ảnh hợp lệ trong thư mục database/")
    embeddings = np.array(embeddings)
    
    # Tạo FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings, imgs, paths_db

def search_face(index, query_embedding, k=1):
    """Tìm ảnh gần nhất trong FAISS index"""
    distances, indices = index.search(np.array([query_embedding]), k)
    return distances[0], indices[0]
