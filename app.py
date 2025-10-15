import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from deepface import DeepFace
import faiss
import streamlit as st

# ======== GIAO DIỆN ========
st.set_page_config(page_title="Face Search App", layout="wide")
st.title("🔍 Face Recognition Search using FAISS + DeepFace")

# ======== LOAD DATABASE ========
database_folder = "database"
paths_db = [
    os.path.join(database_folder, f)
    for f in os.listdir(database_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

@st.cache_data(show_spinner=True)
def load_database(paths):
    embeddings, imgs = [], []
    for path in paths:
        img = np.array(Image.open(path).convert("RGB"))
        try:
            emb = DeepFace.represent(img, model_name="Facenet")[0]["embedding"]
            embeddings.append(np.array(emb, dtype="float32"))
            imgs.append(img)
        except:
            st.warning(f"⚠️ Không trích xuất được đặc trưng từ ảnh: {path}")
    return np.array(embeddings), imgs

if len(paths_db) == 0:
    st.error("❌ Không có ảnh nào trong thư mục `database/`!")
    st.stop()

embeddings_db, imgs_db = load_database(paths_db)
d = embeddings_db.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings_db)

# ======== UPLOAD ẢNH TRUY VẤN ========
uploaded_file = st.file_uploader("📤 Tải lên ảnh truy vấn", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img_query = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(img_query, caption="Ảnh truy vấn", use_container_width=True)

    try:
        emb_query = DeepFace.represent(img_query, model_name="Facenet")[0]["embedding"]
        emb_query = np.array(emb_query, dtype="float32")

        distances, indices = index.search(np.array([emb_query]), 1)
        nearest_idx = indices[0][0]
        nearest_distance = distances[0][0]

        st.subheader("🧠 Kết quả nhận diện")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgs_db[nearest_idx], caption=f"Kết quả khớp ({os.path.basename(paths_db[nearest_idx])})")
        with col2:
            st.metric(label="Khoảng cách (Euclidean)", value=f"{nearest_distance:.4f}")

        # Biểu đồ embedding
        st.subheader("📊 So sánh 20 phần tử đầu của embedding")
        plt.figure(figsize=(8, 4))
        index_plot = np.arange(20)
        plt.bar(index_plot, emb_query[:20], 0.35, label="Query", alpha=0.8)
        plt.bar(index_plot + 0.35, embeddings_db[nearest_idx][:20], 0.35, label="Database", alpha=0.8)
        plt.legend()
        plt.title("So sánh 20 phần tử đầu embedding")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Lỗi khi trích xuất khuôn mặt: {e}")
