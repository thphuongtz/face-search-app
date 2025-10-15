import os
import numpy as np
import faiss
import cv2
from PIL import Image
import streamlit as st
from FAISS import get_face_embedding
import matplotlib.pyplot as plt

# ==============================
# 🚀 Cấu hình giao diện
# ==============================
st.set_page_config(page_title="Face Search App", layout="wide")
st.title("🔍 Face Recognition Search using FAISS + MediaPipe")

# ==============================
# 📂 LOAD DATABASE
# ==============================
database_folder = "database"

if not os.path.exists(database_folder):
    st.error("❌ Không tìm thấy thư mục `database/`! Hãy tạo và thêm ảnh khuôn mặt vào đó.")
    st.stop()

paths_db = [
    os.path.join(database_folder, f)
    for f in os.listdir(database_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

@st.cache_data(show_spinner=True)
def load_database(paths):
    embeddings, imgs = [], []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        emb = get_face_embedding(img)
        if emb is not None:
            embeddings.append(emb)
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(embeddings) == 0:
        st.error("⚠️ Không tạo được embedding nào từ database — hãy kiểm tra ảnh!")
        st.stop()
    return np.array(embeddings), imgs

embeddings_db, imgs_db = load_database(paths_db)
d = embeddings_db.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings_db)

# ==============================
# 📤 UPLOAD ẢNH TRUY VẤN
# ==============================
uploaded_file = st.file_uploader("📤 Tải lên ảnh truy vấn", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_query = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(img_query, caption="Ảnh truy vấn", use_container_width=True)

    # Lấy embedding khuôn mặt từ query
    emb_query = get_face_embedding(cv2.cvtColor(img_query, cv2.COLOR_RGB2BGR))
    if emb_query is None:
        st.error("❌ Không phát hiện khuôn mặt trong ảnh truy vấn.")
    else:
        # Tìm kiếm trong FAISS
        distances, indices = index.search(np.array([emb_query]), 1)
        nearest_idx = indices[0][0]
        nearest_distance = distances[0][0]

        # ==============================
        # 🧠 KẾT QUẢ NHẬN DIỆN
        # ==============================
        st.subheader("🧠 Kết quả nhận diện")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgs_db[nearest_idx], caption=f"Ảnh khớp: {os.path.basename(paths_db[nearest_idx])}")
        with col2:
            st.metric(label="Khoảng cách (Euclidean)", value=f"{nearest_distance:.4f}")

        # ==============================
        # 📊 Biểu đồ embedding
        # ==============================
        st.subheader("📊 So sánh 20 phần tử đầu embedding")
        plt.figure(figsize=(8, 4))
        index_plot = np.arange(20)
        plt.bar(index_plot, emb_query[:20], 0.35, label="Query", alpha=0.8)
        plt.bar(index_plot + 0.35, embeddings_db[nearest_idx][:20], 0.35, label="Database", alpha=0.8)
        plt.legend()
        plt.title("So sánh 20 phần tử đầu embedding")
        st.pyplot(plt)
