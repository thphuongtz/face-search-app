import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import face_recognition
from FAISS import load_database, search_face
import os

# ======== GIAO DIỆN ========
st.set_page_config(page_title="Face Search App", layout="wide")
st.title("🔍 Face Recognition Search using FAISS")

# ======== LOAD DATABASE ========
database_folder = "database"

if not os.path.exists(database_folder):
    st.error("❌ Thư mục 'database/' chưa tồn tại! Hãy tạo và thêm ảnh trước.")
    st.stop()

@st.cache_resource(show_spinner=True)
def init_faiss():
    return load_database(database_folder)

index, embeddings_db, imgs_db, paths_db = init_faiss()

# ======== UPLOAD ẢNH TRUY VẤN ========
uploaded_file = st.file_uploader("📤 Tải lên ảnh truy vấn", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img_query = np.array(Image.open(uploaded_file).convert("RGB"))
    st.image(img_query, caption="Ảnh truy vấn", use_container_width=True)

    enc_query = face_recognition.face_encodings(img_query)
    if len(enc_query) == 0:
        st.error("Không phát hiện khuôn mặt trong ảnh truy vấn.")
    else:
        embedding_query = enc_query[0].astype("float32")
        distances, indices = search_face(index, embedding_query, k=1)
        nearest_idx = indices[0]
        nearest_distance = distances[0]

        # Hiển thị kết quả
        st.subheader("🧠 Kết quả nhận diện")
        col1, col2 = st.columns(2)
        with col1:
            st.image(imgs_db[nearest_idx], caption=f"Kết quả: {os.path.basename(paths_db[nearest_idx])}")
        with col2:
            st.metric(label="Khoảng cách (Euclidean)", value=f"{nearest_distance:.4f}")

        # Biểu đồ embedding
        st.subheader("📊 So sánh 20 phần tử đầu của embedding")
        plt.figure(figsize=(8, 4))
        idx_plot = np.arange(20)
        plt.bar(idx_plot, embedding_query[:20], 0.35, label="Query", alpha=0.8)
        plt.bar(idx_plot + 0.35, embeddings_db[nearest_idx][:20], 0.35, label="Database", alpha=0.8)
        plt.legend()
        plt.title("So sánh 20 phần tử đầu embedding")
        st.pyplot(plt)
