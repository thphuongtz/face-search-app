import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from FAISS import get_face_embedding  # file FAISS.py như ở trên

# ========== GIAO DIỆN ==========
st.set_page_config(page_title="Face Search App (MediaPipe + NumPy)", layout="wide")
st.title("🔍 Face Search App — MediaPipe embeddings + NumPy search")

# ========== KIỂM TRA THƯ MỤC DATABASE ==========
database_folder = "database"
if not os.path.exists(database_folder):
    st.error("❌ Không tìm thấy thư mục `database/`. Hãy tạo folder 'database' và upload ảnh vào đó.")
    st.stop()

paths_db = [
    os.path.join(database_folder, f)
    for f in os.listdir(database_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
if len(paths_db) == 0:
    st.error("❌ Thư mục `database/` rỗng. Upload tối thiểu 1 ảnh.")
    st.stop()

# ========== LOAD DATABASE VÀ TẠO EMBEDDINGS ==========
@st.cache_data(show_spinner=True)
def load_database(paths):
    embeddings = []
    imgs_rgb = []
    good_paths = []
    for p in paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        emb = get_face_embedding(img_bgr)
        if emb is not None:
            embeddings.append(emb)
            imgs_rgb.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            good_paths.append(p)
    if len(embeddings) == 0:
        return None, None, None
    return np.vstack(embeddings), imgs_rgb, good_paths

embeddings_db, imgs_db, paths_db_filtered = load_database(paths_db)
if embeddings_db is None:
    st.error("⚠️ Không tạo được embedding nào từ database. Hãy kiểm tra ảnh (phải có khuôn mặt rõ ràng).")
    st.stop()

# ========== UPLOAD ẢNH TRUY VẤN ==========
uploaded_file = st.file_uploader("📤 Tải lên ảnh truy vấn", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Upload 1 ảnh để truy vấn. Ảnh trong database: " + str(len(paths_db_filtered)))
else:
    # show query
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_query_rgb = np.array(img_pil)
    st.image(img_query_rgb, caption="Ảnh truy vấn", use_container_width=True)

    # convert to BGR for mediapipe
    img_query_bgr = cv2.cvtColor(img_query_rgb, cv2.COLOR_RGB2BGR)
    emb_query = get_face_embedding(img_query_bgr)
    if emb_query is None:
        st.error("Không phát hiện khuôn mặt trong ảnh truy vấn.")
    else:
        # compute distances to all embeddings (Euclidean)
        # embeddings_db shape: (N, D); emb_query shape: (D,)
        diffs = embeddings_db - emb_query
        dists = np.linalg.norm(diffs, axis=1)
        # find best
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        st.subheader("🧠 Kết quả nhận diện")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(imgs_db[best_idx], caption=f"Best match: {os.path.basename(paths_db_filtered[best_idx])}")
        with col2:
            st.metric("Khoảng cách (Euclidean)", f"{best_dist:.4f}")

        # show top-k (tùy chọn)
        k = min(5, len(dists))
        topk_idx = np.argsort(dists)[:k]
        st.subheader(f"Top-{k} candidates")
        cols = st.columns(k)
        for i, idx in enumerate(topk_idx):
            with cols[i]:
                st.image(imgs_db[idx], caption=f"{i+1}. {os.path.basename(paths_db_filtered[idx])}\n{dists[idx]:.4f}", use_column_width=True)

        # biểu đồ so sánh 20 phần tử đầu (nhỏ)
        st.subheader("📊 So sánh 20 phần tử đầu embedding")
        plt.figure(figsize=(8, 4))
        idx_plot = np.arange(20)
        plt.bar(idx_plot, emb_query[:20], 0.35, label="Query")
        plt.bar(idx_plot + 0.35, embeddings_db[best_idx][:20], 0.35, label="Database")
        plt.legend()
        plt.title("So sánh 20 phần tử đầu embedding")
        st.pyplot(plt)
