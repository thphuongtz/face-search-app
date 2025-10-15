import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from FAISS import get_face_embedding

st.set_page_config(page_title="Face Search App (MediaPipe)", layout="wide")

st.title("🔍 Face Search App (MediaPipe + NumPy)")
os.makedirs("data/known_faces", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

tab1, tab2 = st.tabs(["🧠 Add Known Faces", "🔎 Search Face"])

# --- TAB 1 ---
with tab1:
    st.header("🧠 Thêm khuôn mặt mới")
    upload_img = st.file_uploader("Tải ảnh khuôn mặt", type=["jpg", "png", "jpeg"])
    name = st.text_input("Tên người này:")

    if st.button("Lưu vào cơ sở dữ liệu") and upload_img and name:
        img_path = f"data/known_faces/{name}.jpg"
        Image.open(upload_img).save(img_path)
        st.success(f"✅ Đã lưu {name}.jpg vào cơ sở dữ liệu!")

# --- TAB 2 ---
with tab2:
    st.header("🔎 Tìm kiếm khuôn mặt trong ảnh mới")
    query_file = st.file_uploader("Chọn ảnh cần tìm", type=["jpg", "png", "jpeg"])
    if query_file:
        img_path = f"data/uploads/query.jpg"
        Image.open(query_file).save(img_path)
        img = cv2.imread(img_path)

        boxes, query_emb = get_face_embedding(img)
        if query_emb is None:
            st.error("❌ Không phát hiện được khuôn mặt.")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Ảnh tìm kiếm")

            # Tải cơ sở dữ liệu
            db = {}
            for f in os.listdir("data/known_faces"):
                name = os.path.splitext(f)[0]
                img_db = cv2.imread(os.path.join("data/known_faces", f))
                _, emb_db = get_face_embedding(img_db)
                if emb_db is not None and len(emb_db) > 0:
                    db[name] = emb_db[0]

            if len(db) == 0:
                st.warning("⚠️ Cơ sở dữ liệu trống. Hãy thêm khuôn mặt trước.")
            else:
                similarities = {}
                for name, emb in db.items():
                    sim = np.dot(query_emb[0], emb) / (np.linalg.norm(query_emb[0])*np.linalg.norm(emb))
                    similarities[name] = sim

                best_match = max(similarities, key=similarities.get)
                st.success(f"✅ Kết quả gần nhất: **{best_match}** (similarity={similarities[best_match]:.3f})")
                st.bar_chart(similarities)
