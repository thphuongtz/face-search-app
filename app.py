import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import numpy as np
import faiss
import os
import matplotlib.pyplot as plt
from FAISS import build_faiss_index

# --- Cấu hình ---
DB_PATH = "database"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Khởi tạo MTCNN và ResNet ---
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Chuẩn bị database ---
os.makedirs(DB_PATH, exist_ok=True)
image_files = [os.path.join(DB_PATH, f) for f in os.listdir(DB_PATH)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    st.error("⚠️ Thư mục database trống — hãy thêm ảnh mẫu vào.")
    st.stop()

index, filenames = build_faiss_index(image_files)
st.success(f"✅ Đã tải {len(filenames)} ảnh trong database!")

# --- Giao diện ---
st.title("🎥 Nhận diện khuôn mặt tự động")

mode = st.radio("Chọn nguồn ảnh:", ["📸 Webcam", "📁 Tải ảnh từ file"])

if mode == "📸 Webcam":
    st.info("🧠 Hệ thống sẽ tự nhận diện khuôn mặt ngay sau khi chụp.")
    img_data = st.camera_input("Bật webcam để chụp tự động")

    if img_data:
        img = Image.open(img_data).convert("RGB")

elif mode == "📁 Tải ảnh từ file":
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

else:
    img = None

# --- Xử lý ảnh (nếu có) ---
if 'img' in locals():
    boxes, probs, landmarks_all = mtcnn.detect(img, landmarks=True)

    if boxes is None:
        st.error("❌ Không phát hiện được khuôn mặt!")
        st.stop()

    # Crop khuôn mặt đầu tiên (ảnh truy vấn)
    x1, y1, x2, y2 = boxes[0]
    face_pil = img.crop((x1, y1, x2, y2)).resize((160, 160))

    # Embedding khuôn mặt
    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0
    face_tensor = (face_tensor.unsqueeze(0).to(device) * 2) - 1
    query_emb = resnet(face_tensor).detach().cpu().numpy()

    # So khớp với database
    D, I = index.search(query_emb.astype('float32'), 1)
    matched_path = filenames[I[0][0]]
    distance = float(D[0][0])
    person_name = os.path.splitext(os.path.basename(matched_path))[0]
    threshold = 0.9

    # --- Cắt khuôn mặt từ ảnh database ---
    db_img = Image.open(matched_path).convert("RGB")
    detect_result = mtcnn.detect(db_img)

    if detect_result is not None:
        db_boxes, _, _ = detect_result
        if db_boxes is not None:
            x1d, y1d, x2d, y2d = db_boxes[0]
            matched_face = db_img.crop((x1d, y1d, x2d, y2d)).resize((160, 160))
        else:
            matched_face = db_img.resize((160, 160))
    else:
        matched_face = db_img.resize((160, 160))

    # --- Hiển thị kết quả ---
    if distance < threshold:
        st.success(f"✅ Khớp với: **{person_name}** (Khoảng cách: {distance:.4f})")

        col1, col2 = st.columns(2)
        with col1:
            st.image(face_pil, caption="🧩 Ảnh truy vấn (đã cắt)", width=250)
        with col2:
            st.image(matched_face, caption=f"🎯 Ảnh trong DB: {person_name}", width=250)

        # --- Đồ thị so sánh vector đặc trưng ---
        db_face_tensor = torch.tensor(np.array(matched_face)).permute(2, 0, 1).float() / 255.0
        db_face_tensor = (db_face_tensor.unsqueeze(0).to(device) * 2) - 1
        db_emb = resnet(db_face_tensor).detach().cpu().numpy()

        plt.figure(figsize=(8, 4))
        plt.plot(query_emb[0][:20], label="Ảnh truy vấn", marker='o')
        plt.plot(db_emb[0][:20], label="Ảnh trong DB", marker='x')
        plt.title("🔍 So sánh 20 giá trị vector đặc trưng đầu tiên")
        plt.legend()
        st.pyplot(plt)

    else:
        st.warning(f"⚠️ Không khớp với ai trong database (Khoảng cách: {distance:.4f})")
