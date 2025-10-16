import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import numpy as np
import faiss
import os
from FAISS import build_faiss_index
import tempfile

# --- Cấu hình ---
DB_PATH = "database"  # hoặc đường dẫn tới thư mục ảnh của bạn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Khởi tạo MTCNN và ResNet ---
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Hàm vẽ landmarks ---
def show_face_with_landmarks(img, landmarks):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    if landmarks is not None:
        for (x, y) in landmarks.astype(int):
            r = 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    return img_copy

# --- Tải database ---
os.makedirs(DB_PATH, exist_ok=True)
image_files = [os.path.join(DB_PATH, f) for f in os.listdir(DB_PATH)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    st.error("⚠️ Thư mục database trống — hãy thêm ảnh mẫu vào.")
    st.stop()

index, filenames = build_faiss_index(image_files)
st.success(f"✅ Đã tải {len(filenames)} ảnh trong database!")

# --- Giao diện ---
st.title("🎥 Nhận diện khuôn mặt — Webcam & Upload ảnh")

mode = st.radio("Chọn nguồn ảnh:", ["📸 Webcam", "📁 Tải ảnh từ file"])

# --- Chụp ảnh từ webcam ---
if mode == "📸 Webcam":
    img_data = st.camera_input("Chụp ảnh khuôn mặt của bạn")
    if img_data:
        img = Image.open(img_data).convert("RGB")

# --- Upload ảnh từ file ---
else:
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

# --- Xử lý nhận diện nếu có ảnh ---
if 'img' in locals():
    boxes, probs, landmarks_all = mtcnn.detect(img, landmarks=True)

    if boxes is None:
        st.error("❌ Không phát hiện được khuôn mặt!")
        st.stop()

    # --- Crop khuôn mặt ---
    x1, y1, x2, y2 = boxes[0]
    face_pil = img.crop((x1, y1, x2, y2)).resize((160, 160))

    # --- Tạo embedding ---
    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0
    face_tensor = (face_tensor.unsqueeze(0).to(device) * 2) - 1
    query_emb = resnet(face_tensor).detach().cpu().numpy()

    # --- Tìm trong FAISS ---
    D, I = index.search(query_emb.astype('float32'), 1)
    matched_path = filenames[I[0][0]]
    distance = float(D[0][0])
    person_name = os.path.splitext(os.path.basename(matched_path))[0]

    threshold = 0.9  # ngưỡng có thể chỉnh

    st.image(face_pil, caption="Khuôn mặt trích xuất", width=200)

    if distance < threshold:
        st.success(f"✅ Khớp với: **{person_name}** (Khoảng cách: {distance:.4f})")
        st.image(Image.open(matched_path), caption=f"Ảnh trong database ({person_name})", width=200)
    else:
        st.warning(f"⚠️ Không khớp (Khoảng cách: {distance:.4f})")
