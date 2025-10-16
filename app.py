import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import numpy as np
import faiss
import os
from FAISS import build_faiss_index
import matplotlib.pyplot as plt

# --- Cấu hình ---
DB_PATH = "database"
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

# --- Hàm vẽ so sánh 20 vector đầu tiên ---
def plot_embedding_comparison(query_emb, matched_emb):
    fig, ax = plt.subplots()
    x = np.arange(20)
    ax.plot(x, query_emb[:20], label='Query', marker='o')
    ax.plot(x, matched_emb[:20], label='Matched', marker='x')
    ax.set_title("So sánh 20 vector đặc trưng đầu tiên")
    ax.set_xlabel("Index vector")
    ax.set_ylabel("Giá trị embedding")
    ax.legend()
    st.pyplot(fig)

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
    # Phát hiện khuôn mặt
    boxes, probs, landmarks_all = mtcnn.detect(img, landmarks=True)

    if boxes is None:
        st.error("❌ Không phát hiện được khuôn mặt!")
        st.stop()

    # Crop khuôn mặt đầu tiên
    x1, y1, x2, y2 = boxes[0]
    face_pil = img.crop((x1, y1, x2, y2)).resize((160, 160))

    # Vẽ khung lên ảnh gốc
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    st.image(img, caption="📸 Ảnh gốc (đã phát hiện khuôn mặt)", use_container_width=True)

    # Hiển thị ảnh khuôn mặt cắt ra
    st.image(face_pil, caption="🧩 Khuôn mặt tự cắt ra", width=200)

    # Embedding khuôn mặt truy vấn
    face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0
    face_tensor = (face_tensor.unsqueeze(0).to(device) * 2) - 1
    query_emb = resnet(face_tensor).detach().cpu().numpy().flatten()

    # So khớp với database
    D, I = index.search(query_emb.astype('float32').reshape(1, -1), 1)
    matched_path = filenames[I[0][0]]
    distance = float(D[0][0])
    person_name = os.path.splitext(os.path.basename(matched_path))[0]

    threshold = 0.9  # có thể tinh chỉnh

    # Tính embedding ảnh khớp để so sánh (resize ảnh trong DB thành 160x160)
    try:
        matched_img_pil = Image.open(matched_path).convert("RGB").resize((160, 160))
        matched_tensor = torch.tensor(np.array(matched_img_pil)).permute(2, 0, 1).float() / 255.0
        matched_tensor = (matched_tensor.unsqueeze(0).to(device) * 2) - 1
        matched_emb = resnet(matched_tensor).detach().cpu().numpy().flatten()
    except Exception as e:
        matched_emb = None
        st.warning(f"⚠️ Không thể mở ảnh để so sánh embedding: {e}")

    # Hiển thị kết quả
    if distance < threshold:
        st.success(f"✅ Khớp với: **{person_name}** (Khoảng cách: {distance:.4f})")
        st.image(Image.open(matched_path), caption=f"Ảnh trong database ({person_name})", width=200)
    else:
        st.warning(f"⚠️ Không khớp với ai trong database (Khoảng cách: {distance:.4f})")

    # Vẽ đồ thị so sánh 20 vector đầu tiên nếu có matched_emb
    if matched_emb is not None:
        plot_embedding_comparison(query_emb, matched_emb)
