import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
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

# --- Hàm vẽ so sánh 20 vector đầu tiên ---
def plot_embedding_comparison(query_emb, matched_emb):
    fig, ax = plt.subplots()
    x = np.arange(20)
    ax.plot(x, query_emb[:20], label='Ảnh truy vấn', marker='o')
    ax.plot(x, matched_emb[:20], label='Ảnh database', marker='x')
    ax.set_title("So sánh 20 vector đặc trưng đầu tiên")
    ax.set_xlabel("Chỉ số vector")
    ax.set_ylabel("Giá trị")
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
st.title("🤖 Nhận diện khuôn mặt tự động (2 ảnh crop)")

mode = st.radio("Chọn nguồn ảnh:", ["📸 Webcam", "📁 Tải ảnh từ file"])

if mode == "📸 Webcam":
    st.info("🧠 Hệ thống sẽ nhận diện khuôn mặt sau khi chụp.")
    img_data = st.camera_input("Bật webcam để chụp ảnh")
    if img_data:
        img = Image.open(img_data).convert("RGB")

elif mode == "📁 Tải ảnh từ file":
    uploaded_file = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
else:
    img = None

# --- Xử lý ---
if 'img' in locals():
    boxes, probs, _ = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        st.error("❌ Không phát hiện được khuôn mặt!")
        st.stop()

    # Cắt khuôn mặt truy vấn
    x1, y1, x2, y2 = boxes[0]
    query_face = img.crop((x1, y1, x2, y2)).resize((160, 160))

    # Tính embedding
    face_tensor = torch.tensor(np.array(query_face)).permute(2, 0, 1).float() / 255.0
    face_tensor = (face_tensor.unsqueeze(0).to(device) * 2) - 1
    query_emb = resnet(face_tensor).detach().cpu().numpy().flatten()

    # Tìm ảnh database khớp nhất
    D, I = index.search(query_emb.astype('float32').reshape(1, -1), 1)
    matched_path = filenames[I[0][0]]
    distance = float(D[0][0])
    person_name = os.path.splitext(os.path.basename(matched_path))[0]
    threshold = 0.9

    # Cắt khuôn mặt trong database
    db_img = Image.open(matched_path).convert("RGB")
    db_boxes, _, _ = mtcnn.detect(db_img)
    if db_boxes is not None:
        x1, y1, x2, y2 = db_boxes[0]
        matched_face = db_img.crop((x1, y1, x2, y2)).resize((160, 160))
    else:
        matched_face = db_img.resize((160, 160))

    # Embedding DB để so sánh vector
    matched_tensor = torch.tensor(np.array(matched_face)).permute(2, 0, 1).float() / 255.0
    matched_tensor = (matched_tensor.unsqueeze(0).to(device) * 2) - 1
    matched_emb = resnet(matched_tensor).detach().cpu().numpy().flatten()

    # --- Hiển thị ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(query_face, caption="📸 Ảnh truy vấn", width=200)
    with col2:
        st.image(matched_face, caption=f"🧠 Ảnh trong database ({person_name})", width=200)

    # --- Kết quả ---
    if distance < threshold:
        st.success(f"✅ Khớp với **{person_name}** (Khoảng cách: {distance:.4f})")
    else:
        st.warning(f"⚠️ Không khớp với ai trong database (Khoảng cách: {distance:.4f})")

    # --- So sánh vector ---
    plot_embedding_comparison(query_emb, matched_emb)
