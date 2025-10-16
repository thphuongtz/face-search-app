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
DB_PATH = "data/known_faces"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Khởi tạo MTCNN và ResNet ---
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Vẽ landmarks ---
def show_face_with_landmarks(img, landmarks):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    if landmarks is not None:
        for (x, y) in landmarks.astype(int):
            r = 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    return img_copy

# --- Vẽ đồ thị 20 vector đầu tiên ---
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

# --- Streamlit ---
st.title("Nhận diện khuôn mặt - FaceNet + FAISS")

# --- Tải database ---
image_files = [os.path.join(DB_PATH, f) for f in os.listdir(DB_PATH)
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
if not image_files:
    st.error("⚠️ Thư mục data/known_faces trống.")
else:
    index, filenames = build_faiss_index(image_files)
    st.success("✅ Database đã sẵn sàng!")

# --- Upload ảnh ---
uploaded_file = st.file_uploader("📸 Tải ảnh để nhận diện", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # --- Phát hiện khuôn mặt ---
    boxes, probs, landmarks_all = mtcnn.detect(img, landmarks=True)

    if boxes is None:
        st.error("❌ Không phát hiện được khuôn mặt!")
    else:
        # --- Lấy khuôn mặt đầu tiên ---
        x1, y1, x2, y2 = boxes[0]
        face_pil = img.crop((x1, y1, x2, y2)).resize((160, 160))

        # --- Landmarks trên crop ---
        landmarks = landmarks_all[0] if landmarks_all is not None else None
        if landmarks is not None:
            scale_x = 160 / (x2 - x1)
            scale_y = 160 / (y2 - y1)
            landmarks_crop = np.copy(landmarks)
            landmarks_crop[:, 0] = (landmarks[:, 0] - x1) * scale_x
            landmarks_crop[:, 1] = (landmarks[:, 1] - y1) * scale_y
        else:
            landmarks_crop = None

        face_landmarked = show_face_with_landmarks(face_pil, landmarks_crop)

        # --- Embedding truy vấn ---
        face_tensor_resnet = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float() / 255.0
        face_tensor_resnet = (face_tensor_resnet.unsqueeze(0).to(device) * 2) - 1
        query_emb = resnet(face_tensor_resnet).detach().cpu().numpy()

        # --- Tìm khuôn mặt gần nhất ---
        D, I = index.search(query_emb.astype('float32'), 1)
        matched_path = filenames[I[0][0]]
        distance = float(D[0][0])  # khoảng cách FAISS (L2)
        person_name = os.path.splitext(os.path.basename(matched_path))[0]

        # --- Ngưỡng nhận diện ---
        threshold = 0.9  # có thể tinh chỉnh 0.8–1.0 tùy dữ liệu

        # --- Hiển thị ảnh truy vấn ---
        st.image(face_landmarked, caption="Ảnh truy vấn (crop + landmarks, giữ màu gốc)", width=200)

        # --- Nếu khớp ---
        if distance < threshold:
            # --- Ảnh khớp nhất ---
            matched_img = Image.open(matched_path).convert("RGB")
            boxes_db, probs_db, landmarks_all_db = mtcnn.detect(matched_img, landmarks=True)
            if boxes_db is not None:
                x1, y1, x2, y2 = boxes_db[0]
                matched_face_pil = matched_img.crop((x1, y1, x2, y2)).resize((160, 160))
            else:
                matched_face_pil = matched_img.resize((160, 160))

            landmarks_db = landmarks_all_db[0] if landmarks_all_db is not None else None
            if landmarks_db is not None and boxes_db is not None:
                scale_x = 160 / (x2 - x1)
                scale_y = 160 / (y2 - y1)
                landmarks_crop_db = np.copy(landmarks_db)
                landmarks_crop_db[:, 0] = (landmarks_db[:, 0] - x1) * scale_x
                landmarks_crop_db[:, 1] = (landmarks_db[:, 1] - y1) * scale_y
            else:
                landmarks_crop_db = None

            matched_landmarked = show_face_with_landmarks(matched_face_pil, landmarks_crop_db)

            # --- Hiển thị song song ---
            col1, col2 = st.columns(2)
            with col1:
                st.image(face_landmarked, caption="Ảnh truy vấn", use_container_width=True)
            with col2:
                st.image(matched_landmarked, caption=f"Kết quả: {person_name}", use_container_width=True)

            # --- Thông tin nhận dạng ---
            st.markdown("---")
            st.subheader("📊 Kết quả nhận diện")
            st.success(f"✅ **Nhận diện thành công:** {person_name}\n\n📏 Khoảng cách: `{distance:.4f}` (Cùng người)")

            # --- So sánh vector ---
            matched_tensor_resnet = torch.tensor(np.array(matched_face_pil)).permute(2, 0, 1).float() / 255.0
            matched_tensor_resnet = (matched_tensor_resnet.unsqueeze(0).to(device) * 2) - 1
            matched_emb = resnet(matched_tensor_resnet).detach().cpu().numpy()

            plot_embedding_comparison(query_emb.flatten(), matched_emb.flatten())

        # --- Nếu KHÔNG khớp ---
        else:
            st.markdown("---")
            st.subheader("📊 Kết quả nhận diện")
            st.warning(f"⚠️ Không tồn tại trong dữ liệu!\n\n📏 Khoảng cách gần nhất: `{distance:.4f}`")
