import streamlit as st
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import os
from FAISS import build_faiss_index
import faiss
import tempfile
import time

# --- Cấu hình ---
DB_PATH = "database"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Mô hình ---
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

# --- Streamlit UI ---
st.title("🤖 Nhận diện khuôn mặt tự động (Webcam Realtime)")

run = st.checkbox("📷 Bật camera", value=False)

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    st.info("🧠 Hệ thống sẽ tự động chụp khi phát hiện khuôn mặt.")

    last_capture_time = 0
    capture_interval = 3  # chỉ chụp lại sau 3 giây để tránh spam

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Không mở được webcam!")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)

        boxes, probs, landmarks_all = mtcnn.detect(img_pil, landmarks=True)

        # Vẽ khung mặt
        if boxes is not None:
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            # Nếu phát hiện khuôn mặt và đủ thời gian giữa 2 lần chụp
            if time.time() - last_capture_time > capture_interval:
                last_capture_time = time.time()

                # Crop khuôn mặt đầu tiên
                x1, y1, x2, y2 = boxes[0]
                face_pil = img_pil.crop((x1, y1, x2, y2)).resize((160, 160))

                # Tính embedding
                face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).float() / 255.0
                face_tensor = (face_tensor.unsqueeze(0).to(device) * 2) - 1
                query_emb = resnet(face_tensor).detach().cpu().numpy()

                # Tìm người gần nhất
                D, I = index.search(query_emb.astype('float32'), 1)
                matched_path = filenames[I[0][0]]
                distance = float(D[0][0])
                person_name = os.path.splitext(os.path.basename(matched_path))[0]
                threshold = 0.9

                if distance < threshold:
                    label = f"{person_name} ({distance:.2f})"
                    color = (0,255,0)
                else:
                    label = f"Unknown ({distance:.2f})"
                    color = (0,0,255)

                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
else:
    st.warning("👆 Tick vào ô 'Bật camera' để khởi động nhận diện.")
