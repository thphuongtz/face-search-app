import numpy as np
import faiss
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo MTCNN và ResNet
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# --- Hàm lấy embedding từ tensor mặt đã crop ---
def get_face_embedding(face_tensor, resnet_model=resnet):
    """
    face_tensor: tensor [3, H, W] hoặc [1, 3, H, W]
    resnet_model: InceptionResnetV1 instance
    """
    if face_tensor is None:
        return None

    # đảm bảo tensor có batch dimension
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = resnet_model(face_tensor.to(device))  # [1, 512]

    return embedding.squeeze().cpu().numpy()  # [512]


# --- Hàm crop mặt từ file ảnh ---
def crop_face_from_file(img_path):
    img = Image.open(img_path).convert("RGB")
    img_cropped = mtcnn(img)  # trả về tensor hoặc None
    return img_cropped


# --- Hàm xây dựng FAISS index từ file ảnh ---
def build_faiss_index(image_files):
    embeddings = []
    filenames = []

    for img_path in image_files:
        face_tensor = crop_face_from_file(img_path)
        emb = get_face_embedding(face_tensor)
        if emb is not None:
            embeddings.append(emb)
            filenames.append(img_path)

    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, filenames
