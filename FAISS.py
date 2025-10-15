import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_face_embedding(image):
    """Phát hiện khuôn mặt và tạo vector đặc trưng (giả lập 512 chiều)."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)
    if not results.detections:
        return None, None

    h, w, _ = image.shape
    boxes, embeddings = [], []

    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        boxes.append((x1, y1, x2, y2))

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        mean_rgb = np.mean(face_crop, axis=(0, 1)) / 255.0
        emb = np.pad(mean_rgb, (0, 512 - len(mean_rgb)), mode='constant')
        embeddings.append(emb)

    return boxes, np.array(embeddings)
