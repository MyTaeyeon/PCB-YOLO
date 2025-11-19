import cv2
import numpy as np

def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (640, 640))
    img = img[..., ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img.astype(np.float16)

def postprocess(outputs):
    # outputs = list các tensor raw của YOLO (tùy modele bạn export)
    # Tạm thời trả rỗng để test server chạy được
    return [], [], []
