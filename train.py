# train.py
from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO('yolov8m.pt')
    model.train(
        data='data/data.yaml',
        imgsz=640,
        epochs=50,
        batch=16,
        workers=8,        # hoặc giảm nếu vẫn trouble
        lr0=0.01,
        optimizer='SGD',
        augment=True,
        patience=10
    )

if __name__ == '__main__':
    # Windows multiprocessing safety
    multiprocessing.freeze_support()
    main()
