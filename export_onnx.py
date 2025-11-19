from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/train/weights/best.pt')

# Export ONNX
model.export(
    format='onnx',
    opset=13,        
    simplify=True,   
    verbose=True
)
