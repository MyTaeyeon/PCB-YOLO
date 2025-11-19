# 8) Infer realtime: FastAPI + engine

Hai lựa chọn inference server:

* **ONNX Runtime** (dễ triển khai, CPU/GPU via providers) — good for dev/CPU fallback.
* **TensorRT engine** (tốt nhất cho production GPU low latency).

Mẫu `infer.py` (onnxruntime, CPU/GPU):

```python
import cv2, onnxruntime as ort, numpy as np
from utils import preprocess, postprocess   # bạn tự viết theo model

sess = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

def run_infer(image):
    img = preprocess(image)            # resize, normalize, to NCHW float32
    outputs = sess.run(None, {input_name: img})
    boxes, scores, classes = postprocess(outputs)
    return boxes, scores, classes
```

Mẫu FastAPI (minimal):

```python
from fastapi import FastAPI, File, UploadFile
import uvicorn, cv2, numpy as np
app = FastAPI()

@app.post('/detect')
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    boxes, scores, classes = run_infer(img)
    return {'boxes': boxes.tolist(), 'scores': scores.tolist(), 'classes': classes.tolist()}

# run: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Nếu dùng TensorRT engine file `.trt`, bạn sẽ cần code loader bằng TensorRT Python API (tạo context, allocate device memory, enqueue), hoặc dùng prebuilt helper libs (nhiều repo demo trên GitHub). Ví dụ tham khảo cách load engine + inference trên Python có nhiều guide/StackOverflow threads. ([Stack Overflow][5])

---




# 9) Tối ưu hóa tốc độ & Precision

* FP16 inference (TensorRT FP16) giảm latency. Dùng `--fp16` when building engine.
* Int8 còn nhanh hơn nhưng cần calibration dataset.
* NMS & postprocess trên CPU hoặc kernel tối ưu — tránh copy device↔host nhiều lần.
* Batch inference (nếu throughput > latency yêu cầu).
* Tối ưu kích thước ảnh: tradeoff giữa AP và FPS. Thử 512/640/1024.
* Dùng TensorRT builder flags: max workspace, allow GPU fallback for unsupported nodes. ([NVIDIA Docs][3])

---

# 10) Các scripts bạn sẽ cần (tổng hợp)

* `train.py` — huấn luyện & logging.
* `evaluate.py` — chạy val, tạo confusion matrix, per-class AP.
* `export_onnx.py` — export và validate onnx.
* `convert_trt.sh` — dùng `trtexec` hoặc script python build engine.
* `infer_server/` — FastAPI server + model loader.
* `README.md` — cách chạy, requirements, tip h/w.

---

# 11) Ví dụ cấu trúc final & lệnh mẫu

```
pcb_defect_detection/
├── data/
├── yolo_config/
│    └── data.yaml
├── train.py
├── export_onnx.py
├── convert_trt.sh
├── deploy_fastapi/
│    ├── app.py
│    └── requirements.txt
└── README.md
```

Lệnh training nhanh (CLI):

```bash
yolo task=detect mode=train model=yolov8m.pt data=yolo_config/data.yaml epochs=100 imgsz=640 batch=16
```

Export & convert:

```bash
python export_onnx.py --weights runs/detect/exp/weights/best.pt --out model.onnx
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 --workspace=4096
```

(kiểm tra các flag `--verbose` để debug).

---

# 12) Common pitfalls & fixes

* **ONNX exported results differ from .pt**: kiểm tra opset, dynamic axes, simplify; validate on same sample. (GitHub issues và threads có nhiều case). ([GitHub][6])
* **trtexec fails**: kiểm tra dtype (double vs float), unsupported ops, workspace size, ONNX opset. Logs trtexec thường chỉ ra nguyên nhân. ([GitHub][7])
* **Small object miss**: increase image size, use mosaic, increase detector depth/width, tile inference (crop with overlap).
* **Imbalanced classes**: oversample rare classes, weighted loss, focal loss.

---

# 13) KPI & kiểm thử (what to report)

* mAP@0.5, mAP@[0.5:0.95], per-class AP, precision, recall, F1.
* FPS measured: (a) PyTorch baseline, (b) ONNXRuntime (GPU), (c) TensorRT engine (FP16).
* Confusion matrix and ROC/pr curves.
* Demo video: record webcam/RTSP running FastAPI+inference and overlay boxes.

---

# 14) Tài nguyên & tham khảo (nhanh)

* Ultralytics export & train docs (export to ONNX/TensorRT). ([Ultralytics Docs][1])
* NVIDIA TensorRT Quick Start / trtexec usage. ([NVIDIA Docs][3])
* ONNX <-> TensorRT tools / onnx-tensorrt. ([GitHub][8])
* Các discussion / troubleshooting issues trên GitHub & StackOverflow (ONNX export problems & trtexec). ([GitHub][6])

---

# 15) Next actions — checklist bạn có thể thực hiện ngay (pick and run)

1. Kiểm tra `data.yaml` và sample vài image & label (script quick-check).
2. Chạy 1 epoch nhanh với `yolov8n` để verify pipeline.
3. Thiết lập augmentation pipeline (albumentations) và bật `augment=True`.
4. Train full (monitor mAP).
5. Export best.pt → ONNX → validate onnx with onnxruntime on a few images.
6. Convert ONNX → TensorRT (trtexec), test latency.
7. Build FastAPI server và record demo video.

---

Nếu bạn muốn, mình có thể:

* Viết cho bạn `train.py`, `export_onnx.py`, `infer_fastapi.py` mẫu hoàn chỉnh (có logging & checkpoints) theo cấu trúc project trên — mình sẽ gửi code ngay.
* Hoặc bắt đầu bằng **script kiểm tra dataset & tạo data.yaml** và script quick-EDA (số ảnh, bounding box sizes distribution, small vs large objects) — rất hữu ích trước khi train.

Bạn muốn mình **bắt tay vào viết các file code mẫu** (ví dụ `train.py` + `export_onnx.py` + FastAPI) luôn không? Nếu có, mình sẽ gửi full code ngay ở lần trả lời tiếp theo.

[1]: https://docs.ultralytics.com/modes/export/?utm_source=chatgpt.com "Model Export with Ultralytics YOLO"
[2]: https://docs.ultralytics.com/modes/train/?utm_source=chatgpt.com "Model Training with Ultralytics YOLO"
[3]: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html?utm_source=chatgpt.com "Quick Start Guide — NVIDIA TensorRT Documentation"
[4]: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html?utm_source=chatgpt.com "TensorRT Execution Provider - NVIDIA - ONNX Runtime"
[5]: https://stackoverflow.com/questions/59280745/inference-with-tensorrt-engine-file-on-python?utm_source=chatgpt.com "Inference with TensorRT .engine file on python - Stack Overflow"
[6]: https://github.com/ultralytics/ultralytics/issues/14320?utm_source=chatgpt.com "Export - Ultralytics YOLOv8 model to ONNX · Issue #14320"
[7]: https://github.com/NVIDIA/TensorRT/issues/3894?utm_source=chatgpt.com "trtexec: onnx to tensorRT convertion fails · Issue #3894"
[8]: https://github.com/onnx/onnx-tensorrt?utm_source=chatgpt.com "TensorRT backend for ONNX - GitHub"
