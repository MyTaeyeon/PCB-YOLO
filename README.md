# 🚀 **PROJECT 1 – Hệ thống phát hiện lỗi PCB bằng YOLO (Core AOI Project)**

✔ **Dự án mạnh nhất – giống 80% công việc thực tế**

## 🎯 Mục tiêu

* Phát hiện 4–8 loại lỗi PCB: missing component, solder bridge, misalignment, short circuit.
* Train YOLOv8/YOLO11.
* Đạt F1 > 0.9 và FPS > 15.
* Deploy ONNX + TensorRT.

## 🧠 Kỹ thuật bạn sẽ học

* Tiền xử lý ảnh công nghiệp
* Gán nhãn bằng LabelImg / CVAT
* Augmentation ánh sáng kém (important)
* Training YOLOv8/11
* ONNX export
* TensorRT optimization
* FastAPI real-time inference

## 📁 Cấu trúc project

```
pcb_defect_detection/
 ├── data/ (raw + processed)
 ├── labels/ 
 ├── yolo_config/
 ├── train.py
 ├── infer.py
 ├── export_onnx.py
 ├── deploy_fastapi/
 └── README.md
```

## 💡 Kết quả mong đợi

* 1 video demo phát hiện lỗi real-time
* Biểu đồ mAP, confusion matrix
* FPS trước và sau khi tối ưu

---

# 🚀 **PROJECT 2 – Phát hiện đứt mạch, lệch linh kiện bằng OpenCV (Classical CV)**

✔ Nhiều công ty AOI rất thích ứng viên **biết cả classical + deep learning**

## 🎯 Mục tiêu

Xây hệ thống classical CV để:

* Phát hiện đường mạch bị **đứt**
* Phát hiện linh kiện **dịch chuyển**
* So khớp template linh kiện chuẩn

## 🧠 Kỹ thuật bạn sẽ học

* Thresholding (Otsu, adaptive)
* Edge detection (Canny/Sobel)
* Morphology (opening/closing)
* Contour analysis
* Template matching
* Calculating displacement (pixel → mm)

## 📁 Cấu trúc

```
pcb_opencv_inspection/
 ├── data/
 ├── preprocess.py
 ├── detect_break.py
 ├── detect_shift.py
 ├── template_match.py
 └── README.md
```

## 💡 Kết quả

* Ảnh trước/sau xử lý
* Detect vết đứt mạch qua contour
* Detect linh kiện lệch bằng cross-correlation

---

# 🚀 **PROJECT 3 – CCTV AI giám sát hành vi công nhân (phù hợp phần JD mục 6)**

✔ Họ ghi rõ: “Dự án CCTV AI giám sát hành vi công nhân” → bạn làm project này là ăn điểm ngay.

## 🎯 Mục tiêu

* Xây hệ thống CCTV AI detect:

  * Không đội mũ bảo hộ
  * Vào vùng nguy hiểm
  * Ngồi/nằm trong giờ làm
* Pose estimation + object detection

## 🧠 Kỹ thuật dùng:

* YOLOv8n/11n
* YOLO-Pose (pose estimation)
* Rule-based behavior detection
* Line crossing detection (vào vùng cấm)
* Tracking bằng ByteTrack

## 📁 Cấu trúc:

```
cctv_worker_safety/
 ├── datasets/
 ├── detect_helmet.py
 ├── detect_pose.py
 ├── track.py
 ├── roi_zone.json
 ├── rule_engine.py
 └── README.md
```

## 💡 Kết quả:

* Video demo
* Detect worker không đội mũ
* Warning khi vào vùng nguy hiểm

---

# 🚀 **PROJECT 4 – Deploy mô hình AI real-time lên Jetson Nano / Raspberry Pi**

✔ Đây là skill **rất ít ứng viên có** → tạo lợi thế lớn.

> Nếu bạn không có Jetson thì làm “Jetson Simulation” (ONNX + TensorRT) vẫn được.

## 🎯 Mục tiêu

* Deploy YOLO từ PyTorch → ONNX → TensorRT
* Chạy real-time webcam ~15–25 FPS
* Code inference pipeline rất gọn

## 🧠 Kỹ thuật học

* ONNX export
* TensorRT runtime
* Real-time inference loop
* Optimization: FP16, INT8
* Benchmark FPS

## 📁 Cấu trúc:

```
edge_deploy_yolo/
 ├── model/
 ├── export_onnx.py
 ├── build_tensorrt.py
 ├── infer_trt.py
 └── README.md
```

## 💡 Kết quả:

* Bảng so sánh tốc độ:
  PyTorch vs ONNX vs TensorRT
* Demo FPS real-time

---

# 🎁 BONUS (Nếu bạn muốn mở rộng portfolio)

> Nhưng 4 project trên đã quá đủ cho 1 portfolio mạnh.

* Project 5: Robot mô phỏng (Unity) + CV
* Project 6: Defect segmentation bằng U-Net
* Project 7: Hệ thống phân loại chất lượng sản phẩm (OK/NG)

---

# 🎯 Bạn nên làm theo thứ tự thế nào?

**Thứ tự tối ưu để tạo portfolio mạnh nhất trong mắt nhà tuyển dụng AOI:**

1️⃣ Project 1 – PCB Defect Detection (YOLO)
2️⃣ Project 2 – PCB Inspection bằng OpenCV
3️⃣ Project 4 – Deploy real-time (TensorRT)
4️⃣ Project 3 – CCTV Worker Monitoring

→ Làm đúng thứ tự này, bạn sẽ có portfolio ngang trình **1–2 năm kinh nghiệm**.

---

# ✔ Nếu bạn muốn, mình sẽ tạo cho bạn:

* Toàn bộ **4 README chuẩn** để up lên GitHub
* Template folder đầy đủ
* Hướng dẫn từng bước
* Dataset đề xuất
* To-do list chi tiết theo ngày

Bạn muốn mình **chi tiết hóa project số 1 trước** hay **làm roadmap 4 project luôn**?
