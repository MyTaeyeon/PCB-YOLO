import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from trt_infer import TRTInfer

# import hoặc viết 2 hàm này theo model của bạn
# preprocess: BGR image -> NCHW numpy array (dtype np.float16 or np.float32) matching engine
# postprocess: outputs (list of numpy arrays) -> boxes, scores, classes
from utils import preprocess, postprocess

ENGINE_PATH = r"C:\Users\Default\Giang_space\meiko\PCB-YOLO\runs\detect\train\weights\model_fp16.trt"

app = FastAPI()
trt_runner = TRTInfer(ENGINE_PATH)

@app.on_event("startup")
async def startup_event():
    # có thể warmup model 1 lần (tùy chọn)
    print("Starting up FastAPI + TRT engine")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "cannot decode image"}

    # preprocess the image to engine input format
    # should return numpy array: shape (N,C,H,W) or (C,H,W) depending on engine
    inp = preprocess(img_bgr)  # user-provided: must match engine dtype and expected shape
    # ensure contiguous and correct dtype
    inp = np.ascontiguousarray(inp)

    # call blocking inference in threadpool to avoid blocking event loop
    outputs = await asyncio.to_thread(trt_runner.infer, inp)

    # outputs is list of numpy arrays (raw model outputs)
    boxes, scores, classes = postprocess(outputs)  # user-provided postprocess

    # convert numpy types to python lists
    boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
    scores_list = scores.tolist() if isinstance(scores, np.ndarray) else scores
    classes_list = classes.tolist() if isinstance(classes, np.ndarray) else classes

    return {"boxes": boxes_list, "scores": scores_list, "classes": classes_list}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
# Note: use --workers 1 when using GPU, multiworkers will create multiple separate processes (and duplicate GPU contexts)
