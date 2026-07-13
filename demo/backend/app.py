import io
import base64

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# === Config — sửa lại theo model bạn muốn dùng ===
WEIGHTS_PATH = "weights/best.pt"   # đường dẫn tới model .pt đã train
CONF_THRESHOLD = 0.01
IOU_THRESHOLD = 0.5

app = FastAPI(title="VinBigData Chest X-ray Detection Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(WEIGHTS_PATH)


@app.get("/")
def health():
    return {"status": "ok", "weights": WEIGHTS_PATH}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    result = results[0]

    # Ảnh đã vẽ bbox + label + confidence (built-in Ultralytics)
    plotted_bgr = result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    plotted_pil = Image.fromarray(plotted_rgb)

    buffer = io.BytesIO()
    plotted_pil.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Bảng bệnh sắp xếp theo confidence giảm dần
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "class": model.names[cls_id],
            "confidence": round(conf, 4),
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "image_base64": image_base64,
        "detections": detections,
    }
