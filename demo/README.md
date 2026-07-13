# Demo: VinBigData Chest X-ray Detection

FastAPI backend chạy YOLO inference + frontend HTML/JS đơn giản.

## Cách chạy

### 1. Cài dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Đặt model weights
Copy file `best.pt` (model đã train) vào `backend/weights/best.pt`,
hoặc sửa `WEIGHTS_PATH` trong `backend/app.py` trỏ đến đường dẫn khác.

### 3. Chạy backend
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```
Backend chạy tại `http://localhost:8000`. Kiểm tra `http://localhost:8000` trả về `{"status": "ok"}`.

### 4. Mở frontend
Mở trực tiếp file `frontend/index.html` bằng browser (double click hoặc `open frontend/index.html`).

Nếu browser chặn do CORS/local file, chạy static server đơn giản:
```bash
cd frontend
python3 -m http.server 5500
```
Rồi mở `http://localhost:5500`.

## Sử dụng
1. Kéo thả hoặc chọn ảnh X-ray
2. Nhấn "Dự đoán"
3. Ảnh có bounding box + confidence hiển thị, bảng bệnh bên dưới sắp xếp theo độ tin cậy giảm dần

## Cấu hình
Trong `backend/app.py`:
- `WEIGHTS_PATH`: đường dẫn model `.pt`
- `CONF_THRESHOLD`: ngưỡng confidence tối thiểu để hiển thị (default 0.25)
- `IOU_THRESHOLD`: ngưỡng NMS IoU (default 0.5)
