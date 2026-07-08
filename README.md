# Black-box Knowledge Distillation cho VinBigData Chest X-ray Ensemble

## Bài toán

Pipeline hiện tại cho VinBigData Chest X-ray Abnormalities Detection là một ensemble 12 model (Teacher):
- YOLOv5x × 5 folds
- Detectron2 (R101-FPN 3x)
- VFNet (R101-FPN) × 6 (1 full + 5 folds)

Ensemble này cho kết quả tốt nhưng quá nặng để inference trong thực tế. Mục tiêu là dùng
**Black-box KD (Response-based KD / Pseudo-labeling)**: dùng ensemble 12 model làm Teacher để
tạo pseudo-label trên ảnh unlabeled (NIH), sau đó huấn luyện một model Student nhỏ gọn
(YOLO26n) qua 2 giai đoạn.

**Lý do không dùng YOLO26x làm trung gian:** ensemble 12 model đã là Teacher mạnh nhất có thể.
Thêm một tầng YOLO26x trung gian chỉ làm thông tin bị lossy thêm 1 lần, không có lợi.

## Pipeline tổng quan

```
Ensemble 12 model (Teacher — không train, cố định)
         │
         │  inference trên ảnh NIH (5606 ảnh, unlabeled)
         ▼
   WBF merge → Pseudo-label (filtered)
         │
         │  Stage 1: train từ COCO pretrained weights
         ▼
      M_nih  (YOLO26n trained on NIH pseudo-labels)
         │
         │  Stage 2: fine-tune trên VinBigData (strategy TBD)
         ▼
      M_final (YOLO26n trained on VinBigData)
```

## Các bước triển khai

### Bước 1 — Chuẩn bị dữ liệu

Dùng ảnh chưa có nhãn từ [NIH Chest X-rays sample dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
(5606 ảnh) làm nguồn dữ liệu unlabeled. Dataset này khác taxonomy với VinBigData nhưng cùng
domain (chest X-ray) nên phù hợp để Teacher sinh nhãn.

**Script:** `create_nih_test_meta.py`
- Input: `/kaggle/input/sample/images/*.png`
- Output: `test_meta.csv` (`image_id`, `dim0`=height, `dim1`=width)
- `test_meta.csv` là input bắt buộc cho bước merge (dùng để scale bbox normalized → pixel)

### Bước 2 — Teacher Inference

Gồm 2 phần:

**(a) Chạy inference 12 model riêng lẻ** (checkpoint từ
[vinbigdata-final-models](https://www.kaggle.com/datasets/kc3222/vinbigdata-final-models))
trên ảnh NIH. Mỗi model xuất 1 file CSV format:
```
image_id, PredictionString
```
trong đó `PredictionString = "label score x_min y_min x_max y_max ..."` (pixel coordinates).

**(b) WBF Merge** bằng `merge_bboxes_v45_rerun.py`:
- Tách riêng class lớn (`[0,1,3,4,12]`, `iou_thr=0.4`) và class thường (`iou_thr=0.5`)
- `skip_box_thr=0.03` để loại box gần-zero trước khi fuse (không phải threshold pseudo-label)
- Output: 1 file CSV merged với fused scores

**Lưu ý:** Merge trước, lọc sau — fused score đáng tin hơn score riêng lẻ vì tích hợp
đồng thuận của cả 12 model.

### Bước 3 — Pseudo-label Filtering

Lọc trên `final_scores` sau WBF. **Threshold nên per-class** vì phân phối score chênh lệch
lớn giữa các class (VD class 3 p90≈0.62, class 9 p90≈0.07 trên dữ liệu VinBigData test).

**Quy trình chọn threshold:**
1. Chạy `plot_score_histograms.py` trên output NIH thật (sau khi có bước 2)
2. Xem histogram per-class, chọn ngưỡng tại điểm phân phối "gãy" giữa cụm noise và cụm signal
3. Điền vào `class_thresholds` dict trong `filter_pseudo_labels.py`

**Ảnh không còn box nào sau lọc** → giữ nguyên (tương đương "No finding", hợp lệ cho training)

### Bước 4 — Student Training: Stage 1 (NIH)

Train YOLO26n từ COCO pretrained weights trên ảnh NIH + pseudo-label.

| Tham số | Giá trị đề xuất | Ghi chú |
|---|---|---|
| `model` | `yolo11n.pt` hoặc tương đương YOLO26n | Khởi đầu từ COCO pretrained |
| `data` | NIH pseudo-label dataset | 15 classes VinBigData taxonomy |
| `epochs` | 50–100 | Dừng sớm nếu val loss plateau |
| `imgsz` | 640 | Tăng lên 768 nếu GPU cho phép |

**Output:** `M_nih` — checkpoint YOLO26n đã học feature chest X-ray từ pseudo-label.

### Bước 5 — Student Training: Stage 2 (VinBigData fine-tune)

Fine-tune `M_nih` trên VinBigData (dữ liệu có nhãn thật).

| Strategy | Mô tả | Khi nào dùng |
|---|---|---|
| **Freeze backbone** | Freeze backbone N epoch đầu, unfreeze sau | M_nih mAP thấp, tránh catastrophic forgetting |
| **Unfreeze toàn bộ** | Fine-tune toàn bộ weights từ đầu | M_nih mAP đã ổn, muốn adapt nhanh |

→ **Cần đánh giá `M_nih` trên VinBigData val trước khi chọn strategy.**

| Tham số | Giá trị đề xuất | Ghi chú |
|---|---|---|
| `model` | `M_nih` checkpoint | Không dùng COCO pretrained lại |
| `lr0` | 1e-4 (nhỏ hơn Stage 1) | Tránh overwrite feature đã học |
| `epochs` | 30–50 | Ít hơn Stage 1 vì đã có warm start |

**Output:** `M_final` — model deploy được, thay thế ensemble 12 model.

## Thử nghiệm dự kiến

| Experiment | Mô tả | Metric |
|---|---|---|
| E1 | Baseline: train YOLO26n trực tiếp trên VinBigData (không KD) | mAP@0.4 VinBigData val |
| E2 | Stage 1 only: M_nih eval trên VinBigData val | mAP@0.4 — kiểm tra domain transfer |
| E3 | Stage 1 → Stage 2 (freeze backbone) | mAP@0.4 VinBigData val |
| E4 | Stage 1 → Stage 2 (unfreeze toàn bộ) | mAP@0.4 VinBigData val |

E1 là baseline bắt buộc — nếu E3/E4 không beat E1 thì pipeline KD không có giá trị thực tế.

## File trong repo

| File | Mô tả | Trạng thái |
|---|---|---|
| `create_nih_test_meta.py` | Tạo `test_meta.csv` cho ảnh NIH sample | Đã viết |
| Inference script 12 model trên NIH | Sinh 12 CSV input cho merge | Done |
| `merge_bboxes_v45_rerun.py` | WBF fuse output 12 model Teacher | Done |
| `plot_score_histograms.py` | Vẽ histogram score per-class để chọn threshold |Done |
| `filter_pseudo_labels.py` | Lọc pseudo-label theo per-class threshold | Done |
| Training script Stage 1 (NIH) | Train YOLO26n trên pseudo-label | Đang hoàn thiện |
| Training script Stage 2 (VinBigData) | Fine-tune M_nih trên VinBigData | Chưa làm |