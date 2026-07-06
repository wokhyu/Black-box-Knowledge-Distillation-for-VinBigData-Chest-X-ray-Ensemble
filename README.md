# Black-box Knowledge Distillation cho VinBigData Chest X-ray Ensemble

## Bài toán

Pipeline hiện tại cho VinBigData Chest X-ray Abnormalities Detection là một ensemble 12 model (Teacher):
- YOLOv5x × 5 folds
- Detectron2 (R101-FPN 3x)
- VFNet (R101-FPN) × 6 (1 full + 5 folds)

Ensemble này cho kết quả tốt nhưng quá nặng để inference trong thực tế. Mục tiêu là dùng
**Black-box KD (Response-based KD)**: dùng ensemble 12 model làm Teacher để tạo pseudo-label,
sau đó huấn luyện một model Student nhỏ gọn (YOLO26n) bắt chước đầu ra của Teacher.

## Các bước triển khai

1. **Chuẩn bị dữ liệu**
   Dùng ảnh chưa có nhãn từ [NIH Chest X-rays sample dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
   (5606 ảnh) làm nguồn dữ liệu unlabeled cho quá trình pseudo-labeling.

   - `create_nih_test_meta.py`: đọc kích thước (`dim0`=height, `dim1`=width) của từng ảnh
     trong `/kaggle/input/sample/images`, xuất ra `test_meta.csv` — file này là input bắt buộc
     cho bước merge ở dưới (dùng để scale bbox normalized về pixel).

2. **Thu thập tri thức từ Teacher (Teacher Inference)**
   Gồm 2 phần:
   - (a) Chạy inference riêng lẻ 12 model (checkpoint từ
     [vinbigdata-final-models](https://www.kaggle.com/datasets/kc3222/vinbigdata-final-models))
     trên ảnh NIH, mỗi model xuất ra 1 file CSV dạng `image_id, PredictionString`
     (label, score, x_min, y_min, x_max, y_max theo pixel). *(chưa thực hiện)*
   - (b) Fuse 12 file CSV đó bằng `merge_bboxes_v45_rerun.py` (Weighted Boxes Fusion, tách
     riêng nhóm class lớn/nhỏ với `iou_thr` khác nhau) để ra 1 kết quả Bboxes/Classes/Scores
     hợp nhất cho mỗi ảnh.

3. **Lọc và tạo Pseudo-label**
   Lọc bỏ các box có score dưới ngưỡng để giảm nhiễu trước khi dùng làm nhãn huấn luyện.
   *(chưa thực hiện)*

4. **Huấn luyện Student**
   Dùng ảnh NIH + pseudo-label vừa tạo để train YOLO26n. *(chưa thực hiện)*

## File trong repo

| File | Mô tả | Trạng thái |
|---|---|---|
| `create_nih_test_meta.py` | Tạo `test_meta.csv` (kích thước ảnh) cho tập NIH sample | Đã viết |
| Inference script cho 12 model trên ảnh NIH | Sinh 12 CSV đầu vào cho bước merge | Đã có các file inference |
| `merge_bboxes_v45_rerun.py` | Fuse output của 12 model Teacher bằng WBF | Có sẵn |
| Script lọc pseudo-label theo score threshold | Bước 3 | Chưa làm |
| Training script cho Student (YOLO26n) | Bước 4 | Chưa làm |
