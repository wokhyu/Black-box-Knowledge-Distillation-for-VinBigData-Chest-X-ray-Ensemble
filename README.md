# Black-box Knowledge Distillation — VinBigData Chest X-ray Detection

Compress một ensemble 12 model Teacher (YOLOv5x×5 + Detectron2 + VFNet×6) thành
Student YOLO26n/x nhỏ gọn bằng pseudo-labeling trên ảnh NIH Chest X-ray.

## Pipeline tổng quan

```
Ensemble 12 model (Teacher — cố định)
        │
        │  inference trên NIH sample (5606 ảnh unlabeled)
        ▼
   WBF merge → Pseudo-label (filtered p50)
        │
        │  Stage 1: pretraining trên NIH pseudo-label
        ▼
     M_nih  (YOLO26n/x)
        │
        │  Stage 2: fine-tune trên VinBigData (nhãn thật)
        ▼
     M_final → so sánh với 5-fold ensemble baseline
```

## Experiments

| ID | Model | Strategy | mAP@0.4 |
|----|-------|----------|---------|
| E1 | YOLO26n/x | Baseline: direct fine-tune VinBigData | - |
| E2 | YOLO26n/x | NIH pretraining only (domain transfer check) | - |
| E3 | YOLO26n/x | Stage 1 NIH → Stage 2 freeze backbone → unfreeze | - |
| E4 | YOLO26n/x | Stage 1 NIH → Stage 2 full unfreeze | - |
| KFold | YOLO26x | 5-fold ensemble (baseline có sẵn) | - |

## Datasets (Kaggle)
- **Teacher checkpoints**: [vinbigdata-final-models](https://www.kaggle.com/datasets/kc3222/vinbigdata-final-models)
- **Unlabeled (NIH)**: [nih-chest-xrays/sample](https://www.kaggle.com/datasets/nih-chest-xrays/sample)
- **VinBigData images**: [vinbigdata-1024-image-dataset](https://www.kaggle.com/datasets/awsaf49/vinbigdata-1024-image-dataset)
- **VinBigData labels**: [vinbigdata-yolo-labels-dataset](https://www.kaggle.com/datasets/awsaf49/vinbigdata-yolo-labels-dataset)

## Cách chạy

### Bước 1 — Tạo pseudo-label NIH
```
pseudo_label/create_nih_test_meta.py        # tạo test_meta.csv
# chạy Teacher inference (12 model) → 12 CSV
teacher/merge_bboxes_v45_rerun.py           # WBF merge
pseudo_label/plot_score_histograms.py       # chọn threshold (p50)
pseudo_label/prepare_nih_yolo_labels.ipynb  # convert → YOLO .txt
```

### Bước 2 — Stage 1: NIH pretraining
```
student/yolo26n/stage1/train_yolo26n_nih_stage1.ipynb
student/yolo26x/stage1/train_yolo26x_nih_stage1.ipynb
```

### Bước 3 — Stage 2: VinBigData fine-tune
```
student/yolo26n/stage2/train_yolo26n_vinbig_stage2_e3.ipynb  # E3
student/yolo26n/stage2/train_yolo26n_vinbig_stage2_e4.ipynb  # E4
student/yolo26x/stage2/train_yolo26x_vinbig_stage2_e3.ipynb  # E3
student/yolo26x/stage2/train_yolo26x_vinbig_stage2_e4.ipynb  # E4
```


## Kết quả

Xem `results/metrics_summary.csv`