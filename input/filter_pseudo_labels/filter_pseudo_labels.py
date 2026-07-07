# %%
import pandas as pd

# %%
merged_csv_path = '../box_fusion_labels/ensemble/yolov5x_fold0_1_2_3_4_768_conf_0.01_d2_r101fpn3x_054999_vfnet_r101fpn_8020_fold0_1_2_3_4_wbf_skbthr_0.03_v45_submission_rerun.csv'
output_csv_path = '../pseudo_labels_filtered.csv'
keep_percentile = 0.5  # giữ top 50% score cao nhất của mỗi class

# %%
merged_df = pd.read_csv(merged_csv_path)

# %%
# Parse PredictionString ("label score x1 y1 x2 y2 ...") thành 1 dòng / box
rows = []
for _, row in merged_df.iterrows():
    tokens = row['PredictionString'].split(' ')
    for i in range(0, len(tokens), 6):
        rows.append({
            'image_id': row['image_id'],
            'label': tokens[i],
            'score': float(tokens[i + 1]),
            'x_min': tokens[i + 2],
            'y_min': tokens[i + 3],
            'x_max': tokens[i + 4],
            'y_max': tokens[i + 5],
        })
boxes_df = pd.DataFrame(rows)

# %%
# Threshold mỗi class = percentile của chính class đó (data-driven, không hard-code)
class_thresholds = boxes_df.groupby('label')['score'].quantile(keep_percentile)
print('Threshold theo class:')
print(class_thresholds)

# %%
boxes_df['threshold'] = boxes_df['label'].map(class_thresholds)
kept_df = boxes_df[boxes_df['score'] >= boxes_df['threshold']]

# %%
print('\nSố box giữ lại / tổng theo class:')
print(pd.concat([
    boxes_df.groupby('label').size().rename('total'),
    kept_df.groupby('label').size().rename('kept'),
], axis=1))

# %%
# Ghép lại PredictionString theo image_id; ảnh không còn box -> chuỗi rỗng (No finding)
def make_prediction_string(group):
    tokens = []
    for _, r in group.iterrows():
        tokens += [r['label'], str(r['score']), r['x_min'], r['y_min'], r['x_max'], r['y_max']]
    return ' '.join(tokens)

pseudo_df = kept_df.groupby('image_id').apply(make_prediction_string).rename('PredictionString').reset_index()

# Bổ sung lại các image_id bị lọc hết box
all_image_ids = merged_df[['image_id']].drop_duplicates()
pseudo_df = all_image_ids.merge(pseudo_df, on='image_id', how='left')
pseudo_df['PredictionString'] = pseudo_df['PredictionString'].fillna('')

# %%
pseudo_df.to_csv(output_csv_path, index=False)
print(f'\nSố ảnh không còn box nào sau lọc: {(pseudo_df["PredictionString"] == "").sum()} / {len(pseudo_df)}')
print(f'Đã ghi {output_csv_path}')

# %%
