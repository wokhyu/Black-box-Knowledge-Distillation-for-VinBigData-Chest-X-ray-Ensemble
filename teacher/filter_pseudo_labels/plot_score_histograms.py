# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
merged_csv_path = './yolov5x_fold0_1_2_3_4_768_conf_0.01_d2_r101fpn3x_054999_vfnet_r101fpn_8020_fold0_1_2_3_4_wbf_skbthr_0.03_v45_submission_rerun.csv'
output_fig_path = './artifacts/score_histograms.png'

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
            'label': int(tokens[i]),
            'score': float(tokens[i + 1]),
        })
boxes_df = pd.DataFrame(rows)

# %%
# Histogram tổng
plt.figure(figsize=(8, 5))
plt.hist(boxes_df['score'], bins=50)
plt.xlabel('score')
plt.ylabel('count')
plt.title('Score distribution (all classes)')
plt.savefig('./artifacts/score_histogram_overall.png', dpi=150, bbox_inches='tight')
plt.close()

# %%
# Histogram theo từng class
labels = sorted(boxes_df['label'].unique())
n_cols = 4
n_rows = -(-len(labels) // n_cols)  # ceil division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.flatten()
for ax, label in zip(axes, labels):
    scores = boxes_df.loc[boxes_df['label'] == label, 'score']
    ax.hist(scores, bins=30)
    ax.set_title(f'class {label} (n={len(scores)})')
for ax in axes[len(labels):]:
    ax.axis('off')
plt.tight_layout()
plt.savefig(output_fig_path, dpi=150, bbox_inches='tight')
plt.close()

# %%
# Bảng percentile theo class để hỗ trợ chọn threshold
percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
summary = boxes_df.groupby('label')['score'].describe(percentiles=percentiles)
print(summary)
summary.to_csv('./artifacts/score_summary_by_class.csv')

# %%
