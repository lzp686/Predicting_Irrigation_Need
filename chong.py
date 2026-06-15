import pandas as pd
import numpy as np

sub_high = pd.read_csv('./output/submission_part2_0.981.csv')
df74     = pd.read_csv('./data/pd74/5(4) - 0.98074.csv')

diff_mask = sub_high['Irrigation_Need'].values != df74['Irrigation_Need'].values
diff_idx  = np.where(diff_mask)[0]

# 找出那8行：你=Medium, df74=Low
pred_high_vals = sub_high['Irrigation_Need'].values[diff_idx]
pred_74_vals   = df74['Irrigation_Need'].values[diff_idx]

mask_8 = (pred_high_vals == 'Medium') & (pred_74_vals == 'Low')
idx_8  = diff_idx[mask_8]  # 原始行索引

print(f"找到 {len(idx_8)} 行，索引: {idx_8}")

sub_new = sub_high.copy()
sub_new.loc[idx_8, 'Irrigation_Need'] = 'Low'

# 验证
changed = (sub_new['Irrigation_Need'].values != sub_high['Irrigation_Need'].values).sum()
print(f"实际修改行数: {changed}")
print(f"修改前分布:\n{sub_high['Irrigation_Need'].value_counts()}")
print(f"修改后分布:\n{sub_new['Irrigation_Need'].value_counts()}")

sub_new.to_csv('./output/submission_flip8.csv', index=False)
print("\n已保存: ./output/submission_flip8.csv")